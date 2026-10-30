"""Owned, verified checkpoint spill for the single perception_v2 queued run.

Only completed steps 25/50/75 are moved. The final step stays on the data disk.
tick/finalize are nonblocking: call again while their report has pending=True.
No generalist, old teacher, or unrecognized external symlink is modified.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
import uuid

EXPERIMENT = Path(__file__).resolve().parents[1]
EXPECTED_RUN = EXPERIMENT / "runs/perception_v2"
GENERALIST_RUN = EXPERIMENT / "runs/generalist"
BASE_MODEL = Path("/root/autodl-tmp/sharedexpert/models/Qwen3.5-4B")
TMP_ROOT = Path("/root/autodl-tmp/sharedexpert/perception_v2_checkpoint_spill")
OVERLAY_ROOT = Path("/root/.cache/sharedexpert/perception_v2_checkpoint_spill")
RESERVE_BYTES = 3 * 1024**3
MODEL0 = "model_world_size_2_rank_0.pt"
MODEL1 = "model_world_size_2_rank_1.pt"
OPTIM0 = "optim_world_size_2_rank_0.pt"
OPTIM1 = "optim_world_size_2_rank_1.pt"
SPILL_FILES = (OPTIM0, MODEL0, MODEL1)
CHECKPOINT_FILES = (MODEL0, MODEL1, OPTIM0, OPTIM1)
MANIFEST = "perception_v2_storage_manifest.json"
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="perception-v2-storage")
_JOBS = {}
_JOBS_LOCK = threading.Lock()


class StorageCapacityError(RuntimeError):
    pass


class StorageOwnershipError(RuntimeError):
    pass


class _RetiredCheckpoint(RuntimeError):
    pass


def _disk(path):
    path = Path(path)
    while not path.exists():
        path = path.parent
    info = os.statvfs(path)
    return {"free_bytes": info.f_bavail * info.f_frsize, "device": path.stat().st_dev}


def _measure():
    marker = GENERALIST_RUN / "checkpoints/latest_checkpointed_iteration.txt"
    if not marker.is_file() or not marker.read_text().strip().isdigit():
        raise StorageCapacityError("Cannot measure a completed generalist checkpoint")
    actor = marker.parent / ("global_step_" + marker.read_text().strip()) / "actor"
    if not all((actor / name).is_file() for name in CHECKPOINT_FILES):
        raise StorageCapacityError("Completed generalist checkpoint shards are unavailable for sizing")
    sizes = {name: (actor / name).stat().st_size for name in CHECKPOINT_FILES}
    # Include actor metadata and the dataloader state in the full-checkpoint estimate.
    checkpoint = sum(path.stat().st_size for path in actor.parent.rglob("*") if path.is_file())
    hf = sum(path.stat().st_size for path in BASE_MODEL.glob("*.safetensors") if path.is_file())
    if not hf:
        raise StorageCapacityError("Cannot estimate final HF export: base safetensors are unavailable")
    return {"source_checkpoint": str(actor.parent), "checkpoint_bytes": checkpoint,
            "hf_export_bytes": hf, "shards": sizes,
            "spill_tmp_bytes": sizes[OPTIM0] + sizes[MODEL0], "spill_overlay_bytes": sizes[MODEL1],
            "resident_checkpoint_bytes": checkpoint - sum(sizes[name] for name in SPILL_FILES)}


def check_capacity(stage="formal"):
    """Read-only preflight. Sizes come from the completed generalist checkpoint."""
    if stage not in {"smoke", "formal"}:
        raise ValueError("stage must be smoke or formal")
    measured = _measure()
    disks = {"data": _disk(EXPECTED_RUN.parent), "tmp": _disk(TMP_ROOT), "overlay": _disk(OVERLAY_ROOT)}
    if len({d["device"] for d in disks.values()}) != 3:
        raise StorageCapacityError("Data, temporary spill and overlay spill must be three different filesystems")
    data_peak = (measured["checkpoint_bytes"] + measured["hf_export_bytes"] if stage == "smoke" else
                 max(measured["checkpoint_bytes"] + measured["resident_checkpoint_bytes"],
                     measured["checkpoint_bytes"] + measured["hf_export_bytes"]))
    needs = {"data": data_peak + RESERVE_BYTES}
    if stage == "formal":
        needs.update(tmp=measured["spill_tmp_bytes"] + RESERVE_BYTES,
                     overlay=measured["spill_overlay_bytes"] + RESERVE_BYTES)
    report = {"stage": stage, "measured": measured, "disks": disks,
              "required_free_bytes": needs, "reserve_bytes": RESERVE_BYTES}
    insufficient = {name: {"free": disks[name]["free_bytes"], "required": need}
                    for name, need in needs.items() if disks[name]["free_bytes"] < need}
    if insufficient:
        raise StorageCapacityError("Insufficient checkpoint capacity: " + json.dumps(insufficient))
    return report


def _run(run_dir):
    run = Path(run_dir)
    expected = EXPECTED_RUN.parent.resolve() / EXPECTED_RUN.name
    if run.is_symlink() or run.resolve() != expected:
        raise StorageOwnershipError("Only the exact owned perception_v2 run is accepted: " + str(expected))
    return expected


def _no_symlink(path):
    """All owned directories and metadata must be actual paths, not redirects."""
    if Path(path).is_symlink():
        raise StorageOwnershipError("Unexpected symlink: " + str(path))


def _atomic_json(path, data):
    _no_symlink(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing-" + uuid.uuid4().hex)
    with temporary.open("x") as handle:
        json.dump(data, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _state(run):
    path = run / MANIFEST
    _no_symlink(path)
    if path.exists():
        state = json.loads(path.read_text())
        if (state.get("schema") != 1 or state.get("run_dir") != str(run)
                or not re.fullmatch(r"perception_v2-[0-9a-f]{32}", state.get("run_key", ""))):
            raise StorageOwnershipError("Unrecognized spill manifest identity")
        return state
    state = {"schema": 1, "run_dir": str(run), "run_key": "perception_v2-" + uuid.uuid4().hex,
             "created_unix": time.time(), "steps": {}}
    _atomic_json(path, state)
    return state


def _save(run, state, step=None):
    _atomic_json(run / MANIFEST, state)
    if step is not None:
        directory = run / "perception_v2_storage_receipts"
        _no_symlink(directory)
        _atomic_json(directory / f"step_{step}.json", {
            "run_key": state["run_key"], "run_dir": str(run), "step": step,
            **state["steps"][str(step)],
        })


def _paths(run, state, step, filename):
    if step not in {25, 50, 75} or filename not in SPILL_FILES:
        raise StorageOwnershipError("Unexpected checkpoint step or file")
    actor = run / "checkpoints" / f"global_step_{step}" / "actor"
    root = OVERLAY_ROOT if filename == MODEL1 else TMP_ROOT
    destination = root / state["run_key"] / f"step_{step}" / filename
    for directory in (run / "checkpoints", actor.parent, actor, root, root / state["run_key"], destination.parent):
        _no_symlink(directory)
    return actor / filename, destination


def _signature(path):
    value = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(value.st_mode):
        raise StorageOwnershipError("Expected a regular owned file: " + str(path))
    return {"size": value.st_size, "mtime_ns": value.st_mtime_ns, "inode": value.st_ino, "device": value.st_dev}


def _hash(path):
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _known_entry(run, state, step, name, entry):
    source, destination = _paths(run, state, step, name)
    if entry.get("source") != str(source) or entry.get("destination") != str(destination):
        raise StorageOwnershipError("Manifest contains an external or unknown path")
    return source, destination


def _gc(run, state):
    collected = []
    for step_text, step_state in state["steps"].items():
        step = int(step_text)
        actor = run / "checkpoints" / f"global_step_{step}" / "actor"
        _no_symlink(actor)
        if actor.exists() or step_state.get("status") == "collected":
            continue
        # verl retains global_step_N/data.pt; the actor's disappearance is the retirement signal.
        for name, entry in step_state["files"].items():
            _, destination = _known_entry(run, state, step, name, entry)
            _no_symlink(destination)
            if destination.exists():
                if destination.stat().st_size != entry["size"]:
                    raise StorageOwnershipError("Refusing to collect modified spill file: " + str(destination))
                if entry.get("destination_signature"):
                    if _signature(destination) != entry["destination_signature"]:
                        raise StorageOwnershipError("Refusing to collect replaced spill file: " + str(destination))
                elif not entry.get("sha256") or _hash(destination) != entry["sha256"]:
                    raise StorageOwnershipError("Refusing to collect unverified spill file: " + str(destination))
                destination.unlink()
            temporary = destination.with_name("." + destination.name + ".copying")
            _no_symlink(temporary)
            if temporary.exists():
                temporary.unlink()
            if destination.parent.exists() and not any(destination.parent.iterdir()):
                destination.parent.rmdir()
        step_state["status"] = "collected"
        step_state["collected_unix"] = time.time()
        _save(run, state, step)
        collected.append(step)
    return collected


def _link_existing(source, destination, entry):
    if not destination.exists() or destination.is_symlink():
        raise StorageOwnershipError("Known spill destination is missing or redirected")
    if destination.stat().st_size != entry["size"]:
        raise StorageOwnershipError("Known spill destination changed size")
    if source.is_symlink():
        if os.readlink(source) != str(destination):
            raise StorageOwnershipError("Unknown external checkpoint symlink: " + str(source))
        if entry.get("status") == "linked":
            signature = _signature(destination)
            if signature != entry.get("destination_signature"):
                raise StorageOwnershipError("Known spill destination was modified")
            return
    else:
        if _signature(source) != entry["source_signature"]:
            raise StorageOwnershipError("Checkpoint source changed since copy")
    # Crash recovery after verified publication: recheck both bytes before replacing a source.
    if _hash(destination) != entry["sha256"]:
        raise StorageOwnershipError("Published checkpoint copy failed recovery hash")
    if not source.is_symlink() and _hash(source) != entry["sha256"]:
        raise StorageOwnershipError("Source and published copy disagree")
    if not source.is_symlink():
        temporary_link = source.with_name("." + source.name + ".spill-link-" + uuid.uuid4().hex)
        os.symlink(str(destination), temporary_link)
        if _signature(source) != entry["source_signature"]:
            temporary_link.unlink()
            raise StorageOwnershipError("Checkpoint source changed before atomic replacement")
        os.replace(temporary_link, source)
    entry["status"] = "linked"
    entry["destination_signature"] = _signature(destination)


def _move_one(run, state, step, name):
    source, destination = _paths(run, state, step, name)
    files = state["steps"][str(step)]["files"]
    entry = files.get(name)
    if source.is_symlink() and entry is None:
        raise StorageOwnershipError("Unrecognized existing checkpoint symlink: " + str(source))
    if entry is not None:
        _known_entry(run, state, step, name, entry)
        if destination.exists() or source.is_symlink():
            _link_existing(source, destination, entry)
            _save(run, state, step)
            return
    if not source.exists():
        if not source.parent.exists():
            raise _RetiredCheckpoint()
        raise StorageOwnershipError("Completed checkpoint shard is missing: " + str(source))
    original = _signature(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _no_symlink(destination)
    if destination.exists():
        raise StorageOwnershipError("Refusing to overwrite an unrecognized spill destination")
    temporary = destination.with_name("." + destination.name + ".copying")
    _no_symlink(temporary)
    if temporary.exists():
        if entry is None:
            raise StorageOwnershipError("Unrecognized temporary spill file")
        temporary.unlink()  # Only the exact temp filename recorded for this owned source.
    entry = {"source": str(source), "destination": str(destination), "size": original["size"],
             "source_signature": original, "status": "copying"}
    files[name] = entry
    _save(run, state, step)
    digest = hashlib.sha256()
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        with os.fdopen(descriptor, "rb") as incoming, temporary.open("xb") as outgoing:
            for block in iter(lambda: incoming.read(8 * 1024**2), b""):
                outgoing.write(block)
                digest.update(block)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        entry["sha256"] = digest.hexdigest()
        if temporary.stat().st_size != original["size"] or _hash(temporary) != entry["sha256"]:
            raise StorageOwnershipError("Spill copy failed read-back hash verification")
        if not source.parent.exists():
            raise _RetiredCheckpoint()
        if _signature(source) != original:
            raise StorageOwnershipError("Checkpoint source changed while copying")
        entry["status"] = "verified"
        _save(run, state, step)
        # Atomic publication that cannot overwrite a pre-existing destination.
        os.link(temporary, destination)
        temporary.unlink()
        entry["status"] = "published"
        entry["destination_signature"] = _signature(destination)
        _save(run, state, step)
        # The just-copied bytes were already read back and checked; avoid an extra
        # 50+ GiB of repeated reads on the normal path.
        temporary_link = source.with_name("." + name + ".spill-link-" + uuid.uuid4().hex)
        os.symlink(str(destination), temporary_link)
        if not source.parent.exists():
            raise _RetiredCheckpoint()
        if _signature(source) != original:
            temporary_link.unlink()
            raise StorageOwnershipError("Checkpoint changed before symlink replacement")
        os.replace(temporary_link, source)
        entry["status"] = "linked"
        _save(run, state, step)
    finally:
        if temporary.exists():
            temporary.unlink()


def _remaining_capacity(run, state, step):
    remaining = {"tmp": 0, "overlay": 0}
    data_reclaimed = 0
    actual_checkpoint = 0
    actor = run / "checkpoints" / f"global_step_{step}" / "actor"
    for name in CHECKPOINT_FILES:
        source = actor / name
        if source.is_symlink():
            if name not in SPILL_FILES or name not in state["steps"][str(step)]["files"]:
                raise StorageOwnershipError("Unknown checkpoint symlink")
            entry = state["steps"][str(step)]["files"][name]
            _, destination = _known_entry(run, state, step, name, entry)
            _link_existing(source, destination, entry)
        actual_checkpoint += source.stat().st_size
        if name in SPILL_FILES and not source.is_symlink():
            data_reclaimed += source.stat().st_size
            entry = state["steps"][str(step)]["files"].get(name)
            published = False
            if entry:
                _, destination = _known_entry(run, state, step, name, entry)
                _no_symlink(destination)
                published = destination.exists()
            if not published:
                remaining["overlay" if name == MODEL1 else "tmp"] += source.stat().st_size
    problems = {}
    for key, root in (("tmp", TMP_ROOT), ("overlay", OVERLAY_ROOT)):
        available = _disk(root)["free_bytes"]
        if available < remaining[key] + RESERVE_BYTES:
            problems[key] = {"free": available, "required": remaining[key] + RESERVE_BYTES}
    available_data = _disk(run)["free_bytes"]
    predicted = available_data + data_reclaimed
    if predicted < actual_checkpoint + RESERVE_BYTES:
        problems["data_next_checkpoint"] = {"predicted_free_after_spill": predicted,
                                             "required": actual_checkpoint + RESERVE_BYTES}
    if problems:
        raise StorageCapacityError("Insufficient capacity before spill: " + json.dumps(problems))


@contextmanager
def _locked(run):
    path = run / ".perception_v2_storage.lock"
    _no_symlink(path)
    with path.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def _tick_sync(run, final=False):
    with _locked(run):
        state = _state(run)
        collected = _gc(run, state)
        marker = run / "checkpoints/latest_checkpointed_iteration.txt"
        _no_symlink(marker)
        latest = int(marker.read_text().strip()) if marker.exists() else None
        report = {"pending": False, "run_key": state["run_key"], "latest_step": latest,
                  "collected_steps": collected, "manifest": str(run / MANIFEST), "finalized": final}
        if final or latest in {None, 100}:
            return report
        if latest not in {25, 50, 75}:
            raise StorageOwnershipError("Unexpected completed perception_v2 checkpoint step")
        actor = run / "checkpoints" / f"global_step_{latest}" / "actor"
        _no_symlink(actor)
        if not actor.exists():
            report["waiting"] = "old actor retired; waiting for latest marker update"
            return report
        step_state = state["steps"].setdefault(str(latest), {"status": "pending", "files": {}})
        if step_state["status"] == "collected":
            raise StorageOwnershipError("A retired checkpoint was unexpectedly recreated")
        if step_state["status"] == "spilled":
            # verl removes old actor files one at a time before updating latest.
            # Do not reread retired source links during that valid rotation window.
            report["spilled_step"] = latest
            report["spilled_bytes"] = sum(item["size"] for item in step_state["files"].values())
            return report
        if step_state["status"] != "spilled":
            _remaining_capacity(run, state, latest)
        try:
            for name in SPILL_FILES:
                _move_one(run, state, latest, name)
            step_state["status"] = "spilled"
            _save(run, state, latest)
        except _RetiredCheckpoint:
            report["collected_steps"] += _gc(run, state)
            report["waiting"] = "checkpoint retired during spill"
            return report
        report["spilled_step"] = latest
        report["spilled_bytes"] = sum(item["size"] for item in step_state["files"].values())
        return report


def _poll(run_dir, final=False):
    run = _run(run_dir)
    if not run.exists():
        return {"pending": False, "waiting": "run directory not yet created", "finalized": final}
    key = str(run)
    with _JOBS_LOCK:
        job = _JOBS.get(key)
        if job:
            future, was_final = job
            if not future.done():
                return {"pending": True, "operation": "finalize" if was_final else "spill"}
            del _JOBS[key]
            result = future.result()  # Worker failures reach the queue; never silently continue.
            if not final or was_final:
                return result
        _JOBS[key] = (_EXECUTOR.submit(_tick_sync, run, final), final)
        return {"pending": True, "operation": "finalize" if final else "spill"}


def tick(run_dir):
    return _poll(run_dir)


def finalize(run_dir):
    """Wait nonblockingly for previous work, then collect retired owned spill only."""
    return _poll(run_dir, final=True)
