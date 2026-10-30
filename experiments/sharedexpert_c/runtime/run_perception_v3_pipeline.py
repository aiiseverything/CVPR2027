#!/usr/bin/env python3
"""Owned, resumable candidate screening -> exact mixture -> formal v3 training.

No GPU smoke stage. Existing v2 artifacts and unrelated processes are untouched.
The user authorized same-source expansion, strict 2..6/8 screening, and training.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

RUNTIME = Path(__file__).resolve().parent
EXPERIMENT = RUNTIME.parent
WORKSPACE = EXPERIMENT.parents[1]
DATA = EXPERIMENT / "data_pipeline"
SOURCES = DATA / "perception_v3_sources"
FILTER = DATA / "prepared/perception_v3_filter"
STATE = RUNTIME / "perception_v3_pipeline.json"
TRAIN_PYTHON = EXPERIMENT / ".venv-train/bin/python"
CPU_PYTHON = WORKSPACE / ".venv-eval/bin/python"
MONITOR_PYTHON = Path("/root/miniconda3/bin/python")
CHILD = None
STOP = False


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".writing")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def identity(pid):
    try:
        proc = Path("/proc") / str(pid)
        stat = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] in {"Z", "X"}:
            return None
        return {"pid": int(pid), "start_ticks": int(stat[19]),
                "argv": [x.decode() for x in (proc / "cmdline").read_bytes().split(b"\0") if x]}
    except (OSError, ValueError):
        return None


def alive(expected):
    observed = identity(expected["pid"])
    return bool(observed and all(expected[k] == observed[k] for k in ("pid", "start_ticks", "argv")))


def update(state, stage=None, **kwargs):
    state.update(kwargs, updated_at=dt.datetime.now(dt.timezone.utc).isoformat())
    if stage:
        state["stage"] = stage
    atomic_json(STATE, state)


def ensure_running():
    if STOP:
        raise InterruptedError("Pipeline received a stop signal")


def run(state, stage, command, log, reason):
    global CHILD
    ensure_running()
    verify_code(state)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", buffering=1) as f:
        f.write("\n" + json.dumps({"command": list(map(str, command)), "stage": stage}) + "\n")
        CHILD = subprocess.Popen(list(map(str, command)), cwd=WORKSPACE, stdout=f,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        process = identity(CHILD.pid)
        update(state, stage, child=process, child_log=str(log), status_reason=reason)
        while CHILD.poll() is None:
            ensure_running()
            update(state)
            time.sleep(5)
        code = CHILD.returncode
        CHILD = None
        update(state, child=None, last_child_returncode=code)
        if code:
            raise RuntimeError(f"{stage} exited {code}; inspect {log}")


def verify_code(state):
    for relative, expected in state["code_sha256"].items():
        path = EXPERIMENT / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError(f"Pinned pipeline source changed: {relative}")


def transient_network_failure(log, start_offset):
    """Only transport failures from this attempt may trigger an automatic retry."""
    if not log.is_file():
        return False
    with log.open("rb") as stream:
        stream.seek(max(start_offset, log.stat().st_size - (256 << 10)))
        tail = stream.read().decode(errors="replace")
    permanent = ("Upstream LFS verification failed", "Metadata identity mismatch",
                 "Wrong Content-Range", "Expected HTTP206", "Wrong range byte count",
                 "SSLError", "CERTIFICATE_VERIFY_FAILED", "401 Client Error",
                 "403 Client Error", "404 Client Error", "ValueError:")
    transient = ("ProxyError", "ConnectionError", "ConnectTimeout", "ReadTimeout",
                 "TimeoutError", "RemoteDisconnected", "Connection reset",
                 "Connection refused", "Temporary failure in name resolution",
                 "NameResolutionError", "502 Server Error", "503 Server Error",
                 "504 Server Error", "429 Client Error")
    return not any(term in tail for term in permanent) and any(term in tail for term in transient)


def wait_network_retry(state, delay):
    deadline = time.monotonic() + delay
    while True:
        ensure_running()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        update(state, "waiting_network", network_retry_in_seconds=round(remaining),
               status_reason="Source download interrupted by the network; retrying automatically with verified cache preserved")
        time.sleep(min(5, remaining))


def run_source_download(state, command, log, reason):
    failures = 0
    while True:
        ensure_running()
        start_offset = log.stat().st_size if log.exists() else 0
        try:
            run(state, "expanding", command, log, reason)
            update(state, network_retry_in_seconds=0, network_retry_count=0)
            return
        except RuntimeError as exc:
            if not transient_network_failure(log, start_offset):
                raise
            failures += 1
            delay = min(300, 30 * (2 ** min(failures - 1, 4)))
            update(state, "waiting_network", network_retry_count=failures,
                   last_transient_network_error=str(exc),
                   status_reason="Temporary source-download network failure; automatic retry scheduled")
            print(json.dumps({"event": "source_network_retry", "attempt": failures,
                              "delay_seconds": delay, "log": str(log)}), flush=True)
            wait_network_retry(state, delay)


def source_lock_idle(state):
    lock_path = SOURCES / "expansion.lock"
    if not lock_path.exists():
        return
    with lock_path.open("a") as lock:
        while True:
            ensure_running()
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError:
                update(state, "expanding", status_reason="Waiting for the owned same-source download/normalization to finish")
                time.sleep(5)


def supplemental_paths():
    return sorted((SOURCES / "normalized").glob("*.jsonl"))


def extra_changed(pool, paths):
    audit = json.loads((pool / "audit.json").read_text())
    old = {str(Path(x["path"]).resolve()): x["sha256"] for x in audit["sources"]["input_files"]}
    return any(old.get(str(p.resolve())) != sha(p) for p in paths)


def next_shard(source, initial):
    indices = set()
    for p in (SOURCES / "normalized_shards" / source).glob("*.jsonl"):
        match = re.search(r"train-(\d+)-of-", p.name)
        if match:
            indices.add(int(match[1]))
    # Resume holes left by interrupted concurrent downloads before advancing.
    index = initial
    while index in indices:
        index += 1
    return index


def make_pool(state, paths):
    number = len(state["pool_dirs"]) + 1
    output = DATA / "prepared" / f"perception_v3_pool_round{number}"
    if output.exists():
        # A CPU worker may have prepared the next immutable pool while screening.
        manifest = json.loads((output / "manifest.json").read_text())
        audit = json.loads((output / "audit.json").read_text())
        if (sha(output / "audit.json") != manifest["audit_sha256"] or
                sha(output / "candidates.records.jsonl") != manifest["records_sha256"] or
                sha(output / "candidates.parquet") != manifest["data_sha256"] or
                Path(audit.get("previous_pool", "")).resolve() != Path(state["pool_dirs"][-1]).resolve()):
            raise RuntimeError(f"Prebuilt candidate pool has incompatible provenance: {output}")
        recorded = {str(Path(x["path"]).resolve()): x["sha256"] for x in audit["sources"]["input_files"]}
        if any(recorded.get(str(p.resolve())) != sha(p) for p in paths):
            raise RuntimeError("Prebuilt expanded pool does not include the current source snapshot")
        state["pool_dirs"].append(str(output))
        update(state, status_reason="Adopted the verified expanded candidate pool prepared during screening")
        return
    command = [CPU_PYTHON, DATA / "prepare_perception_v3_pool.py", "--verify-workers", "8",
               "--previous-pool", state["pool_dirs"][-1], "--output-dir", output]
    for path in paths:
        command += ["--extra-records", path]
    run(state, "prepping", command, RUNTIME / f"perception_v3_pool_round{number}.log",
        "Verifying expanded train-only candidates and evaluation exclusions")
    state["pool_dirs"].append(str(output))
    update(state)


def screening(state):
    command = [TRAIN_PYTHON, "-u", DATA / "filter_perception_v3_parallel.py",
               "--output-dir", FILTER, "--gpus", "0,1"]
    for pool in state["pool_dirs"]:
        command += ["--pool-dir", pool]
    run(state, "screening", command, FILTER / "screening.log",
        "Two independent GPUs screen disjoint questions; eight samples each, retain two through six correct")


def expand(state, remaining):
    fs = os.statvfs(SOURCES)
    if fs.f_files and fs.f_favail < 2048:
        raise RuntimeError("Insufficient filesystem inodes for expansion and training outputs")
    image_fs = os.statvfs(SOURCES / "images")
    if image_fs.f_files and image_fs.f_favail < 16000:
        raise RuntimeError("Insufficient image-cache filesystem inodes for another source expansion")
    vqa_start, doc_start = next_shard("vqav2", 4), next_shard("docvqa", 2)
    need_vqa = any(remaining.get(k, 0) for k in ("object_attribute", "spatial_detail", "general_vqa_replay",
                                               "existence_yes", "existence_no"))
    vqa_stop = min(136, vqa_start + 16) if need_vqa else vqa_start
    doc_stop = min(12, doc_start + 2) if remaining.get("ocr_document", 0) else doc_start
    raw_estimate = (vqa_stop - vqa_start) * (120 << 20) + (doc_stop - doc_start) * (300 << 20)
    # Budget transport plus decoded candidate assets; preserve checkpoint staging headroom.
    projected = raw_estimate * 3 + ((3 << 30) if remaining.get("grounding", 0) else 0)
    if shutil.disk_usage(SOURCES).free - projected < 60 * 1024**3:
        raise RuntimeError("Insufficient projected staging disk for source expansion; completed receipts are preserved")
    expanded = False
    if vqa_stop > vqa_start or doc_stop > doc_start:
        run_source_download(state, [CPU_PYTHON, "-u", DATA / "expand_perception_v3_sources.py",
            "--vqav2-start", vqa_start, "--vqav2-stop", vqa_stop,
            "--docvqa-start", doc_start, "--docvqa-stop", doc_stop], SOURCES / "expand.log",
            "Adding pinned original VQAv2/DocVQA training shards for categories still below quota")
        expanded = True
    if remaining.get("grounding", 0):
        run_source_download(state, [CPU_PYTHON, "-u", DATA / "expand_perception_v3_refcoco.py",
            "--new-images", "1000"], SOURCES / "refcoco_expand.log",
            "Adding original RefCOCO training images and unchanged referring expressions")
        expanded = True
    if not expanded:
        raise RuntimeError("Available same-source shard ranges exhausted before strict mixture quotas were filled")


def release_owned_raw_cache(state):
    """Retire reproducible source transport files after the final dataset exists.

All original annotations, image assets, manifests, and screening outputs remain.
Only this job's new parquet download cache can be removed; no v2 path is touched.
    """
    root = SOURCES / "raw"
    if not root.exists():
        return
    if root.is_symlink():
        raise RuntimeError("Unexpected symlink in owned source cache")
    manifest = json.loads((FILTER / "dataset/manifest.json").read_text())
    if manifest.get("status") != "complete" or manifest.get("records") != 3000:
        raise RuntimeError("Source cache retirement requires the complete filtered dataset")
    released = []
    for path in sorted(root.rglob("*.parquet")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            raise RuntimeError("Unexpected source cache path")
        released.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)})
    receipt = {"dataset_manifest_sha256": sha(FILTER / "dataset/manifest.json"), "files": released,
               "scope": "Only this job's reproducible downloaded parquet cache; all images and normalized records retained",
               "status": "prepared"}
    atomic_json(SOURCES / "cache_retirement.json", receipt)
    for item in released:
        Path(item["path"]).unlink()
    receipt["status"] = "complete"
    atomic_json(SOURCES / "cache_retirement.json", receipt)
    update(state, retired_source_cache_bytes=sum(x["bytes"] for x in released))


def wait_gpu(state):
    while True:
        ensure_running()
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.used",
                                        "--format=csv,noheader,nounits"], text=True)
        info = {parts[0]: (parts[1], int(parts[2])) for line in rows.splitlines()
                if len(parts := [p.strip() for p in line.split(",")]) == 3}
        if not {"0", "1"} <= set(info):
            raise RuntimeError("Original two-GPU layout is unavailable")
        uuids = {info[g][0] for g in ("0", "1")}
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                        "--format=csv,noheader"], text=True)
        if all(info[g][1] <= 1024 for g in ("0", "1")) and not any(
                line.split(",", 1)[0].strip() in uuids for line in apps.splitlines()):
            return
        update(state, "waiting_gpu", status_reason="Waiting for both original training GPUs; no other process is stopped")
        time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--initial-filter-receipt", type=Path,
                        default=RUNTIME / "perception_v3_initial_filter.json")
    args = parser.parse_args()
    lock = (RUNTIME / ".perception_v3_pipeline.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    current = identity(os.getpid())
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    if state.get("stage") == "complete":
        print(json.dumps(state), flush=True)
        return
    if state.get("child") and alive(state["child"]):
        raise RuntimeError("A prior owned pipeline child is still alive; refusing duplicate launch")
    code_names = ["runtime/run_perception_v3_pipeline.py", "runtime/monitor_perception_v3.py",
                  "data_pipeline/filter_perception_v3.py", "data_pipeline/filter_perception_v3_parallel.py",
                  "data_pipeline/perception_v3_filtering.py",
                  "data_pipeline/prepare_perception_v3_pool.py", "data_pipeline/prepare_perception_v2.py",
                  "data_pipeline/expand_perception_v3_sources.py", "data_pipeline/expand_perception_v3_refcoco.py",
                  "training/launch_perception_v3.py", "training/launch_local.py", "training/run.py",
                  "training/agent_loop.py", "training/reward_manager.py", "training/preflight.py",
                  "training/config/common_local_100.json", "training/config/perception_v3.json",
                  "training/config/perception_v3_storage_reference.json", "rewards.py"]
    pinned = {name: sha(EXPERIMENT / name) for name in code_names}
    if state.get("code_sha256") and state["code_sha256"] != pinned:
        raise RuntimeError("Pipeline source differs from the previously armed run")
    initial_pools = [str(DATA / "prepared/perception_v3_pool")]
    if args.initial_filter_receipt.exists():
        argv = json.loads(args.initial_filter_receipt.read_text())["argv"]
        inherited = [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "--pool-dir"]
        if inherited:
            initial_pools = inherited
    state.update(schema=1, pid=current["pid"], start_ticks=current["start_ticks"], error=None, child=None,
                 code_sha256=pinned, target_rows=3000, teacher="perception_v3", smoke_required=False,
                 pool_dirs=state.get("pool_dirs", initial_pools))
    update(state, "prepping", status_reason="Preparing strict filtering and automatic formal training")
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    monitor = None
    try:
        with (RUNTIME / "perception_v3_monitor.log").open("a") as log:
            monitor = subprocess.Popen([str(MONITOR_PYTHON), str(RUNTIME / "monitor_perception_v3.py"), "run"],
                                       cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(state, monitor=identity(monitor.pid))
        if args.initial_filter_receipt.is_file() and not state.get("initial_filter_observed"):
            initial = json.loads(args.initial_filter_receipt.read_text())
            if not any(Path(arg).name in {"filter_perception_v3.py", "filter_perception_v3_parallel.py"}
                       for arg in initial["argv"]):
                raise RuntimeError("Invalid initial filter process receipt")
            while alive(initial):
                ensure_running()
                update(state, "screening", initial_filter=initial,
                       status_reason="Initial frozen-model screening running; same-source expansion proceeds separately")
                time.sleep(5)
            if not (FILTER / "status.json").is_file():
                raise RuntimeError("Initial screening exited without a completion status; inspect screening.log")
            status = json.loads((FILTER / "status.json").read_text())
            if status.get("status") not in {"complete", "needs_more_candidates"}:
                raise RuntimeError("Initial screening stopped before completing its candidate pass")
            update(state, initial_filter_observed=True)
        elif not (FILTER / "status.json").exists():
            screening(state)
        while True:
            ensure_running()
            status = json.loads((FILTER / "status.json").read_text())
            if status.get("status") == "complete":
                break
            if status.get("status") != "needs_more_candidates":
                raise RuntimeError("Filtering did not finish normally; refusing an automatic reroll")
            source_lock_idle(state)
            paths = supplemental_paths()
            if not paths or not extra_changed(Path(state["pool_dirs"][-1]), paths):
                expand(state, status["remaining"])
                paths = supplemental_paths()
            make_pool(state, paths)
            screening(state)
        update(state, "preflight", status_reason="All3000 rows passed strict filtering; validating frozen data and unchanged training recipe")
        release_owned_raw_cache(state)
        launcher = [TRAIN_PYTHON, EXPERIMENT / "training/launch_perception_v3.py", "--gpus", "0,1",
                    "--dataset-dir", FILTER / "dataset"]
        run(state, "preflight", launcher + ["--check"], RUNTIME / "perception_v3_training_preflight.log",
            "CPU dataset, reward receipt, dependency, configuration and checkpoint-capacity checks")
        wait_gpu(state)
        run(state, "training", launcher + ["--execute"], RUNTIME / "perception_v3_training_launcher.log",
            "Formal perception_v3:100steps,batch30,group8; normal checkpoints and final HF export")
        marker = EXPERIMENT / "runs/perception_v3/hf/global_step_100/sharedexpert_export_complete.json"
        if not marker.is_file() or json.loads(marker.read_text()).get("step") != 100:
            raise RuntimeError("Training returned without the verified final model export")
        update(state, "complete", status_reason="Strictly filtered3000-row mixture trained and finalstep100 exported",
               model_export=str(marker.parent), dataset_manifest_sha256=sha(FILTER / "dataset/manifest.json"))
    except BaseException as exc:
        update(state, "failed", error=f"{type(exc).__name__}: {exc}", status_reason="Pipeline stopped; committed data and checkpoints preserved")
        if CHILD is not None and CHILD.poll() is None:
            os.killpg(CHILD.pid, signal.SIGTERM)
            try:
                CHILD.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(CHILD.pid, signal.SIGKILL)
        raise
    finally:
        if monitor is not None and monitor.poll() is None and state.get("stage") == "failed":
            # Give the monitor one sampling interval to record the terminal error.
            time.sleep(6)
            if monitor.poll() is None:
                os.killpg(monitor.pid, signal.SIGTERM)


if __name__ == "__main__":
    main()
