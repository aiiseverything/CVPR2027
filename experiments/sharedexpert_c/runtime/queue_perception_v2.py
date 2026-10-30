#!/usr/bin/env python3
"""Durable, isolated generalist -> perception_v2 handoff; never stops other jobs.

Arm only after the data and runner have been prepared. --check is read-only and
does not initialize CUDA or query a GPU while the upstream job is still active.
The queue is not an automatic retry controller: a failed/interrupted child needs
an explicit recovery decision. Its state is separate from experiment_state.json.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import datetime as dt
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from zoneinfo import ZoneInfo

EXPERIMENT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT.parents[1]
STATE = EXPERIMENT / "runtime/perception_v2_queue.json"
LOG = EXPERIMENT / "runtime/perception_v2_queue.log"
RUNNER = EXPERIMENT / "training/launch_perception_v2.py"
MONITOR = EXPERIMENT / "runtime/monitor_perception_v2.py"
PYTHON = EXPERIMENT / ".venv-train/bin/python"
MONITOR_PYTHON = Path("/root/miniconda3/bin/python")
_stop = False


def timestamp():
    return dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def process_identity(pid):
    try:
        proc = Path("/proc") / str(int(pid))
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        return {"pid": int(pid), "start_ticks": int(fields[19]),
                "argv": [x.decode() for x in (proc / "cmdline").read_bytes().split(b"\0") if x]}
    except (OSError, ValueError, TypeError, IndexError):
        return None


def same_process(expected, observed):
    return bool(observed and all(observed.get(key) == expected.get(key)
                                 for key in ("pid", "start_ticks", "argv")))


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".perception-v2-", delete=False) as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def update(state, stage=None, **fields):
    state.update(fields, updated_at=timestamp())
    if stage is not None:
        state["stage"] = stage
    atomic_json(STATE, state)


def log(message):
    line = f"[{timestamp()}] {message}"
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", buffering=1) as stream:
        stream.write(line + "\n")
    print(line, flush=True)


@contextmanager
def queue_lock(path=None):
    key = hashlib.sha256(str(EXPERIMENT.resolve()).encode()).hexdigest()[:24]
    path = Path(path) if path else Path(tempfile.gettempdir()) / f"sharedexpert-perception-v2-queue-{key}.lock"
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another perception_v2 queue holds the lock; no second job was started") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def pinned_paths():
    relative = [
        "runtime/queue_perception_v2.py", "runtime/monitor_perception_v2.py",
        "runtime/perception_v2_storage.py", "monitor_training.py",
        "training/launch_perception_v2.py", "training/config/perception_v2.json",
        "data_pipeline/prepare_perception_v2.py",
        "data_pipeline/prepared/perception_v2.parquet",
        "data_pipeline/prepared/perception_v2_smoke.parquet",
        "data_pipeline/prepared/perception_v2.manifest.json",
        "data_pipeline/prepared/perception_v2.audit.json",
        "data_pipeline/prepared/perception_v2.records.jsonl",
        "data_pipeline/prepared/perception_v2.samples.json",
        "training/launch.py", "training/launch_local.py", "training/run.py",
        "training/agent_loop.py", "training/reward_manager.py", "training/preflight.py", "rewards.py",
        "training/config/common_local_100.json", "runs/generalist/protocol.json",
        "runs/generalist/resolved_config.yaml", "training/config/agent_loops.yaml",
    ]
    baseline = json.loads((EXPERIMENT / "runs/generalist/protocol.json").read_text())
    relative.extend("logs/" + name for name in baseline.get("training_environment_sha256", {}))
    return [EXPERIMENT / name for name in dict.fromkeys(relative)]


def repository_diff_sha256():
    diff = subprocess.check_output(["git", "-C", str(WORKSPACE / "pre-lab/repo"), "diff", "HEAD", "--"])
    return hashlib.sha256(diff).hexdigest()


def freeze_inputs():
    return {"files": {str(path.relative_to(EXPERIMENT)): sha256(path) for path in pinned_paths()},
            "repository_diff_sha256": repository_diff_sha256()}


def verify_inputs(frozen):
    for relative, digest in frozen["files"].items():
        path = EXPERIMENT / relative
        if not path.is_file() or sha256(path) != digest:
            raise RuntimeError(f"Pinned input changed or disappeared: {relative}; refusing launch/resume drift")
    if repository_diff_sha256() != frozen["repository_diff_sha256"]:
        raise RuntimeError("Training repository diff changed after the queue was armed")


def capture_upstream(pid):
    identity = process_identity(pid)
    autopilot = str(EXPERIMENT / "runtime/autopilot.sh")
    if not identity or not any(str(Path(arg).resolve()) == autopilot for arg in identity["argv"]):
        raise RuntimeError("--after-pid must identify the currently running project autopilot.sh")
    path = EXPERIMENT / "runtime/autopilot.log"
    stat = path.stat()
    return {"process": identity, "log_offset": stat.st_size,
            "log_device": stat.st_dev, "log_inode": stat.st_ino,
            "generalist_protocol_sha256": sha256(EXPERIMENT / "runs/generalist/protocol.json")}


def upstream_gate(upstream, experiment=EXPERIMENT, observe=process_identity):
    """Return (ready, reason), or raise if the completed upstream lacks evidence.

    An intermediate failed driver may be retried by the *existing* autopilot;
    the queue waits for that controller to exit before interpreting its outcome.
    """
    identity = upstream["process"]
    if same_process(identity, observe(identity["pid"])):
        return False, "Waiting for the captured generalist autopilot to finish training, export and archive"
    state = json.loads((experiment / "runtime/experiment_state.json").read_text())
    if state.get("teacher") != "generalist" or state.get("stage") != "complete" or state.get("error"):
        raise RuntimeError(f"Generalist autopilot exited without successful driver completion: {state.get('stage')}: {state.get('error')}")
    driver = observe(state.get("driver_pid"))
    if driver and driver.get("start_ticks") == state.get("driver_start_ticks"):
        return False, "Waiting for the generalist driver process to exit"
    child = observe(state.get("child_pid"))
    if child and "generalist" in child.get("argv", []) and any(Path(arg).name == "launch_local.py" for arg in child.get("argv", [])):
        return False, "Waiting for the generalist launcher process to exit"
    if sha256(experiment / "runs/generalist/protocol.json") != upstream["generalist_protocol_sha256"]:
        raise RuntimeError("Generalist protocol changed while waiting")
    path = experiment / "runtime/autopilot.log"
    stat = path.stat()
    if (stat.st_dev, stat.st_ino) != (upstream["log_device"], upstream["log_inode"]) or stat.st_size < upstream["log_offset"]:
        raise RuntimeError("Upstream log was replaced/truncated; cannot establish completion of the captured run")
    with path.open("rb") as stream:
        stream.seek(upstream["log_offset"])
        tail = stream.read().decode(errors="replace")
    if "AUTOPILOT_COMPLETE elapsed_min=" not in tail:
        raise RuntimeError("Captured autopilot exited without its new AUTOPILOT_COMPLETE receipt")
    run = experiment / "runs/generalist"
    if (run / "checkpoints/latest_checkpointed_iteration.txt").read_text().strip() != "100":
        raise RuntimeError("Generalist did not save checkpoint step 100")
    validate_export(run, 100)
    archived = experiment / "results_generalist_step100/protocol.json"
    if not archived.is_file() or sha256(archived) != upstream["generalist_protocol_sha256"]:
        raise RuntimeError("Generalist result archive does not match the captured protocol")
    return True, "Generalist training, export and archive completed; captured controller and trainer have exited"


def validate_export(run, step):
    target = run / f"hf/global_step_{step}"
    marker = json.loads((target / "sharedexpert_export_complete.json").read_text())
    if marker.get("step") != step or Path(marker.get("source", "")).resolve() != (run / f"checkpoints/global_step_{step}/actor").resolve():
        raise RuntimeError(f"Export receipt has wrong step/source: {target}")
    if not (target / "config.json").is_file() or not any(p.stat().st_size for p in target.glob("*.safetensors")):
        raise RuntimeError(f"Export config/weights are missing: {target}")


def gpu_idle(gpus, query=subprocess.check_output):
    rows = query(["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"], text=True)
    known = {parts[0]: (parts[1], int(parts[2])) for line in rows.splitlines()
             if (parts := [x.strip() for x in line.split(",")]) and len(parts) == 3}
    if any(gpu not in known for gpu in gpus):
        raise RuntimeError(f"Requested GPU missing: {gpus}; found {list(known)}")
    busy = {gpu: known[gpu][1] for gpu in gpus if known[gpu][1] > 1024}
    if busy:
        return False, f"Waiting for GPU memory to fall below 1 GiB: {busy}"
    apps = query(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    uuids = {known[gpu][0] for gpu in gpus}
    if any(line.split(",", 1)[0].strip() in uuids for line in apps.splitlines()):
        return False, "Waiting for selected GPUs to have no compute processes"
    return True, "Selected GPUs are idle"


def wait_for_idle(state, gpus, interval):
    while True:
        if _stop:
            raise InterruptedError("Queue received a stop signal")
        ready, reason = gpu_idle(gpus)
        if ready:
            return
        update(state, "waiting", status_reason=reason)
        time.sleep(interval)


def stop_child(child):
    """Only the process group created by this queue; never arbitrary GPU PIDs."""
    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=30)


def check_memory_pressure(cgroup=Path("/sys/fs/cgroup")):
    limit_text = (cgroup / "memory.max").read_text().strip()
    if not limit_text.isdigit() or int(limit_text) == 0:
        return
    limit = int(limit_text)
    current = int((cgroup / "memory.current").read_text())
    stats = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    working = max(0, current - int(stats.get("inactive_file", 0)))
    if working > int(limit * 0.94):
        raise MemoryError(f"Cgroup working set {working} exceeds 94% of {limit}; stopping only this queue's child process group")


def load_storage():
    name = "perception_v2_storage"
    spec = importlib.util.spec_from_file_location(name, EXPERIMENT / "runtime/perception_v2_storage.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def finish_storage(state, storage):
    """Finish safe copies and collect only spill files whose checkpoints vanished."""
    deadline = time.monotonic() + 7200
    while True:
        report = storage.finalize(EXPERIMENT / "runs/perception_v2")
        update(state, storage_report=report)
        if not report.get("pending"):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("Storage copy did not finish within 7200s; its worker was not killed mid-copy")
        time.sleep(10)


def run_child(state, stage, command, timeout, storage=None):
    verify_inputs(state["frozen_inputs"])
    if _stop:
        raise InterruptedError("Queue received a stop signal before child launch")
    reason = ("Archiving smoke evidence and reclaiming only this smoke run's weights"
              if "--cleanup-smoke" in command else f"Running {stage}")
    update(state, stage, status_reason=reason, command=command,
           child_pid=None, child_start_ticks=None)
    log(f"{stage}: {json.dumps(command)}")
    with LOG.open("a", buffering=1) as output:
        child = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        identity = process_identity(child.pid)
        update(state, child_pid=child.pid, child_start_ticks=identity["start_ticks"] if identity else None)
        deadline = time.monotonic() + timeout
        try:
            while child.poll() is None:
                if _stop:
                    raise InterruptedError(f"Queue received a stop signal during {stage}")
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"{stage} exceeded its {timeout}s limit")
                check_memory_pressure()
                if storage is not None:
                    # The storage manager has one asynchronous worker. Never
                    # block trainer/memory monitoring on a multi-GiB copy.
                    report = storage.tick(EXPERIMENT / "runs/perception_v2")
                    update(state, storage_report=report)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode:
                raise RuntimeError(f"{stage} failed with exit status {child.returncode}; see {LOG}")
        finally:
            stop_child(child)
            if storage is not None:
                update(state, child_returncode=child.returncode,
                       status_reason="Training child stopped; finishing safe checkpoint copies and storage cleanup")
                finish_storage(state, storage)
        update(state, child_returncode=child.returncode)


def launch_monitor(state):
    command = [str(MONITOR_PYTHON), str(MONITOR), "run"]
    with LOG.open("a", buffering=1) as output:
        child = subprocess.Popen(command, cwd=WORKSPACE, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
    update(state, monitor_process=process_identity(child.pid), monitor_command=command)


def run(args):
    global _stop
    with queue_lock():
        if STATE.exists():
            state = json.loads(STATE.read_text())
            if state.get("stage") == "complete":
                print(json.dumps({"already_complete": True, "state": str(STATE)}))
                return
            if state.get("stage") != "waiting" or state.get("execution_started_at"):
                raise RuntimeError("Existing queue is not waiting; refusing an implicit retry. Inspect its saved state/log first")
            if state["upstream"]["process"]["pid"] != args.after_pid or state["gpus"] != args.gpus.split(","):
                raise RuntimeError("Existing queued request differs from the requested upstream/GPU identity")
        else:
            state = {"schema_version": 1, "stage": "waiting", "teacher": "perception_v2",
                     "created_at": timestamp(), "error": None, "gpus": args.gpus.split(","),
                     "upstream": capture_upstream(args.after_pid), "frozen_inputs": freeze_inputs()}
        current = process_identity(os.getpid())
        update(state, pid=os.getpid(), start_ticks=current["start_ticks"],
               status_reason="Waiting for the captured generalist job", error=None)
        log("Queue armed; fixed inputs and upstream process identity persisted")
        try:
            verify_inputs(state["frozen_inputs"])
            while True:
                if _stop:
                    raise InterruptedError("Queue received a stop signal while waiting")
                ready, reason = upstream_gate(state["upstream"])
                update(state, "waiting", status_reason=reason)
                if ready:
                    break
                time.sleep(args.poll_seconds)
            verify_inputs(state["frozen_inputs"])
            update(state, upstream_completed_at=timestamp(),
                   upstream_final_state=json.loads((EXPERIMENT / "runtime/experiment_state.json").read_text()))
            wait_for_idle(state, state["gpus"], args.poll_seconds)
            command = [str(PYTHON), str(RUNNER), "--gpus", args.gpus]
            update(state, execution_started_at=timestamp())
            run_child(state, "preflight", command + ["--check"], 1800)
            wait_for_idle(state, state["gpus"], args.poll_seconds)
            storage = load_storage()
            update(state, storage_capacity_before_smoke=storage.check_capacity(stage="smoke"))
            update(state, "smoke", status_reason="Starting isolated v2 smoke and its monitor")
            launch_monitor(state)
            run_child(state, "smoke", command + ["--smoke"], 7200)
            smoke = EXPERIMENT / "runs/perception_v2_smoke"
            validate_export(smoke, 1)
            validation = json.loads((smoke / "smoke_validation.json").read_text())
            if validation.get("passed") is not True:
                raise RuntimeError("Perception v2 smoke lacks a passing validation receipt")
            update(state, smoke_validation=validation, smoke_completed_at=timestamp())
            run_child(state, "preflight", command + ["--cleanup-smoke"], 7200)
            cleanup = json.loads((smoke / "smoke_cleanup.json").read_text())
            if cleanup.get("status") != "complete":
                raise RuntimeError("Smoke weight cleanup lacks its completed archive receipt")
            update(state, smoke_cleanup=cleanup,
                   storage_capacity_before_formal=storage.check_capacity(stage="formal"))
            wait_for_idle(state, state["gpus"], args.poll_seconds)
            run_child(state, "training", command + ["--execute"], 172800, storage=storage)
            final_run = EXPERIMENT / "runs/perception_v2"
            validate_export(final_run, 100)
            completion = json.loads((final_run / "training_complete.json").read_text())
            if (completion.get("teacher") != "perception_v2" or completion.get("step") != 100
                    or completion.get("protocol_sha256") != sha256(final_run / "protocol.json")
                    or completion.get("data_sha256") != state["frozen_inputs"]["files"]["data_pipeline/prepared/perception_v2.parquet"]):
                raise RuntimeError("Formal training completion receipt does not match step 100 and the pinned protocol/data")
            update(state, "complete", status_reason="Perception v2 completed 100 steps and exported",
                   finished_at=timestamp(), training_complete=completion, error=None,
                   child_pid=None, child_start_ticks=None)
            log("PERCEPTION_V2_QUEUE_COMPLETE")
        except Exception as exc:
            stage = "interrupted" if isinstance(exc, InterruptedError) else "failed"
            update(state, stage, error=f"{type(exc).__name__}: {exc}", finished_at=timestamp(),
                   status_reason="Stopped; no automatic retry or downstream training will be launched")
            log(state["error"])
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--run", action="store_true")
    parser.add_argument("--after-pid", type=int, required=True)
    parser.add_argument("--gpus", default="0,1", choices=["0,1"])
    parser.add_argument("--poll-seconds", type=float, default=30)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60:
        parser.error("--poll-seconds must be between 1 and 60")
    def on_signal(signum, frame):
        global _stop
        _stop = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, on_signal)
    if args.check:
        state = json.loads(STATE.read_text()) if STATE.exists() else None
        upstream = state["upstream"] if state else capture_upstream(args.after_pid)
        frozen = state["frozen_inputs"] if state else freeze_inputs()
        verify_inputs(frozen)
        ready, reason = upstream_gate(upstream)
        print(json.dumps({"ready": ready, "status_reason": reason, "gpu_started": False,
                          "state_written": False, "pinned_files": len(frozen["files"]),
                          "upstream": upstream, "existing_stage": state.get("stage") if state else None},
                         ensure_ascii=False, indent=2))
    else:
        run(args)


if __name__ == "__main__":
    main()
