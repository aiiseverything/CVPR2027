#!/usr/bin/env python3
"""Minimal durable driver for the local 2-GPU generalist 100-step run.

It writes the same runtime/experiment_state.json contract used by the
three-teacher orchestrator, so monitor_training.py can report stage/teacher,
detect an unexpected driver exit, and stop after a clean completion.
"""
from __future__ import annotations

import datetime
import json
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
WORKSPACE = EXPERIMENT.parents[1]
STATE = HERE / "experiment_state.json"
LOG = HERE / "experiment.log"
PYTHON = EXPERIMENT / ".venv-train/bin/python"
LAUNCH = EXPERIMENT / "training/launch_local.py"
GPUS = os.environ.get("SHAREDEXPERT_GPUS", "0,1")
_stop = False


def now() -> str:
    return datetime.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def start_ticks(pid: int) -> int:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[19])


def memory_pressure() -> tuple[int, int]:
    """Return (working_set_bytes, limit_bytes); limit 0 means unlimited."""
    root = Path("/sys/fs/cgroup")
    limit_text = (root / "memory.max").read_text().strip()
    limit = int(limit_text) if limit_text.isdigit() else 0
    current = int((root / "memory.current").read_text())
    stats = {key: int(value) for key, value in
             (line.split() for line in (root / "memory.stat").read_text().splitlines())}
    return max(0, current - stats.get("inactive_file", 0)), limit


def write_state(stage: str, **fields) -> None:
    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text())
        except ValueError:
            state = {}
    state.update(stage=stage, teacher="generalist", updated_at=now(), **fields)
    if stage != "failed":
        state["error"] = None
    with tempfile.NamedTemporaryFile("w", dir=STATE.parent, prefix=".experiment-", delete=False) as stream:
        json.dump(state, stream, ensure_ascii=False, indent=2)
        temporary = Path(stream.name)
    temporary.replace(STATE)


def on_signal(signum, frame):
    global _stop
    _stop = True


def main() -> int:
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)
    command = [str(PYTHON), str(LAUNCH), "generalist", "--gpus", GPUS, "--execute"]
    write_state("preflight", driver_pid=os.getpid(), driver_start_ticks=start_ticks(os.getpid()),
                command=command, gpus=GPUS, child_pid=None, started_at=now())
    with LOG.open("a", buffering=1) as log:
        log.write(f"\n=== local driver start {now()} gpus={GPUS} ===\n")
        child = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
        write_state("training", driver_pid=os.getpid(), driver_start_ticks=start_ticks(os.getpid()),
                    command=command, gpus=GPUS, child_pid=child.pid)
        while child.poll() is None:
            if _stop:
                child.terminate()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                write_state("interrupted", child_pid=child.pid,
                            error="Local driver received a stop signal; trainer was terminated")
                return 128 + signal.SIGTERM
            working, limit = memory_pressure()
            if limit and working > int(limit * 0.94):
                message = (f"Preventive memory guard: cgroup working set {working} exceeds 94% of {limit}; "
                           "terminated only this experiment child group")
                log.write(message + "\n")
                child.terminate()
                try:
                    child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                write_state("failed", child_pid=child.pid, error=message)
                return 137
            write_state("training", driver_pid=os.getpid(), driver_start_ticks=start_ticks(os.getpid()),
                        command=command, gpus=GPUS, child_pid=child.pid)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
        if child.returncode == 0:
            write_state("complete", child_pid=child.pid, finished_at=now())
            return 0
        write_state("failed", child_pid=child.pid, finished_at=now(),
                    error=f"launch_local.py exited with status {child.returncode}; see runs/generalist/training.log")
        return child.returncode


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] != "run":
        raise SystemExit("usage: run_generalist_local.py run")
    sys.exit(main())
