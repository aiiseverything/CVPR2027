#!/usr/bin/env python3
"""Durable baseline finalization: wait for owned core client, reuse rows, finish final suite.

Multiple callers serialize on a lock. Thus the training orchestrator can call this
at the end even while a detached copy is already waiting/running.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

from common import HERE, ROOT, digest, file_sha, load_suite, read_jsonl, local_lock_path
from run_api import request_json


def wait_for_endpoint(base_url, model, timeout=1200):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            models = request_json(base_url.rstrip('/') + '/models', timeout=5)
            if model in {entry['id'] for entry in models['data']}:
                return
        except (OSError, ValueError, KeyError):
            pass
        time.sleep(15)
    raise TimeoutError('Baseline endpoint unavailable for 20 minutes; cached successes remain resumable')


def matching_core_client(pid):
    try:
        args = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
    except FileNotFoundError:
        return False
    return any(a.endswith(b"/evaluation/run_api.py") for a in args) and any(a.endswith(b"/evaluation/core_suite.jsonl") for a in args)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", type=Path, default=HERE / "suite.jsonl")
    ap.add_argument("--base-url", default="http://127.0.0.1:8112/v1")
    ap.add_argument("--model", default="Qwen3.5-4B")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--output-root", type=Path, default=HERE / "results")
    args = ap.parse_args()
    rows, meta = load_suite(args.suite)
    if not meta.get("grounding_included"):
        raise SystemExit("The final baseline requires a frozen suite including heldout grounding")
    py = ROOT / "pre-lab/venv/bin/python"
    args.output_root.mkdir(parents=True, exist_ok=True)
    print("Acquiring baseline finalization lock", flush=True)
    with local_lock_path(args.output_root, 'baseline-final').open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pid_file = HERE / "baseline_core.pid"
        pid = int(pid_file.read_text()) if pid_file.exists() else None
        count = 0
        while pid and matching_core_client(pid):
            if count % 10 == 0:
                print(f"Waiting for existing core client PID {pid}; no duplicate inference", flush=True)
            count += 1
            time.sleep(30)
        command = [str(py), str(HERE / "run_api.py"), "--suite", str(args.suite.resolve()),
                   "--base-url", args.base_url, "--model", args.model, "--name", "baseline",
                   "--concurrency", str(args.concurrency), "--output-root", str(args.output_root.resolve())]
        source = HERE / "core_results/full/baseline"
        if (source / "run.json").exists():
            command.extend(["--import-results", str(source)])
        while True:
            wait_for_endpoint(args.base_url, args.model)
            result = subprocess.run(command)
            if result.returncode == 0:
                break
            if result.returncode != 2:
                # Protocol and provenance failures are not transient API errors.
                raise subprocess.CalledProcessError(result.returncode, command)
            print('Retrying API errors after 30 seconds; successful generations are preserved', flush=True)
            time.sleep(30)
        subprocess.run([str(py), str(HERE / "summarize.py"), "--suite", str(args.suite.resolve()),
                        "--output-root", str(args.output_root.resolve())], check=True)
        print("Final baseline complete on frozen suite", flush=True)


if __name__ == "__main__":
    main()
