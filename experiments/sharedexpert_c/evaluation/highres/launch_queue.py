#!/usr/bin/env python3
"""Durable selection/prepare/freeze queue for four perception-only evaluations."""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

from protocol import DATA, HERE, RESULTS, atomic_json, digest, file_sha, PROTOCOL
from run import (AUTHORIZED_MODELS, BENCHMARKS, gpu_status, local_lock_path,
                 offline_environment, selection_snapshot)
from scheduling import configuration as scheduling_configuration
import throughput

ORDER = list(AUTHORIZED_MODELS)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def state_of(path):
    if not path.exists():
        return {"status": "missing"}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"status": "unreadable"}


def stage_of(value):
    return value.get("status", value.get("stage"))


def config_history(previous, config, output_root):
    """Allow a recorded scheduler change only before any model run exists."""
    history = list(previous.get("configuration_history", []))
    before = previous.get("config")
    if before and before != config:
        changed = {key for key in set(before) | set(config) if before.get(key) != config.get(key)}
        permitted = {"concurrency", "adaptive_concurrency", "throughput_mode", "scheduling_policy"}
        model_artifacts = any((Path(output_root) / "full").glob("*/run.json")) or any(
            (Path(output_root) / "full").glob("*/predictions.jsonl"))
        if not changed <= permitted or previous.get("jobs") or previous.get("child_pid") is not None or model_artifacts:
            raise ValueError("Existing queue configuration differs; scheduler migration is only allowed before all model runs with no active child")
        history.append({"changed_at": now(), "reason": "Pre-inference scheduling adjustment; selection and inference protocol unchanged",
                        "changed_fields": sorted(changed), "before": before, "after": config})
    return history


def resource_snapshot(root, output_root):
    result = {"checked_at": now(), "disks": {}}
    paths = {"packed_data": Path(root), "results": Path(output_root),
             "source_scratch": Path(root) / "sources"}
    for name, path in paths.items():
        path = path.resolve()
        while not path.exists() and path != path.parent:
            path = path.parent
        usage = shutil.disk_usage(path)
        result["disks"][name] = {"path": str(path), "total_bytes": usage.total,
                                  "used_bytes": usage.used, "free_bytes": usage.free}
    for name in ("memory.max", "memory.current"):
        path = Path("/sys/fs/cgroup") / name
        if path.is_file():
            value = path.read_text().strip()
            result[name] = int(value) if value.isdigit() else value
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DATA)
    parser.add_argument("--output-root", type=Path, default=RESULTS)
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(BENCHMARKS))
    parser.add_argument("--ids", type=Path,
                        help="JSON highres-subset-v1 manifest selecting frozen suite row IDs")
    parser.add_argument("--frozen-suite", action="store_true",
                        help="Reuse the existing frozen suite without preparation or preflight replay")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=2)
    parser.add_argument("--adaptive-concurrency", action="store_true")
    parser.add_argument("--throughput-mode", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.max_attempts < 1 or not 1 <= args.poll_seconds <= 60:
        parser.error("max-attempts must be positive and poll-seconds must be in 1..60")
    if args.adaptive_concurrency and args.concurrency != 16:
        parser.error("adaptive concurrency requires --concurrency 16")
    if args.throughput_mode and (args.concurrency != 64 or args.adaptive_concurrency):
        parser.error("throughput mode requires --concurrency 64 and cannot combine with adaptive-concurrency")
    if args.ids is not None:
        if not args.ids.is_file():
            parser.error(f"subset manifest not found: {args.ids}")
        if not args.frozen_suite:
            parser.error("--ids requires --frozen-suite")
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock = local_lock_path(args.output_root, "highres-queue").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.output_root / "status.json"
    previous = state_of(state_path)
    config = {"root": str(args.root.resolve()), "output_root": str(args.output_root.resolve()),
              "python": str(Path(args.python).absolute()), "gpu": args.gpu,
              "concurrency": args.concurrency, "models": ORDER,
              "adaptive_concurrency": args.adaptive_concurrency,
              "throughput_mode": args.throughput_mode,
              "scheduling_policy": (throughput.configuration() if args.throughput_mode else scheduling_configuration() if args.adaptive_concurrency else
                                    {"version": "fixed-concurrency-v1", "max_inflight": args.concurrency}),
              "selection": str((args.root / "selection.json").resolve()),
              "protocol_sha256": digest(PROTOCOL), "max_attempts": args.max_attempts}
    if args.benchmarks is not None:
        config["benchmarks"] = sorted(set(args.benchmarks))
    if args.frozen_suite:
        config["frozen_suite"] = True
    if args.ids is not None:
        config["subset"] = {"path": str(args.ids.resolve()), "sha256": file_sha(args.ids)}
    configuration_history = config_history(previous, config, args.output_root)
    attempt_counts = {key: previous.get(key, 0) for key in ("prepare_attempts", "freeze_attempts")}
    # A user-requested scheduling handover is not a failed preparation attempt.
    command = previous.get("command", [])
    if previous.get("stage") == "interrupted" and previous.get("child_pid") is None and len(command) > 1:
        interrupted_stage = Path(command[1]).stem
        key = interrupted_stage + "_attempts"
        if key in attempt_counts and attempt_counts[key] > 0:
            attempt_counts[key] -= 1
            configuration_history.append({"changed_at": now(), "reason": "Interrupted CPU stage did not fail; return its consumed attempt",
                                          "stage": interrupted_stage, "before_attempts": previous[key], "after_attempts": attempt_counts[key]})
    state = {"pid": os.getpid(), "started_at": previous.get("started_at", now()),
             "resumed_at": now(), "stage": "waiting_data", "child_pid": None,
             "config": config, "jobs": previous.get("jobs", {}),
             "configuration_history": configuration_history,
             "selection_sha256": previous.get("selection_sha256"),
             **attempt_counts,
             "resources": resource_snapshot(args.root, args.output_root)}
    child = None
    def update(**values):
        state.update(values, updated_at=now())
        if state.get("current_model"):
            detail = state_of(args.output_root / "full" / state["current_model"] / "status.json")
            state["model_progress"] = detail
        atomic_json(state_path, state)
    def interrupt(signum, frame):
        update(stage="interrupted", signal=signum)
        if child is not None and child.poll() is None:
            # Signal the launcher, which owns and cleans its own vLLM process group.
            child.terminate()
            try:
                child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                update(error="Owned launcher did not exit within 60 seconds; no unrelated processes were signalled")
        update(child_pid=None if child is None or child.poll() is not None else child.pid)
        raise SystemExit(128 + signum)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupt)
    def launch(command, log_path, stage):
        nonlocal child
        update(stage=stage, command=command, log=str(log_path), error=None)
        with log_path.open("a") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=offline_environment())
        update(child_pid=child.pid, last_child_pid=child.pid)
        while child.poll() is None:
            update()
            time.sleep(args.poll_seconds)
        code = child.returncode
        child = None
        update(child_pid=None, last_returncode=code)
        return code
    def backoff(seconds, stage):
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            update(stage=stage, retry_seconds=max(0, int(deadline - time.monotonic())))
            time.sleep(min(args.poll_seconds, max(0.01, deadline - time.monotonic())))
    update()
    try:
        while True:
            download = state_of(args.root / "download_state.json")
            prepare = state_of(args.root / "prepare_state.json")
            selection = selection_snapshot(args.root, require_ready=False)
            update(stage="waiting_selection" if stage_of(download) == "complete" else "waiting_data",
                   download_state=download, prepare_state=prepare, selection=selection)
            if stage_of(download) == "failed":
                raise RuntimeError("Source download failed; see the recorded download state")
            if stage_of(download) == "complete" and selection is not None:
                if state["selection_sha256"] not in (None, selection["sha256"]):
                    raise ValueError("Selection changed after this queue began preparation")
                update(selection_sha256=selection["sha256"], resources=resource_snapshot(args.root, args.output_root))
                break
            time.sleep(args.poll_seconds)
        # Downloads deliberately do not auto-prepare. This queue owns both CPU stages.
        stages = [("prepare", [args.python, str(HERE / "prepare.py"), "--root", str(args.root),
                               "--selection", str(args.root / "selection.json")]),
                  ("freeze", [args.python, str(HERE / "freeze.py"), "--root", str(args.root)])]
        if args.frozen_suite:
            if not (args.root / "suite.meta.json").is_file() or not (args.root / "suite.jsonl").is_file():
                raise ValueError("--frozen-suite requires an existing frozen suite")
            stages = []
        for stage, command in stages:
            attempt_key = stage + "_attempts"
            while True:
                if selection_snapshot(args.root)["sha256"] != state["selection_sha256"]:
                    raise ValueError("Selection changed during the queued evaluation")
                if state[attempt_key] >= args.max_attempts:
                    raise RuntimeError(f"CPU {stage} exhausted its persisted retry budget")
                state[attempt_key] += 1
                code = launch(command, args.output_root / f"{stage}.log", "preparing" if stage == "prepare" else "freezing")
                receipt = state_of(args.root / f"{stage}_state.json")
                update(**{stage + "_state": receipt})
                if code == 0 and stage_of(receipt) == "complete":
                    manifest_name = "dataset_manifest.json" if stage == "prepare" else "suite.meta.json"
                    manifest = state_of(args.root / manifest_name)
                    if manifest.get("selection_sha256") != state["selection_sha256"]:
                        raise ValueError(f"{manifest_name} does not bind the queued perception selection")
                    state[attempt_key] = 0
                    update()
                    break
                if state[attempt_key] >= args.max_attempts:
                    raise RuntimeError(f"CPU {stage} failed after {state[attempt_key]} attempts (rc={code}, state={stage_of(receipt)})")
                backoff(30 * 2 ** (state[attempt_key] - 1), "retrying_" + stage)
        for name in ORDER:
            if selection_snapshot(args.root)["sha256"] != state["selection_sha256"]:
                raise ValueError("Selection changed before model evaluation")
            update(current_model=name)
            job = state["jobs"].setdefault(name, {"attempts": 0, "returncodes": []})
            # Completed entries still pass run.py's exact cache checks on resume.
            while True:
                resources = gpu_status(args.gpu)
                if resources["busy"] and job.get("stage") != "complete":
                    update(stage="waiting_gpu", resources=resources)
                    time.sleep(args.poll_seconds)
                    continue
                if job.get("stage") != "complete" and job["attempts"] >= args.max_attempts:
                    raise RuntimeError(f"{name}: exhausted persisted retry budget ({args.max_attempts})")
                was_complete = job.get("stage") == "complete"
                job["stage"] = "running"
                if not was_complete:
                    job["attempts"] += 1
                command = [args.python, str(HERE / "run.py"), "--name", name,
                           "--gpu", str(args.gpu), "--root", str(args.root),
                           "--output-root", str(args.output_root), "--concurrency", str(args.concurrency)]
                if args.adaptive_concurrency:
                    command.append("--adaptive-concurrency")
                if args.throughput_mode:
                    command.extend(["--throughput-mode", "--execution-profile", str(job.get("execution_profile", 0))])
                if args.benchmarks is not None:
                    command.extend(["--benchmarks", *config["benchmarks"]])
                if args.ids is not None:
                    command.extend(["--ids", str(args.ids)])
                code = launch(command, args.output_root / f"{name}.log", "evaluating")
                job["returncodes"].append({"returncode": code, "time": now()})
                if code == 75:
                    if not was_complete:
                        job["attempts"] -= 1
                    job["stage"] = "waiting_gpu"
                    backoff(args.poll_seconds, "waiting_gpu")
                    continue
                if code == 0:
                    job.update(stage="complete", completed_at=now())
                    update(stage="summarizing_model")
                    from scoring import summarize
                    summarize(root=args.root, results=args.output_root)
                    update(last_summary_at=now())
                    break
                job.update(stage="failed", last_returncode=code)
                update(error=f"{name} launcher returned {code}; see {name}.log")
                if code == 3 or was_complete or job["attempts"] >= args.max_attempts:
                    raise RuntimeError(f"{name}: evaluation failed (rc={code}, attempts={job['attempts']})")
                detail = state_of(args.output_root / "full" / name / "status.json")
                profile_index = job.get("execution_profile", 0)
                current_oom = (detail.get("gpu_oom") is True
                               and detail.get("pid") == state.get("last_child_pid")
                               and detail.get("config", {}).get("execution_profile") == profile_index)
                if args.throughput_mode and current_oom and profile_index + 1 < len(throughput.configuration()["profiles"]):
                    job["execution_profile"] = profile_index + 1
                    job.setdefault("profile_changes", []).append({"changed_at": now(), "from": profile_index,
                        "to": profile_index + 1, "reason": "CUDA OOM found in this owned server attempt log",
                        "server_log_offset": detail.get("server_log_offset")})
                    update()
                backoff(30 * 2 ** (job["attempts"] - 1), "retrying_model")
        update(stage="summarizing", current_model=None)
        from scoring import summarize
        summarize(root=args.root, results=args.output_root)
        update(stage="complete", completed_at=now(), error=None)
    except Exception as error:
        update(stage="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    main()
