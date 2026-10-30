#!/usr/bin/env python3
"""Complete missing frozen extension evaluations on the current single GPU."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

RUNTIME = Path(__file__).resolve().parent
EXP = RUNTIME.parent
ROOT = EXP.parents[1]
EVAL = EXP / "evaluation"
sys.path.insert(0, str(EVAL))
from common import load_suite, read_jsonl, digest, local_lock_path
from evaluate_teachers_20261024 import PredictionCounter

MODELS = {
    "generalist": EXP / "runs/generalist/hf/global_step_100",
    "perception-v2-step100": EXP / "runs/perception_v2/hf/global_step_100",
    "baseline": EXP / "models/Qwen3.5-4B",
    "perception-step100": EXP / "models/Qwen3.5-4B-perception-step100",
    "reasoning-step100": EXP / "models/Qwen3.5-4B-reasoning-step100",
}
SUITES = {"general": ("suite_general.jsonl", "results_general"),
          "blink_mmvp": ("suite_blink_mmvp.jsonl", "results_blink_mmvp")}
# The local OCR/IF assets are already present. Restore BLINK assets in parallel.
# User explicitly excluded baseline-9b; keep its historical results untouched.
ORDER = [(name, "general") for name in MODELS] + [
    ("generalist", "blink_mmvp"), ("perception-v2-step100", "blink_mmvp"),
]
STATE = RUNTIME / "eval_extensions_20261025.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, data):
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default="/root/.venv-eval/bin/python")
    ap.add_argument("--checker", default="/root/.venv-eval/external/general_checker_20261025")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.gpu < 0 or args.concurrency < 1:
        ap.error("GPU must be nonnegative; concurrency must be positive")
    suites = {k: load_suite(EVAL / files[0]) for k, files in SUITES.items()}
    jobs, counters = [], {}
    for name, suite in ORDER:
        filename, output = SUITES[suite]
        command = [args.python, "-u", str(EVAL / "run_checkpoint.py"), "--model", str(MODELS[name]),
                   "--name", name, "--gpu", str(args.gpu), "--suite", str(EVAL / filename),
                   "--output-root", str(EVAL / output), "--concurrency", str(args.concurrency)]
        predictions = EVAL / output / "full" / name / "predictions.jsonl"
        key = f"{name}/{suite}"
        jobs.append({"key": key, "name": name, "suite": suite, "stage": "pending",
                     "n_expected": len(suites[suite][0]), "command": command,
                     "log": str(RUNTIME / f"eval_20261025_{name}_{suite}.log"),
                     "predictions": str(predictions), "returncode": None})
        counters[key] = PredictionCounter(predictions)
    state = {"pid": os.getpid(), "started_at": now(), "stage": "starting", "gpu": args.gpu,
             "python": args.python, "checker_overlay": args.checker, "jobs": jobs,
             "total_requested_predictions": sum(j["n_expected"] for j in jobs),
             "suite_sha256": {k: v[1]["suite_sha256"] for k, v in suites.items()},
             "summaries": []}
    if args.dry_run:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0
    if not Path(args.checker).is_dir():
        raise SystemExit("The verified OCR/IF checker overlay is missing")
    lock = local_lock_path(Path(__file__), "supervisor").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Extension supervisor already running")
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
               PYTHONPATH=args.checker, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.pop("CUDA_VISIBLE_DEVICES", None)
    active, stopping = None, []

    def interrupted(signum, frame):
        stopping.append(signum)
        if active and active.poll() is None:
            active.terminate()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)

    def publish():
        for job in jobs:
            job.update(counters[job["key"]].update())
        state["updated_at"] = now()
        state["n_successful"] = sum(j.get("n_successful", 0) for j in jobs)
        atomic_json(STATE, state)

    def execute(command, log, job):
        nonlocal active
        with Path(log).open("a") as stream:
            stream.write(f"\n[{now()}] {json.dumps(command)}\n")
            stream.flush()
            active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                      stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        job["child_pid"] = active.pid
        while active.poll() is None:
            publish()
            try:
                active.wait(timeout=15)
            except subprocess.TimeoutExpired:
                pass
        rc = active.returncode
        job["child_pid"] = None
        active = None
        return rc

    try:
        state["stage"] = "evaluating"
        publish()
        for job in jobs:
            if stopping:
                break
            rows, meta = suites[job["suite"]]
            model = MODELS[job["name"]]
            required = [model / "config.json", model / "tokenizer_config.json"]
            missing = [str(p) for p in required if not p.is_file()]
            missing += [str(p) for p in {Path(image) for row in rows for image in row.get("images", [])}
                        if not p.is_file()]
            if missing:
                job.update(stage="blocked_missing_inputs", missing_count=len(missing), missing_examples=missing[:5])
                publish()
                continue
            job.update(stage="running", started_at=now(), attempts=[])
            for attempt in range(1, 4):
                rc = execute(job["command"], job["log"], job)
                job["attempts"].append({"attempt": attempt, "returncode": rc, "finished_at": now()})
                # Existing API client exits 2 for transient request failures;
                # launcher wraps that as 1, so retry only if log identifies it.
                log_tail = Path(job["log"]).read_text()[-8000:]
                transient_api = "run_api.py" in log_tail and "exit status 2" in log_tail
                if stopping or rc == 0 or not transient_api or attempt == 3:
                    break
                time.sleep(5)
            job.update(returncode=rc, finished_at=now(), stage="complete" if rc == 0 else "failed")
            if rc == 0:
                records = {r["id"]: r for r in read_jsonl(job["predictions"])}
                good = all(r["id"] in records and not records[r["id"]].get("error")
                           and "raw_response" in records[r["id"]]
                           and records[r["id"]].get("row_sha256") == digest(r) for r in rows)
                if not good or len(records) != len(rows):
                    job.update(stage="failed_validation", error="Successful exit without exact complete frozen suite")
            publish()
            print(f"[{now()}] {job['key']}: {job['stage']}", flush=True)
        if stopping:
            state["stage"] = "interrupted"
        else:
            state["stage"] = "summarizing"
            commands = [[args.python, str(EVAL / "summarize.py"), "--suite", str(EVAL / filename),
                         "--output-root", str(EVAL / output)] for filename, output in SUITES.values()]
            commands += [[args.python, str(EVAL / "merge_summary.py")],
                         [args.python, str(EVAL / "merge_summary.py"), "--include-general"]]
            for command in commands:
                if stopping:
                    break
                summary = {"command": command, "started_at": now()}
                state["summaries"].append(summary)
                summary["returncode"] = execute(command, RUNTIME / "eval_20261025_summaries.log", summary)
                summary["finished_at"] = now()
            state["stage"] = ("interrupted" if stopping else "complete" if
                              all(j["stage"] == "complete" for j in jobs) and
                              all(s["returncode"] == 0 for s in state["summaries"]) else "incomplete")
        state["finished_at"] = now()
        publish()
        return 0 if state["stage"] == "complete" else 1
    except BaseException as exc:
        state.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        if active and active.poll() is None:
            active.terminate()
            try:
                active.wait(timeout=60)
            except subprocess.TimeoutExpired:
                state["cleanup_pending_pid"] = active.pid
        publish()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
