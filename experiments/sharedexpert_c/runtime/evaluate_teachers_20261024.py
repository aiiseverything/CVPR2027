#!/usr/bin/env python3
"""Resume the two step-100 teacher evaluations on separate GPUs, then summarize.

The existing launcher owns model validation, GPU locks, protocol verification,
resumable predictions and vLLM cleanup. This supervisor only coordinates jobs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time


RUNTIME = Path(__file__).resolve().parent
EXP = RUNTIME.parent
ROOT = EXP.parents[1]
EVAL = EXP / "evaluation"
STATE = RUNTIME / "eval_teachers_20261024.json"
STATUS = STATE.with_suffix(".md")
MODELS = (("generalist", "generalist", 0), ("perception-v2-step100", "perception_v2", 1))
SUITES = (("main", "suite.jsonl", "results"),
          ("blink_mmvp", "suite_blink_mmvp.jsonl", "results_blink_mmvp"),
          ("general", "suite_general.jsonl", "results_general"))


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_write(path, text):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text)
    temporary.replace(path)


class PredictionCounter:
    """Read only newly appended, complete JSONL records; keep the latest ID."""
    def __init__(self, path):
        self.path = Path(path)
        self.offset = 0
        self.identity = None
        self.records = {}

    def update(self):
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return {"n_predicted": 0, "n_successful": 0, "api_errors": 0}
        identity = (stat.st_dev, stat.st_ino)
        if identity != self.identity or stat.st_size < self.offset:
            self.offset, self.records, self.identity = 0, {}, identity
        if stat.st_size > self.offset:
            with self.path.open("rb") as stream:
                stream.seek(self.offset)
                while True:
                    start = stream.tell()
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        self.offset = start
                        break
                    row = json.loads(line)
                    self.records[row["id"]] = (not row.get("error") and "raw_response" in row,
                                               bool(row.get("error")))
        return {"n_predicted": len(self.records),
                "n_successful": sum(value[0] for value in self.records.values()),
                "api_errors": sum(value[1] for value in self.records.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=Path("/root/.venv-eval/bin/python"))
    parser.add_argument("--include-extensions", action="store_true")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--server-config", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print commands; do not write state or launch jobs")
    args = parser.parse_args()
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    # Do not resolve a virtualenv's Python symlink to the base interpreter.
    python = str(args.python.absolute())
    if not args.python.is_file():
        parser.error(f"Python interpreter missing: {args.python}")
    if args.server_config and not args.server_config.is_file():
        parser.error(f"Server configuration missing: {args.server_config}")
    suites = SUITES if args.include_extensions else SUITES[:1]
    state = {"pid": os.getpid(), "started_at": now(), "stage": "starting", "models": {},
             "configuration": {"python": python, "include_extensions": args.include_extensions,
                               "concurrency": args.concurrency,
                               "server_config": str(args.server_config.absolute()) if args.server_config else None,
                               "pythonpath": os.environ.get("PYTHONPATH", ""),
                               "suites": [name for name, _, _ in suites]}, "summaries": []}
    counters = {}
    for name, run, gpu in MODELS:
        model = EXP / "runs" / run / "hf/global_step_100"
        jobs = []
        for suite, filename, output in suites:
            path, output_root = EVAL / filename, EVAL / output
            # Counting rows is metadata-only; full hashes are verified by run_checkpoint.
            n_expected = sum(bool(line.strip()) for line in path.open())
            command = [python, "-u", str(EVAL / "run_checkpoint.py"), "--model", str(model),
                       "--name", name, "--gpu", str(gpu), "--suite", str(path),
                       "--output-root", str(output_root), "--concurrency", str(args.concurrency)]
            if args.server_config:
                command += ["--server-config", str(args.server_config.absolute())]
            log = RUNTIME / f"eval_20261024_{name}_{suite}.log"
            predictions = output_root / "full" / name / "predictions.jsonl"
            jobs.append({"suite": suite, "stage": "pending", "command": command,
                         "n_expected": n_expected, "log": str(log), "predictions": str(predictions),
                         "returncode": None, "child_pid": None})
            counters[(name, suite)] = PredictionCounter(predictions)
        state["models"][name] = {"gpu": gpu, "model": str(model), "stage": "pending", "jobs": jobs}
    if args.dry_run:
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return 0

    lock_directory = Path(f"/tmp/sharedexpert-evaluation-locks-{os.getuid()}")
    lock_directory.mkdir(mode=0o700, exist_ok=True)
    lock_name = hashlib.sha256(str(Path(__file__).resolve()).encode()).hexdigest() + ".lock"
    lock = (lock_directory / lock_name).open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another teacher evaluation supervisor is already running")

    active = {}
    stopping = []
    last_write = 0.0
    environment = dict(os.environ, PYTHONUNBUFFERED="1")

    def interrupted(signum, frame):
        if not stopping:
            stopping.append(signum)

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, interrupted)

    def publish(force=False):
        nonlocal last_write
        if not force and time.monotonic() - last_write < 30:
            return
        for name, model in state["models"].items():
            for job in model["jobs"]:
                try:
                    job.update(counters[(name, job["suite"])].update())
                    job.pop("progress_error", None)
                except (OSError, ValueError, KeyError) as exc:
                    job["progress_error"] = str(exc)
        state["updated_at"] = now()
        atomic_write(STATE, json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        lines = ["# 感知教师 v2 / 通用教师评测", "", f"更新时间：{state['updated_at']}",
                 f"阶段：**{state['stage']}**；监督进程 PID：{state['pid']}", "",
                 "| 模型 | GPU | 套件 | 阶段 | 成功 / 总题数 | API 错误 | 子进程 PID | 返回码 |",
                 "|---|---:|---|---|---:|---:|---:|---:|"]
        for name, model in state["models"].items():
            for job in model["jobs"]:
                lines.append(f"| {name} | {model['gpu']} | {job['suite']} | {job['stage']} | "
                             f"{job.get('n_successful', 0)} / {job['n_expected']} | "
                             f"{job.get('api_errors', 0)} | {job['child_pid'] or '—'} | "
                             f"{job['returncode'] if job['returncode'] is not None else '—'} |")
        lines += ["", "仅完成全部题目且无 API 错误的结果可作为最终成绩。",
                  "逐任务命令、日志和输出路径见同名 JSON 状态文件。",
                  "本次主套件汇总：`evaluation/results/summary_main.md`；保留历史增量的总表：`evaluation/results/summary.md`。"]
        if args.include_extensions:
            lines += ["扩展总表：`evaluation/results_expanded/summary.md`。"]
        if state.get("error"):
            lines += ["", f"错误：{state['error']}"]
        atomic_write(STATUS, "\n".join(lines) + "\n")
        last_write = time.monotonic()

    def launch(key, job):
        log_path = Path(job["log"])
        with log_path.open("a") as log:
            log.write(f"\n[{now()}] {json.dumps(job['command'], ensure_ascii=False)}\n")
            log.flush()
            process = subprocess.Popen(job["command"], cwd=ROOT, env=environment,
                                       stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        job.update(stage="running", child_pid=process.pid, started_at=now())
        active[key] = (process, job)
        print(f"[{now()}] started {key}, pid={process.pid}", flush=True)

    def collect():
        for key, (process, job) in list(active.items()):
            result = process.poll()
            if result is None:
                continue
            job.update(returncode=result, finished_at=now(), child_pid=None,
                       stage="complete" if result == 0 else "failed")
            del active[key]
            print(f"[{now()}] finished {key}, rc={result}", flush=True)

    def summarize(command, label):
        job = {"stage": "pending", "command": command, "name": label,
               "log": str(RUNTIME / "eval_20261024_summaries.log"), "returncode": None}
        state["summaries"].append(job)
        launch("summary", job)
        while active and not stopping:
            collect()
            publish()
            time.sleep(1)
        if stopping:
            return False
        publish(True)
        return job["returncode"] == 0

    try:
        state["stage"] = "evaluating"
        publish(True)
        while not stopping:
            collect()
            for name, model in state["models"].items():
                if name in active or model["stage"] in {"complete", "failed"}:
                    continue
                if any(job["stage"] == "failed" for job in model["jobs"]):
                    model["stage"] = "failed"
                    for job in model["jobs"]:
                        if job["stage"] == "pending":
                            job["stage"] = "skipped_after_failure"
                    continue
                job = next((job for job in model["jobs"] if job["stage"] == "pending"), None)
                if job is None:
                    model["stage"] = "complete"
                    continue
                model["stage"] = "running"
                try:
                    launch(name, job)
                except OSError as exc:
                    job.update(stage="failed", error=str(exc), finished_at=now())
            publish()
            if all(model["stage"] in {"complete", "failed"} for model in state["models"].values()):
                break
            time.sleep(1)
        if stopping:
            return 128 + stopping[0]
        all_complete = all(model["stage"] == "complete" for model in state["models"].values())
        state["stage"] = "summarizing"
        publish(True)
        summary_ok = True
        for suite, filename, output in suites:
            ok = summarize([python, str(EVAL / "summarize.py"), "--suite", str(EVAL / filename),
                            "--output-root", str(EVAL / output)], suite)
            summary_ok = summary_ok and ok
            if suite == "main" and ok:
                for extension in ("md", "json"):
                    shutil.copy2(EVAL / output / f"summary.{extension}",
                                 EVAL / output / f"summary_main.{extension}")
            if stopping:
                return 128 + stopping[0]
        # Restore the main + BLINK/MMVP table after per-suite summaries.
        summary_ok = summarize([python, str(EVAL / "merge_summary.py")], "merged") and summary_ok
        if args.include_extensions and not stopping:
            summary_ok = summarize([python, str(EVAL / "merge_summary.py"), "--include-general"],
                                   "expanded") and summary_ok
        if stopping:
            return 128 + stopping[0]
        state["stage"] = "complete" if all_complete and summary_ok else "failed"
        state["finished_at"] = now()
        return 0 if state["stage"] == "complete" else 1
    except Exception as exc:
        state.update(stage="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if stopping or active:
            state["stage"] = "interrupted" if stopping else "failed"
            # Signal only children we launched. run_checkpoint's own handler
            # then reclaims its separately-sessioned vLLM process group.
            for process, job in active.values():
                if process.poll() is None:
                    process.terminate()
            deadline = time.monotonic() + 60
            while active and time.monotonic() < deadline:
                collect()
                time.sleep(1)
            if active:
                state["cleanup_pending_pids"] = [process.pid for process, _ in active.values()]
            state["finished_at"] = now()
        publish(True)
        lock.close()


if __name__ == "__main__":
    sys.exit(main())
