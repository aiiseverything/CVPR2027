#!/usr/bin/env python3
"""Wait for the formal v3 export, then evaluate highres, OCR and main in order."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

RUNTIME = Path(__file__).resolve().parent
EXP = RUNTIME.parent
ROOT = EXP.parents[1]
EVAL = EXP / "evaluation"
RUN = EXP / "runs/perception_v3"
MODEL = RUN / "hf/global_step_100"
NAME = "perception-v3-step100"
INPUTS = RUNTIME / "eval_perception_v3_inputs"
RESULTS = EVAL / "results_perception_v3"
STATE = RUNTIME / "eval_perception_v3.json"
BENCHMARKS = ["vstar", "zoom", "hrbench_4k", "hrbench_8k"]
sys.path.insert(0, str(EVAL))
from common import digest, file_sha, load_suite, local_lock_path, read_jsonl
from evaluate_teachers_20261024 import PredictionCounter


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def process_identity(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else int(fields[19])
    except (FileNotFoundError, ProcessLookupError):
        return None


def export_ready(run, expected_config):
    """An incomplete directory or an old/different run's receipt never suffices."""
    complete = run / "fast_start_complete.json"
    if not complete.exists():
        return False
    receipt = json.loads(complete.read_text())
    model = run / "hf/global_step_100"
    if (receipt.get("status") != "complete" or receipt.get("step") != 100
            or receipt.get("config_sha256") != expected_config
            or Path(receipt.get("hf_model", "")).resolve() != model.resolve()):
        raise ValueError("Formal training completion receipt does not match step100/config/model")
    marker = json.loads((model / "sharedexpert_export_complete.json").read_text())
    if marker.get("step") != 100 or Path(marker.get("source", "")).resolve() != (run / "checkpoints/global_step_100/actor").resolve():
        raise ValueError("Export marker does not match the formal step100 checkpoint")
    for name in ("config.json", "tokenizer_config.json"):
        if not (model / name).is_file():
            raise ValueError(f"Export missing {name}")
    if not any((model / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        raise ValueError("Export missing processor metadata")
    weights = {p.name: p for p in model.glob("*.safetensors")}
    if not weights or any(p.stat().st_size == 0 for p in weights.values()):
        raise ValueError("Export missing nonempty weights")
    index = model / "model.safetensors.index.json"
    if index.exists() and not set(json.loads(index.read_text())["weight_map"].values()) <= weights.keys():
        raise ValueError("Export index references missing weights")
    return True


def gpu_free(gpu):
    uuid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                                   "--format=csv,noheader"], text=True, timeout=30).strip()
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                   "--format=csv,noheader"], text=True, timeout=30)
    memory = int(subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used",
                                        "--format=csv,noheader,nounits"], text=True, timeout=30).strip())
    return not any(line.split(",")[0].strip() == uuid for line in apps.splitlines()) and memory <= 256


def training_step():
    path = RUN / "training.log"
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 64000))
        steps = re.findall(rb"training/global_step:(\d+)", stream.read())
    return int(steps[-1]) if steps else None


def commands(python, gpu):
    model_args = ["--name", NAME, "--model", str(MODEL), "--gpu", str(gpu)]
    highres = [python, "-u", str(EVAL / "highres/run.py"), *model_args,
               "--root", str(INPUTS / "highres"), "--output-root", str(RESULTS / "highres"),
               "--benchmarks", *BENCHMARKS, "--concurrency", "64", "--throughput-mode", "--enforce-eager"]
    jobs = [{"suite": "highres", "n_expected": 1722, "command": highres,
             "summary_command": [python, str(EVAL / "highres/scoring.py"), "--root", str(INPUTS / "highres"),
                                 "--results", str(RESULTS / "highres")]}]
    for suite, path, count in (("ocr", INPUTS / "suite_ocr.jsonl", 997), ("main", EVAL / "suite.jsonl", 3423)):
        jobs.append({"suite": suite, "n_expected": count,
                     "command": [python, "-u", str(EVAL / "run_checkpoint.py"), *model_args,
                                 "--suite", str(path), "--output-root", str(RESULTS / suite),
                                 "--concurrency", "16", "--server-config", str(INPUTS / "server.json")],
                     "summary_command": [python, str(EVAL / "summarize.py"), "--suite", str(path),
                                         "--output-root", str(RESULTS / suite)]})
    for job in jobs:
        job.update(stage="pending", attempts=[], log=str(RUNTIME / f"eval_perception_v3_{job['suite']}.log"),
                   predictions=str(RESULTS / job["suite"] / "full" / NAME / "predictions.jsonl"),
                   summary=str(RESULTS / job["suite"] / "summary.json"))
    return jobs


def validate_predictions(job):
    if job["suite"] == "highres":
        rows = [r for r in read_jsonl(INPUTS / "highres/suite.jsonl") if r["benchmark"] in BENCHMARKS]
    else:
        rows, _ = load_suite(INPUTS / "suite_ocr.jsonl" if job["suite"] == "ocr" else EVAL / "suite.jsonl")
    records = {r["id"]: r for r in read_jsonl(job["predictions"])}
    if len(rows) != job["n_expected"] or set(records) != {r["id"] for r in rows}:
        raise ValueError(f"{job['suite']}: incomplete or unexpected predictions")
    if any(records[r["id"]].get("error") or "raw_response" not in records[r["id"]]
           or records[r["id"]].get("row_sha256") != digest(r) for r in rows):
        raise ValueError(f"{job['suite']}: failed or mismatched prediction rows")


def validate_summary(job):
    report = json.loads(Path(job["summary"]).read_text())
    result = report.get("models" if job["suite"] == "highres" else "status", {}).get(NAME, {})
    if result.get("status") != "complete" or result.get("n_expected") != job["n_expected"]:
        raise ValueError(f"{job['suite']}: summary does not contain the complete v3 result")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", default="/root/.venv-eval/bin/python")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 60 or args.gpu < 0 or not Path(args.python).is_file():
        parser.error("Invalid interpreter, GPU or polling interval")
    jobs = commands(args.python, args.gpu)
    registration = [args.python, str(EVAL / "highres/register_model.py"),
                    "--source-root", str(EVAL / "data/highres_perception_v1"), "--root", str(INPUTS / "highres"),
                    "--model", str(MODEL), "--name", NAME, "--benchmarks", *BENCHMARKS]
    if args.dry_run:
        print(json.dumps({"order": [j["suite"] for j in jobs], "registration": registration, "jobs": jobs}, indent=2))
        return 0
    lock = local_lock_path(STATE, "supervisor").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = file_sha(RUN / "resolved_config.fast.yaml")
    if json.loads((RUN / "fast_start.json").read_text())["config_sha256"] != config:
        raise ValueError("Training config differs from active launch receipt")
    training = json.loads((RUNTIME / "perception_v3_resume_20261030.json").read_text())["process"]
    inputs = [EVAL / "suite.jsonl", EVAL / "suite.meta.json", INPUTS / "suite_ocr.jsonl",
              INPUTS / "suite_ocr.meta.json", INPUTS / "server.json"]
    inputs += [EVAL / "data/highres_perception_v1" / n for n in
               ("suite.jsonl", "suite.meta.json", "selection.json", "source_manifest.json", "dataset_manifest.json", "questions.jsonl")]
    hashes = {str(p): file_sha(p) for p in inputs}
    for path in (EVAL / "suite.jsonl", INPUTS / "suite_ocr.jsonl"):
        rows, _ = load_suite(path)
        missing = [image for row in rows for image in row.get("images", []) if not Path(image).is_file()]
        if missing:
            raise ValueError(f"Evaluation images missing: {missing[:3]}")
    state = {"schema": 1, "pid": os.getpid(), "start_ticks": process_identity(os.getpid()),
             "started_at": now(), "stage": "waiting_training_export", "model": str(MODEL),
             "order": [j["suite"] for j in jobs], "gpu": args.gpu, "python": args.python,
             "training_process": training, "training_config_sha256": config, "input_sha256": hashes,
             "jobs": jobs, "registration": registration, "gpu_smoke_performed": False}
    previous = json.loads(STATE.read_text()) if STATE.exists() else {}
    if previous:
        if any(previous.get(k) != state[k] for k in ("order", "model", "training_config_sha256", "input_sha256", "python")):
            raise ValueError("Existing evaluation queue has a different identity")
        if previous.get("stage") == "complete":
            print("Evaluation already completed; see " + str(STATE))
            return 0
        old_jobs = {j["suite"]: j for j in previous["jobs"]}
        for job in jobs:
            old = old_jobs[job["suite"]]
            if old["command"] != job["command"]:
                raise ValueError("Existing evaluation command differs")
            job["attempts"] = old.get("attempts", [])
            if old.get("stage") == "complete":
                job.update(stage="complete", finished_at=old.get("finished_at"))
        state["resumed_from"] = {"pid": previous["pid"], "stage": previous["stage"], "started_at": previous["started_at"]}
    counters = {j["suite"]: PredictionCounter(j["predictions"]) for j in jobs}
    active = None
    environment = {k: v for k, v in os.environ.items()
                   if k.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"} and k != "PYTHONPATH"}
    environment.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                       NO_PROXY="127.0.0.1,localhost,::1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8",
                       OPENBLAS_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
                       VLLM_USE_FLASHINFER_SAMPLER="0")

    def publish(**values):
        state.update(values, updated_at=now())
        for job in jobs:
            try:
                job.update(counters[job["suite"]].update())
                job.pop("progress_error", None)
            except (OSError, ValueError, KeyError) as exc:
                job["progress_error"] = str(exc)
        atomic_json(STATE, state)

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    def execute(command, log_path, cpu=False):
        nonlocal active
        with Path(log_path).open("a") as log:
            log.write(f"\n[{now()}] {json.dumps(command)}\n")
            log.flush()
            env = dict(environment, CUDA_VISIBLE_DEVICES="") if cpu else environment
            active = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        publish(child_pid=active.pid, command=command)
        while active.poll() is None:
            publish()
            try:
                active.wait(timeout=args.poll_seconds)
            except subprocess.TimeoutExpired:
                pass
        code = active.returncode
        active = None
        publish(child_pid=None, last_returncode=code)
        return code

    def wait_gpu():
        while not gpu_free(args.gpu):
            publish(stage="waiting_gpu")
            time.sleep(args.poll_seconds)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    try:
        publish()
        while True:
            ready = export_ready(RUN, config)
            alive = process_identity(training["pid"]) == training["start_ticks"]
            if not alive and not ready:
                # The launcher atomically commits the receipt immediately before
                # exit; it may do both between the two observations above.
                ready = export_ready(RUN, config)
            publish(last_training_step=training_step(), export_ready=ready, training_launcher_alive=alive)
            if ready and not alive:
                break
            if not alive:
                raise RuntimeError("Training/export launcher exited without a valid final completion receipt")
            time.sleep(args.poll_seconds)
        if any(file_sha(Path(path)) != expected for path, expected in hashes.items()):
            raise ValueError("Frozen evaluation inputs changed while waiting for training")
        publish(stage="registering_export", export_verified_at=now())
        if execute(registration, RUNTIME / "eval_perception_v3_registration.log", cpu=True):
            raise RuntimeError("CPU registration of exported model failed; see registration log")
        # The existing highres scorer reads the selected benchmark scope here.
        atomic_json(RESULTS / "highres/status.json", {"config": {"benchmarks": sorted(BENCHMARKS)}, "owner": str(STATE)})
        for job in jobs:
            if job["stage"] == "complete":
                validate_predictions(job)
                validate_summary(job)
                continue
            wait_gpu()
            job.update(stage="running", started_at=now())
            publish(stage="evaluating", current_suite=job["suite"])
            command = list(job["command"])
            profile = 0
            if job["attempts"] and "--execution-profile" in job["attempts"][-1]["command"]:
                previous_command = job["attempts"][-1]["command"]
                profile = int(previous_command[previous_command.index("--execution-profile") + 1])
                command += ["--execution-profile", str(profile)]
            while True:
                if len(job["attempts"]) >= 3:
                    raise RuntimeError(f"{job['suite']}: three recorded attempts exhausted")
                attempt = {"started_at": now(), "command": command}
                job["attempts"].append(attempt)
                code = execute(command, job["log"])
                attempt.update(returncode=code, finished_at=now())
                if code == 0:
                    break
                status_path = RESULTS / job["suite"] / "full" / NAME / "status.json"
                detail = json.loads(status_path.read_text()) if status_path.exists() else {}
                if job["suite"] == "highres" and code == 75:
                    job["attempts"].pop()  # GPU contention did not start an inference attempt.
                    wait_gpu()
                    time.sleep(args.poll_seconds)
                elif job["suite"] == "highres" and detail.get("gpu_oom") and profile < 2:
                    profile += 1
                    command = [*job["command"], "--execution-profile", str(profile)]
                    wait_gpu()
                elif (job["suite"] == "highres" and code == 2) or (job["suite"] != "highres" and "exit status 2" in Path(job["log"]).read_text()[-8000:]):
                    wait_gpu()
                else:
                    raise RuntimeError(f"{job['suite']} failed with return code {code}; see {job['log']}")
            validate_predictions(job)
            job.update(stage="summarizing")
            publish(stage="summarizing")
            if execute(job["summary_command"], job["log"], cpu=True):
                raise RuntimeError(f"{job['suite']} summary failed")
            validate_summary(job)
            job.update(stage="complete", finished_at=now())
            publish()
        publish(stage="complete", current_suite=None, finished_at=now())
        print("PERCEPTION_V3_EVALUATION_COMPLETE", flush=True)
        return 0
    except BaseException as exc:
        if active is not None and active.poll() is None:
            active.terminate()  # The owned launcher cleans up its own vLLM server.
            try:
                active.wait(timeout=60)
                state["child_pid"] = None
            except subprocess.TimeoutExpired:
                state["cleanup_pending_pid"] = active.pid
        for job in jobs:
            if job["stage"] in {"running", "summarizing"}:
                job["stage"] = "failed"
        publish(stage="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=f"{type(exc).__name__}: {exc}", finished_at=now())
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
