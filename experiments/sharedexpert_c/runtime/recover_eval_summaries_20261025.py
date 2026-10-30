#!/usr/bin/env python3
"""Recover interrupted CPU summaries from existing predictions; no GPU jobs."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

RUNTIME = Path(__file__).resolve().parent
EVAL = RUNTIME.parent / "evaluation"
ROOT = RUNTIME.parents[2]
STATE = RUNTIME / "eval_extensions_20261025.json"
PYTHON = "/root/.venv-eval/bin/python"
MODELS = ["baseline", "generalist", "perception-step100", "reasoning-step100", "perception-v2-step100"]


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    state = json.loads(STATE.read_text())
    if state.get("stage") == "complete":
        raise SystemExit("The extension evaluation is already complete; no recovery needed.")
    for pid in [state.get("pid")] + [s.get("child_pid") for s in state.get("summaries", [])]:
        if pid and (Path("/proc") / str(pid)).exists():
            raise SystemExit(f"Recorded PID {pid} exists; investigate before recovering.")
    if not all(j.get("stage") == "complete" and j.get("returncode") == 0 for j in state["jobs"]):
        raise SystemExit("Inference jobs are not all complete.")
    audit = RUNTIME / "eval_summary_recovery_20261025"
    audit.mkdir(exist_ok=False)
    shutil.copy2(STATE, audit / "original_state.json")
    record = {"stage": "recovering", "started_at": now(), "gpu_runs": False,
              "original_stage": state["stage"], "commands": []}
    receipt = audit / "receipt.json"

    def save():
        receipt.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

    save()
    try:
        checker = json.loads((RUNTIME / "eval_extensions_20261025_checker.json").read_text())
        source = Path(checker["source_site_packages"])
        target = Path(checker["overlay"])
        if not target.exists():
            stage = target.with_name(target.name + ".recovering")
            stage.mkdir(parents=True, exist_ok=False)

            def copy(item):
                relative = Path(item["relative_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise RuntimeError("Invalid checker inventory path")
                data = (source / relative).read_bytes()
                if len(data) != item["size_bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
                    raise RuntimeError(f"Checker source differs from frozen receipt: {relative}")
                destination = stage / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                if hashlib.sha256(destination.read_bytes()).hexdigest() != item["sha256"]:
                    raise RuntimeError(f"Checker copy failed: {relative}")

            with ThreadPoolExecutor(max_workers=16) as pool:
                list(pool.map(copy, checker["files"]))
            stage.rename(target)
            record["checker_recovered"] = {"files": len(checker["files"]), "sha256_mismatches": 0,
                                            "path": str(target), "bytes": checker["total_bytes"]}
        else:
            raise RuntimeError("Checker appeared during recovery; inspect concurrent activity")
        save()
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1",
                   PYTHONUNBUFFERED="1", PYTHONPATH=str(target), HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        commands = [
            ("checker_tests", [PYTHON, "-m", "unittest", "discover", "-s", str(EVAL), "-p", "test_general.py", "-v"]),
            ("blink_mmvp", [PYTHON, str(EVAL / "summarize.py"), "--suite", str(EVAL / "suite_blink_mmvp.jsonl"), "--output-root", str(EVAL / "results_blink_mmvp")]),
            ("merged_main_perception", [PYTHON, str(EVAL / "merge_summary.py")]),
            ("expanded_all", [PYTHON, str(EVAL / "merge_summary.py"), "--include-general"]),
        ]
        for label, command in commands:
            item = {"label": label, "command": command, "started_at": now()}
            record["commands"].append(item)
            save()
            print(label, flush=True)
            started = time.monotonic()
            with (audit / f"{label}.log").open("w") as log:
                result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=1800)
            item.update(returncode=result.returncode, finished_at=now(), elapsed_seconds=round(time.monotonic() - started, 2))
            save()
            if result.returncode:
                raise RuntimeError(f"{label} failed; see recovery log")
        report = json.loads((EVAL / "results_expanded/summary.json").read_text())
        for model in MODELS:
            status = report["status"][model]
            if any(status[suite]["status"] != "complete" for suite in ["main", "blink_mmvp", "general"]):
                raise RuntimeError(f"Incomplete expanded results: {model}")
            if status["combined"]["n_predicted"] != 7162 or status["combined"]["missing"] or status["combined"]["api_errors"]:
                raise RuntimeError(f"Unexpected combined counts: {model}")
        record.update(stage="complete", finished_at=now(), verified_models=MODELS, predictions_per_model=7162)
        save()
        # Preserve the interrupted supervisor's record and append recovery provenance.
        for item in state.get("summaries", []):
            if item.get("child_pid") and "returncode" not in item:
                item.update(stage="interrupted_process_absent", former_child_pid=item["child_pid"], child_pid=None)
        state.update(stage="complete", finished_at=now(), updated_at=now(),
                     summary_recovery={"receipt": str(receipt), "original_state": str(audit / "original_state.json"),
                                       "gpu_runs": False, "completed_commands": record["commands"]})
        STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"stage": "complete", "report": str(EVAL / "results_expanded/summary.md")}), flush=True)
    except BaseException as exc:
        record.update(stage="failed", error=f"{type(exc).__name__}: {exc}", failed_at=now())
        save()
        raise


if __name__ == "__main__":
    main()
