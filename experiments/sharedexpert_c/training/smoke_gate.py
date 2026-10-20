#!/usr/bin/env python3
"""Finish this isolated setup, validate kernels, then execute only one RL smoke step."""

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
EXPERIMENT = TRAINING.parent
WORKSPACE = TRAINING.parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-pid", type=int, required=True)
    parser.add_argument("--reuse-validated-environment", action="store_true",
                        help="Skip unchanged installation/import/kernel checks on a code-only smoke retry")
    args = parser.parse_args()
    runtime = EXPERIMENT / "runtime"
    runtime.mkdir(exist_ok=True)
    report = {"pid": os.getpid(), "status": "running", "stage": "waiting_for_attention_build"}
    def update(**values):
        report.update(values, updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (runtime / "smoke_gate.json").write_text(json.dumps(report, indent=2))
    update()
    proc = Path(f"/proc/{args.build_pid}/stat")
    start = proc.read_text().split()[21] if proc.exists() else None
    try:
        while proc.exists():
            stat = proc.read_text().split()
            if stat[21] != start or stat[2] == "Z":
                break
            time.sleep(10)
        build = json.loads((EXPERIMENT / "logs/flash-attention-build.json").read_text())
        if not build.get("wheel_sha256"):
            raise RuntimeError("Attention build did not complete; inspect setup_train_env.log")
        common = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                      PYTHONPATH=str(WORKSPACE) + ":" + str(WORKSPACE / "pre-lab/repo"))
        def run(stage, command, env=None):
            update(stage=stage)
            with open(EXPERIMENT / "logs" / (stage + ".log"), "a") as output:
                subprocess.run(command, cwd=WORKSPACE, env=env or dict(common, CUDA_VISIBLE_DEVICES=""),
                               stdout=output, stderr=subprocess.STDOUT, check=True)
        if args.reuse_validated_environment:
            validation = json.loads((EXPERIMENT / "logs/flash-attention-validation.json").read_text())
            if not validation.get("passed"):
                raise RuntimeError("Cannot reuse an environment without successful GPU attention validation")
            frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            if frozen != (EXPERIMENT / "logs/training-environment.freeze.txt").read_text():
                raise RuntimeError("Environment changed since kernel/import validation")
        else:
            run("install_repository", [sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation",
                                       "-e", str(WORKSPACE / "pre-lab/repo")])
            run("dependency_metadata", [sys.executable, str(TRAINING / "check_dependency_metadata.py")])
            with open(EXPERIMENT / "logs/training-environment.freeze.txt", "w") as stream:
                subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=stream, check=True)
            run("import_preflight", [sys.executable, str(TRAINING / "preflight.py"), "--imports-only"])
            run("flash_attention_gpu_validation", [sys.executable, str(TRAINING / "validate_flash_attention.py")],
                env=dict(common, CUDA_VISIBLE_DEVICES="3"))
        run("cpu_validation_final", [sys.executable, str(TRAINING / "validate_cpu.py")])
        run("training_smoke", [sys.executable, str(TRAINING / "launch.py"), "perception", "--gpus", "3,4,5,6",
                               "--smoke", "--smoke-data", str(EXPERIMENT / "data_pipeline/prepared/smoke.parquet")],
            env=common)
        marker = EXPERIMENT / "runs/perception_smoke/hf/global_step_1/sharedexpert_export_complete.json"
        if not marker.exists():
            raise RuntimeError("Smoke returned without an exported step-1 checkpoint")
        update(status="complete", stage="smoke_exported", export=str(marker.parent))
    except Exception as exc:
        update(status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
