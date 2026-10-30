#!/usr/bin/env python3
"""Fast-start the already-resolved perception_v3 training without CPU preflight.

User request (2026-10-30): skip the repeated dataset/config/import preflight and
start the formal run directly. The dataset was already independently replayed
(runtime/perception_v3_dataset_validation.json) and the resolved config was
produced by the frozen launcher earlier; this script starts run.py with that
config, keeps the cgroup/GPU watchdog, and performs the final merge/export.

The resolved configuration controls model dtype and FSDP offloading; the
single-GPU OMP/MKL/OPENBLAS thread count is 8. No training hyperparameter or
data selection is changed. Recovery may perform separate CPU-only checks
before invoking this entry point.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from omegaconf import OmegaConf

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "training"), str(EXPERIMENT / "data_pipeline"),
                str(WORKSPACE), str(WORKSPACE / "pre-lab/repo")]
import launch_perception_v3_single_gpu as wrapper

RUN = EXPERIMENT / "runs/perception_v3"
DATASET = EXPERIMENT / "data_pipeline/prepared/perception_v3_filter/dataset"
GPU_LOCK = Path("/tmp/sharedexpert-perception-v3-gpu-0.lock")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser_args = sys.argv[1:]
    if len(parser_args) != 2 or parser_args[0] != "--config":
        raise SystemExit("usage: fast_start_perception_v3.py --config <resolved yaml>")
    config = Path(parser_args[1]).resolve()
    if not config.is_file():
        raise SystemExit(f"missing resolved config: {config}")
    configuration = OmegaConf.load(config)
    fsdp = configuration.actor_rollout_ref.actor.fsdp_config
    RUN.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update({
        "PYTHONPATH": str(WORKSPACE) + ":" + str(WORKSPACE / "pre-lab/repo"),
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONUNBUFFERED": "1", "VLLM_USE_V1": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "8",
        "WANDB_MODE": "disabled", "TENSORBOARD_DIR": str(RUN / "tensorboard"),
    })
    # Refuse a busy GPU without stopping anyone.
    wrapper.la.local.gpu_check(["0"])
    lock = GPU_LOCK.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not (RUN / "protocol.json").exists():
        write_json(RUN / "protocol.json", {
            "gpus": ["0"], "mode": "fast_start_no_preflight",
            "config_sha256": sha(config),
            "dataset_manifest_sha256": sha(DATASET / "manifest.json"),
        })
    receipt = {
        "schema": 1, "mode": "fast_start_no_preflight", "devices": ["0"],
        "config": str(config), "config_sha256": sha(config),
        "preflight_skipped": True,
        "preflight_skip_reason": "User requested no preflight; dataset was already independently "
                                 "validated and the resolved config was produced earlier.",
        "dataset_manifest_sha256": sha(DATASET / "manifest.json"),
        "dataset_validation_sha256": sha(EXPERIMENT / "runtime/perception_v3_dataset_validation.json"),
        "offload_policy": bool(fsdp.get("offload_policy", False)),
        "param_offload": bool(fsdp.get("param_offload", False)),
        "optimizer_offload": bool(fsdp.get("optimizer_offload", False)),
        "model_dtype": str(fsdp.get("model_dtype", "fp32")),
        "gpu_memory_utilization": float(configuration.actor_rollout_ref.rollout.gpu_memory_utilization),
        "enforce_eager": bool(configuration.actor_rollout_ref.rollout.enforce_eager),
        "cpu_threads": 8, "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_reward_changed": False,
    }
    write_json(RUN / "fast_start.json", receipt)
    print(json.dumps({"stage": "fast_start", **receipt}), flush=True)
    log_path = RUN / "training.log"
    code = 0
    try:
        with wrapper.monitored_cgroup(RUN):
            with log_path.open("a", buffering=1) as log:
                log.write("\nFormal training resume; " + json.dumps({
                    key: receipt[key] for key in (
                        "started_at", "model_dtype", "offload_policy", "param_offload",
                        "optimizer_offload", "enforce_eager", "cpu_threads")
                }) + "\n")
                result = subprocess.run(
                    [sys.executable, str(EXPERIMENT / "training/run.py"), "--config", str(config)],
                    cwd=WORKSPACE, env=environment, stdout=log, stderr=subprocess.STDOUT)
                code = result.returncode
        if code == 0 and (RUN / "checkpoints/global_step_100/actor/fsdp_config.json").is_file():
            export = wrapper.la.local.merge_checkpoint(RUN, 100)
            wrapper.validate_export(RUN)
            write_json(RUN / "fast_start_complete.json", {
                "status": "complete", "step": 100, "hf_model": str(export),
                "config_sha256": receipt["config_sha256"],
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        elif code == 0:
            write_json(RUN / "fast_start_complete.json", {
                "status": "run_py_returned_without_step100",
                "config_sha256": receipt["config_sha256"],
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
        else:
            write_json(RUN / "fast_start_failed.json", {
                "status": "failed", "returncode": code,
                "config_sha256": receipt["config_sha256"],
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            })
    finally:
        lock.close()
    raise SystemExit(code)


if __name__ == "__main__":
    main()
