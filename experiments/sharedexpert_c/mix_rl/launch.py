#!/usr/bin/env python3
"""Independent mixed-data GRPO baseline: 200 steps x 32 prompts x 8 responses."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
WORKSPACE = HERE.parents[2]
REPO = WORKSPACE / "pre-lab/repo"
sys.path[:0] = [str(WORKSPACE), str(REPO)]


def load_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError("Configuration must be an object")
    for key, expected in (("algorithm", "grpo"), ("steps", 200),
                          ("batch_size", 32), ("rollout_n", 8)):
        if config.get(key) != expected or type(config.get(key)) is not type(expected):
            raise ValueError(f"{key} must be {expected!r} for this comparison")
    for key in ("student_path", "data_path", "opd_config_path"):
        value = config.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{key} must be a nonempty local path")
        path = Path(value).expanduser()
        config[key] = str((path if path.is_absolute() else WORKSPACE / path).resolve())
    for key in ("prompt_length", "response_length"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if (config["prompt_length"], config["response_length"]) != (2048, 8192):
        raise ValueError("The established runner requires prompt/response lengths 2048/8192")
    for key in ("temperature", "top_p", "learning_rate"):
        value = config.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if config["temperature"] != 1 or config["top_p"] != 1 or config.get("top_k") != -1:
        raise ValueError("This comparison uses temperature=1, top_p=1, top_k=-1")
    digest = config.get("expected_data_sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError("expected_data_sha256 must pin the exact OPD parquet bytes")
    return config


def make_overrides(config, run_dir, gpu_count, smoke=False):
    from experiments.sharedexpert_c.training.launch import make_overrides as teacher_overrides
    if type(gpu_count) is not int or gpu_count not in (1, 2, 4, 8):
        raise ValueError("Use 1, 2, 4 or 8 student GPUs")
    base = json.loads((EXPERIMENT / "training/config/common.json").read_text())
    base.update(model=config["student_path"], learning_rate=config["learning_rate"],
                temperature=config["temperature"], top_p=config["top_p"], top_k=config["top_k"],
                microbatch_max_tokens=config["prompt_length"] + config["response_length"])
    batch = max(4, gpu_count) if smoke else config["batch_size"]
    steps = 1 if smoke else config["steps"]
    values = {key: json.loads(value) for key, value in (item.split("=", 1) for item in
              teacher_overrides(base, {"data_file": config["data_path"]},
                                gpu_count, Path(run_dir), steps, batch))}
    values.update({
        "data.max_prompt_length": config["prompt_length"],
        "data.max_response_length": config["response_length"],
        "data.train_max_samples": config["steps"] * config["batch_size"],
        "data.dataloader_num_workers": 0,
        "actor_rollout_ref.model.external_lib": None,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0 if smoke else 10,
        "actor_rollout_ref.rollout.n": config["rollout_n"],
        "actor_rollout_ref.rollout.prompt_length": config["prompt_length"],
        "actor_rollout_ref.rollout.response_length": config["response_length"],
        "actor_rollout_ref.rollout.max_model_len": config["prompt_length"] + config["response_length"],
        "actor_rollout_ref.rollout.max_num_batched_tokens": config["prompt_length"] + config["response_length"],
        "actor_rollout_ref.rollout.max_num_seqs": 8,
        "actor_rollout_ref.rollout.agent.num_workers": 1 if smoke else 2,
        "critic.enable": False,
        "reward_model.launch_reward_fn_async": False,
        "trainer.project_name": "SharedExpert-Mix-RL",
        "trainer.total_epochs": 1,
        "trainer.max_actor_ckpt_to_keep": 2,
        "trainer.rollout_data_dir": None,
        "ray_kwargs.ray_init.num_cpus": 12,
        "+ray_kwargs.ray_init.num_gpus": gpu_count,
    })
    return [key + "=" + json.dumps(value, ensure_ascii=False) for key, value in values.items()]


def compose_config(overrides):
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "verl/trainer/config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def protocol_identity(config, overrides):
    source_files = sorted(HERE.glob("*.py")) + [
        EXPERIMENT / "training/launch.py", EXPERIMENT / "training/run.py", EXPERIMENT / "training/agent_loop.py",
        EXPERIMENT / "training/config/common.json", EXPERIMENT / "training/config/agent_loops.yaml",
        EXPERIMENT / "training/reward_manager.py", EXPERIMENT / "rewards.py",
        EXPERIMENT / "distillation/config.py", Path(config["opd_config_path"]),
    ]
    model_files = sorted(p for p in Path(config["student_path"]).iterdir()
                         if p.is_file() and p.suffix in (".json", ".safetensors", ".jinja", ".txt"))
    diff = subprocess.check_output(["git", "-C", str(REPO), "diff", "HEAD"], text=True)
    return {"config": config, "overrides": overrides, "data_sha256": sha256(config["data_path"]),
            "student_files": {p.name: sha256(p) for p in model_files},
            "source_sha256": {str(p): sha256(p) for p in source_files},
            "verl_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
            "verl_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def run_completed(run_dir, target, smoke=False):
    latest = Path(run_dir) / "checkpoints/latest_checkpointed_iteration.txt"
    if not latest.is_file() or int(latest.read_text().strip()) < target:
        return False
    if smoke and not (Path(run_dir) / "smoke_complete.json").is_file():
        raise RuntimeError("Smoke checkpoint exists without a success receipt (possibly interrupted); "
                           "repeat --smoke using a new --run-name mix_rl_<name>")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/mix_rl.json")
    parser.add_argument("--gpus", default="0,1", help="All selected GPUs train the single mixed-data policy")
    parser.add_argument("--run-name", default="mix_rl")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if not args.run_name.startswith("mix_rl") or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):
        parser.error("run-name must start with mix_rl and contain only letters, numbers, _ or -")
    gpus = args.gpus.split(",")
    if len(gpus) not in (1, 2, 4, 8) or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        parser.error("Select 1, 2, 4 or 8 distinct physical GPU indices")
    config = load_config(args.config)
    run_dir = EXPERIMENT / "runs" / (args.run_name + ("_smoke" if args.smoke else ""))
    overrides = make_overrides(config, run_dir, len(gpus), args.smoke)
    if args.dry_run:
        print(json.dumps({"config": config, "prompt_exposures": 6400, "sampled_responses": 51200,
                          "responses_per_step": 256, "student_gpu_count": len(gpus),
                          "teacher_gpu_count": 0, "run_dir": str(run_dir),
                          "overrides": overrides}, indent=2, ensure_ascii=False))
        return
    os.environ.update({"PYTHONPATH": str(WORKSPACE) + ":" + str(REPO), "TOKENIZERS_PARALLELISM": "false",
                       "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    if args.merge_only:
        from experiments.sharedexpert_c.training.launch import merge_checkpoint
        print(merge_checkpoint(run_dir, config["steps"]))
        return
    from experiments.sharedexpert_c.mix_rl.preflight import check_inputs
    report = check_inputs(config)
    if report["errors"]:
        parser.error("; ".join(report["errors"]))
    cfg = compose_config(overrides)
    from verl.utils.config import validate_config
    validate_config(cfg, use_reference_policy=False, use_critic=False)
    if args.check:
        print(json.dumps({"ok": True, "gpu_training_verified": False, **report}, indent=2, ensure_ascii=False))
        return
    from experiments.sharedexpert_c.training.launch import gpu_check
    with ExitStack() as stack:
        # Share lock names with the OPD launcher, preventing concurrent launches
        # before either child has allocated GPU memory.
        for gpu in sorted(gpus):
            lock = stack.enter_context(open(f"/tmp/sharedexpert-opd-gpu-{gpu}.lock", "a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_check(gpus)
        identity = protocol_identity(config, overrides)
        if not args.smoke:
            receipt = EXPERIMENT / "runs" / (args.run_name + "_smoke") / "smoke_complete.json"
            if not receipt.is_file():
                raise RuntimeError("Complete --smoke with this --run-name before --execute")
            tested = json.loads(receipt.read_text())
            stable = ("config", "data_sha256", "student_files", "source_sha256", "verl_commit", "verl_diff_sha256")
            if tested.get("gpu_count") != len(gpus) or any(tested.get("identity", {}).get(k) != identity[k] for k in stable):
                raise RuntimeError("Smoke protocol/code/GPU count changed; choose a new run name and repeat smoke")
        run_dir.mkdir(parents=True, exist_ok=True)
        protocol = run_dir / "protocol.json"
        if protocol.exists() and json.loads(protocol.read_text()) != identity:
            raise RuntimeError("Refusing changed run inputs; choose a new --run-name")
        latest = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
        target = 1 if args.smoke else config["steps"]
        if run_completed(run_dir, target, args.smoke):
            print(f"Already completed step {target}: {run_dir}")
            return
        protocol.write_text(json.dumps(identity, ensure_ascii=False, indent=2))
        (run_dir / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        from omegaconf import OmegaConf
        config_path = run_dir / "resolved_config.yaml"
        config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus), PYTHONUNBUFFERED="1",
                   VLLM_USE_V1="1", WANDB_MODE="disabled", TENSORBOARD_DIR=str(run_dir / "tensorboard"))
        with open(run_dir / "training.log", "a", buffering=1) as log:
            result = subprocess.run([sys.executable, str(HERE / "run.py"), "--config", str(config_path)],
                                    cwd=WORKSPACE, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"Mix-RL exited {result.returncode}; see {run_dir / 'training.log'}")
        if not latest.is_file() or int(latest.read_text().strip()) != target:
            raise RuntimeError("Training exited without the expected final checkpoint")
        if args.smoke:
            (run_dir / "smoke_complete.json").write_text(json.dumps(
                {"gpu_count": len(gpus), "completed_step": 1, "identity": identity}, indent=2))


if __name__ == "__main__":
    main()
