#!/usr/bin/env python3
"""Compose, inspect and launch domain OPD without changing the teacher recipe."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[2]
REPO = WORKSPACE / "pre-lab/repo"
EXPERIMENT = HERE.parent
sys.path[:0] = [str(WORKSPACE), str(REPO)]
from experiments.sharedexpert_c.distillation.config import (
    load_config, validate_config, missing_requirements, check_artifacts,
)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_overrides(config, run_dir, total_gpus, smoke=False):
    # Reuse the established multimodal/FSDP settings, then explicitly replace
    # every GRPO-specific setting. No imports of the active local launcher.
    from experiments.sharedexpert_c.training.launch import make_overrides as teacher_overrides
    base = json.loads((EXPERIMENT / "training/config/common.json").read_text())
    base.update(model=config["student_path"] or "UNRESOLVED_STUDENT",
                learning_rate=config["learning_rate"], temperature=config["temperature"],
                microbatch_max_tokens=config["prompt_length"] + config["response_length"])
    scorer_gpus = 1  # One separate GPU, teachers scored sequentially.
    student_gpus = total_gpus - scorer_gpus
    if student_gpus < 1:
        raise ValueError("Select at least two GPUs: student pool plus one teacher-scoring GPU")
    steps = 1 if smoke else config["steps"]
    batch = max(3, student_gpus) if smoke else config["batch_size"]
    if smoke:
        batch = ((batch + student_gpus - 1) // student_gpus) * student_gpus
    if batch * config["rollout_n"] % student_gpus:
        raise ValueError("batch_size * rollout_n must be divisible by student GPU count")
    items = teacher_overrides(base, {"data_file": config["data_path"] or "UNRESOLVED_DATA"},
                              student_gpus, Path(run_dir), steps, batch)
    values = {key: json.loads(value) for key, value in (item.split("=", 1) for item in items)}
    values.update({
        "data.max_prompt_length": config["prompt_length"],
        "data.max_response_length": config["response_length"],
        "data.train_max_samples": config["steps"] * config["batch_size"],
        "data.dataloader_num_workers": 0,
        "actor_rollout_ref.model.external_lib": "experiments.sharedexpert_c.distillation.verl_plugin",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "mopd_reverse_kl",
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0 if smoke else min(10, steps - 1),
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.rollout.n": config["rollout_n"],
        "actor_rollout_ref.rollout.prompt_length": config["prompt_length"],
        "actor_rollout_ref.rollout.response_length": config["response_length"],
        "actor_rollout_ref.rollout.max_model_len": config["prompt_length"] + config["response_length"],
        "actor_rollout_ref.rollout.max_num_batched_tokens": config["prompt_length"] + config["response_length"],
        "actor_rollout_ref.rollout.max_num_seqs": 8,
        "actor_rollout_ref.rollout.agent.num_workers": 1 if smoke else 2,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "domain_opd_single_turn",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(HERE / "agent_loops.yaml"),
        "algorithm.adv_estimator": "mopd_targets",
        "algorithm.norm_adv_by_std_in_grpo": False,
        "critic.enable": False,
        "reward_model.launch_reward_fn_async": False,
        "reward_manager.name": "HardTeacherRewardManager",
        "reward_manager.module.path": str(HERE / "reward_manager.py"),
        "+reward_model.reward_kwargs.mopd_config": config,
        "custom_reward_function.path": None,
        "trainer.project_name": "SharedExpert-Domain-OPD",
        "trainer.total_epochs": 1,
        "trainer.max_actor_ckpt_to_keep": 2,
        "trainer.rollout_data_dir": None,
        "ray_kwargs.ray_init.num_cpus": 12,
        "+ray_kwargs.ray_init.num_gpus": total_gpus,
    })
    return [key + "=" + json.dumps(value, ensure_ascii=False) for key, value in values.items()]


def compose_config(overrides):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    # Hydra's CLI grammar does not accept JSON-quoted mapping keys. Apply the
    # nested user mapping structurally, preserving arbitrary domain labels.
    prefix = "+reward_model.reward_kwargs.mopd_config="
    nested = [json.loads(item[len(prefix):]) for item in overrides if item.startswith(prefix)]
    with initialize_config_dir(config_dir=str(REPO / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=[item for item in overrides if not item.startswith(prefix)])
    if len(nested) != 1:
        raise ValueError("Exactly one domain OPD config is required")
    OmegaConf.update(cfg, "reward_model.reward_kwargs.mopd_config", nested[0], force_add=True)
    return cfg


def check_dataset(config):
    import pyarrow.parquet as pq
    from experiments.sharedexpert_c.distillation.routing import route_record
    from collections import Counter
    counts = Counter()
    rows = 0
    errors = []
    parquet = pq.ParquetFile(config["data_path"])
    if "prompt" not in parquet.schema_arrow.names:
        return {"errors": ["Data must have VERL prompt messages"], "rows": 0}
    for batch in parquet.iter_batches(batch_size=512):
        for row in batch.to_pylist():
            try:
                counts[route_record(row, config)] += 1
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"row {rows}: {exc}")
                if len(errors) >= 10:
                    return {"errors": errors, "rows": rows}
            if not isinstance(row.get("prompt"), list) or not row["prompt"]:
                errors.append(f"row {rows}: prompt must be nonempty chat messages")
            for image in row.get("images") or []:
                if isinstance(image, dict) and image.get("path") and not image.get("bytes"):
                    if not Path(image["path"]).is_file():
                        errors.append(f"row {rows}: missing image {image['path']}")
            rows += 1
    required = config["steps"] * config["batch_size"]
    if rows < required:
        errors.append(f"Need {required} rows for one ordered pass; found {rows}")
    return {"errors": errors[:20], "rows": rows, "teacher_counts": dict(counts),
            "sha256": sha256(config["data_path"])}


def protocol_identity(config, overrides):
    """Hash immutable model bytes as well as data/code; protect automatic resume."""
    model_hashes = {}
    paths = {"student": config["student_path"], **config["teacher_paths"]}
    for role, path in paths.items():
        directory = Path(path)
        files = sorted(p for p in directory.iterdir() if p.is_file() and
                       p.suffix in (".json", ".safetensors", ".jinja", ".txt"))
        model_hashes[role] = {p.name: sha256(p) for p in files}
    diff = subprocess.check_output(["git", "-C", str(REPO), "diff", "HEAD"], text=True)
    return {"config": config, "overrides": overrides, "models": model_hashes,
            "data_sha256": sha256(config["data_path"]),
            "code": {str(p.relative_to(WORKSPACE)): sha256(p) for p in
                     sorted(HERE.glob("*.py")) + sorted(HERE.glob("*.yaml")) + [EXPERIMENT / "training/launch.py",
                     EXPERIMENT / "training/agent_loop.py", EXPERIMENT / "training/config/common.json"]},
            "verl_commit": subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True).strip(),
            "verl_diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "configs/domain_opd.json")
    parser.add_argument("--gpus", default="0,1", help="All available GPUs for this run; one reserved for teachers")
    parser.add_argument("--run-name", default="domain_opd")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check", action="store_true", help="CPU config, dataset and checkpoint checks")
    mode.add_argument("--smoke", action="store_true", help="One update in a separate directory")
    mode.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.run_name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):
        parser.error("run-name must consist of letters, numbers, underscores or hyphens")
    gpus = args.gpus.split(",")
    if len(gpus) < 2 or len(set(gpus)) != len(gpus) or not all(g.isdigit() for g in gpus):
        parser.error("Select at least two distinct physical GPU indices")
    config = load_config(args.config)
    errors = validate_config(config, require_resolved=False)
    if errors:
        parser.error("; ".join(errors))
    if config.get("teacher_backend", {}).get("device", "cuda") != "cuda":
        parser.error("This launcher reserves a CUDA teacher worker; CPU scorer is for tests only")
    run_dir = EXPERIMENT / "runs" / (args.run_name + ("_smoke" if args.smoke else ""))
    overrides = make_overrides(config, run_dir, len(gpus), args.smoke)
    if args.dry_run:
        print(json.dumps({"config": config, "unresolved": missing_requirements(config),
                          "prompt_exposures": config["steps"] * config["batch_size"],
                          "sampled_responses": config["steps"] * config["batch_size"] * config["rollout_n"],
                          "student_gpu_count": len(gpus) - 1, "teacher_gpu_count": 1,
                          "run_dir": str(run_dir), "overrides": overrides}, indent=2, ensure_ascii=False))
        return
    errors = validate_config(config, require_resolved=True)
    if not errors:
        errors += check_artifacts(config)
    if errors:
        parser.error("; ".join(errors))
    dataset_report = check_dataset(config)
    if dataset_report["errors"]:
        parser.error("; ".join(dataset_report["errors"]))
    # No CUDA context, model load or Ray cluster is created by --check.
    os.environ.update({"PYTHONPATH": str(WORKSPACE) + ":" + str(REPO),
                       "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "1",
                       "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    cfg = compose_config(overrides)
    from experiments.sharedexpert_c.distillation import verl_plugin  # register on the driver
    from verl.utils.config import validate_config as validate_verl
    validate_verl(cfg, use_reference_policy=False, use_critic=False)
    if args.check:
        print(json.dumps({"ok": True, "dataset": dataset_report,
                          "gpu_training_verified": False}, indent=2, ensure_ascii=False))
        return
    from experiments.sharedexpert_c.training.launch import gpu_check
    with ExitStack() as stack:
        for gpu in sorted(gpus):
            lock = stack.enter_context(open(f"/tmp/sharedexpert-opd-gpu-{gpu}.lock", "a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_check(gpus)
        identity = protocol_identity(config, overrides)
        if not args.smoke:
            smoke_dir = EXPERIMENT / "runs" / (args.run_name + "_smoke")
            receipt = smoke_dir / "smoke_complete.json"
            if not receipt.is_file():
                raise RuntimeError("Run --smoke with this --run-name before --execute")
            tested = json.loads(receipt.read_text())
            stable_fields = ("config", "models", "data_sha256", "code", "verl_commit", "verl_diff_sha256")
            if tested.get("gpu_count") != len(gpus) or any(
                    tested.get("identity", {}).get(key) != identity[key] for key in stable_fields):
                raise RuntimeError("Smoke inputs/code/GPU count differ; rerun --smoke under a new run name")
        run_dir.mkdir(parents=True, exist_ok=True)
        protocol_path = run_dir / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != identity:
            raise RuntimeError("Run inputs/code changed; choose a new --run-name instead of resuming")
        latest = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
        target = 1 if args.smoke else config["steps"]
        if latest.exists() and int(latest.read_text().strip()) >= target:
            print(f"Run already reached step {target}: {run_dir}")
            return
        protocol_path.write_text(json.dumps(identity, indent=2, ensure_ascii=False))
        (run_dir / "preflight.json").write_text(json.dumps(dataset_report, indent=2))
        from omegaconf import OmegaConf
        config_path = run_dir / "resolved_config.yaml"
        config_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(gpus), PYTHONUNBUFFERED="1",
                   VLLM_USE_V1="1", WANDB_MODE="disabled", TENSORBOARD_DIR=str(run_dir / "tensorboard"))
        # Separate entry point imports the plugin on Ray drivers and actors.
        with open(run_dir / "training.log", "a", buffering=1) as log:
            result = subprocess.run([sys.executable, str(HERE / "run.py"), "--config", str(config_path)],
                                    cwd=WORKSPACE, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"OPD exited {result.returncode}; see {run_dir / 'training.log'}")
        if not latest.is_file() or int(latest.read_text().strip()) != target:
            raise RuntimeError("Training returned without the expected final checkpoint")
        if args.smoke:
            (run_dir / "smoke_complete.json").write_text(json.dumps(
                {"gpu_count": len(gpus), "identity": identity, "completed_step": 1}, indent=2))


if __name__ == "__main__":
    main()
