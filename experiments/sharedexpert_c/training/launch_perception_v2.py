#!/usr/bin/env python3
"""Isolated perception-v2 run, using the active generalist's frozen protocol.

CPU: --gpus 0,1 --dry-run / --check. After the generalist finishes, run
--smoke (six prompts, one step), then --execute (100 steps, batch 30).
No existing process is killed and the generalist's files are never modified.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

TRAINING = Path(__file__).resolve().parent
EXPERIMENT = TRAINING.parent
WORKSPACE = TRAINING.parents[2]
sys.path[:0] = [str(TRAINING), str(WORKSPACE), str(WORKSPACE / "pre-lab/repo")]
import launch_local as local

BASELINE = EXPERIMENT / "runs/generalist"
DATA_DIR = EXPERIMENT / "data_pipeline/prepared"
RUN_DIR = EXPERIMENT / "runs/perception_v2"
SMOKE_DIR = EXPERIMENT / "runs/perception_v2_smoke"
CATEGORIES = dict(zip(("object_attribute", "grounding", "ocr_document", "spatial_detail",
                       "existence_confusion", "general_vqa_replay"), (20, 20, 20, 15, 15, 10)))
SHARED_FILES = ("launch.py", "launch_local.py", "run.py", "agent_loop.py", "reward_manager.py")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def load_protocol(gpus):
    baseline = read_json(BASELINE / "protocol.json")
    p = read_json(TRAINING / "config/common_local_100.json")
    t = read_json(TRAINING / "config/perception_v2.json")
    require(p == baseline["protocol"], "Local common protocol differs from active generalist protocol")
    require(gpus == baseline["gpus"] == ["0", "1"], "Use the same physical GPU layout as the generalist: 0,1")
    require((p["steps"], p["batch_size"], p["group_size"]) == (100, 30, 8), "Unexpected baseline budget")
    require(t["teacher"] == "perception_v2", "Incorrect teacher identity")
    require(Path(WORKSPACE / t["data_file"]).resolve() == (DATA_DIR / "perception_v2.parquet").resolve(),
            "Perception-v2 must use its own prepared data")
    require(t["category_weights"] == {key: value / 100 for key, value in CATEGORIES.items()},
            "Category weights must be exactly 20:20:20:15:15:10")
    require(Path(p["model"]).name == "Qwen3.5-4B" and Path(p["model"]).is_dir(),
            "The unchanged original Qwen3.5-4B model is required")
    return p, t, baseline


def recipe_identity(p, baseline):
    """Refuse code/reward/environment drift from the actual generalist run."""
    shared = {name: local.sha256(TRAINING / name) for name in SHARED_FILES}
    require(shared == baseline["training_code_sha256"], "Shared trainer changed since generalist launch")
    reward = local.sha256(EXPERIMENT / "rewards.py")
    require(reward == baseline["reward_sha256"], "Reward changed since generalist launch")
    repo = WORKSPACE / p["repository"]
    commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    require(commit == p["repository_commit"], "Training repository HEAD changed")
    patch = subprocess.check_output(["git", "-C", str(repo), "diff", "HEAD", "--"], text=True)
    patch_hash = hashlib.sha256(patch.encode()).hexdigest()
    require(patch_hash == baseline["repository_diff_sha256"], "Training repository patch changed")
    environment = {name: local.sha256(EXPERIMENT / "logs" / name)
                   for name in baseline["training_environment_sha256"]}
    require(environment == baseline["training_environment_sha256"], "Validated training environment records changed")
    return {"reward_sha256": reward, "repository_diff_sha256": patch_hash,
            "training_code_sha256": {**shared, "launch_perception_v2.py": local.sha256(__file__)},
            "training_environment_sha256": environment,
            "baseline_protocol_sha256": local.sha256(BASELINE / "protocol.json"),
            "baseline_resolved_config_sha256": local.sha256(BASELINE / "resolved_config.yaml")}, patch


def make_overrides(p, teacher, gpus, smoke=False):
    teacher = dict(teacher)
    if smoke:
        teacher["data_file"] = str(DATA_DIR / "perception_v2_smoke.parquet")
    steps, batch = (1, 6) if smoke else (p["steps"], p["batch_size"])
    run = SMOKE_DIR if smoke else RUN_DIR
    # Shared implementation preserves every optimizer, image, reward and thinking setting.
    return local.make_overrides(p, teacher, len(gpus), run, steps, batch), teacher, run, steps, batch


def validate_mixture(path, smoke=False):
    import pyarrow.parquet as pq
    from preflight import check_dataset
    expected_rows = 6 if smoke else 3000
    report = check_dataset(path, expected_rows)
    if report["errors"]:
        return report
    rows = pq.read_table(path, columns=["extra_info"]).to_pylist()
    categories = [(row["extra_info"] or {}).get("perception_category") for row in rows]
    expected = {key: 1 if smoke else weight * 30 for key, weight in CATEGORIES.items()}
    actual = dict(Counter(categories))
    report["category_counts"] = actual
    if actual != expected:
        report["errors"].append(f"Wrong category counts: expected {expected}, found {actual}")
    if not smoke:
        per_block = {key: weight // 5 for key, weight in CATEGORIES.items()}
        for start in range(0, len(categories), 20):
            if dict(Counter(categories[start:start + 20])) != per_block:
                report["errors"].append(f"Mixture drift in 20-prompt block beginning at {start}")
                break
    return report


def compose_and_verify(overrides, p, teacher, run, steps, batch):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(WORKSPACE / p["repository"] / "verl/trainer/config"), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    # Compare the entire resolved tree against what actually trained the generalist.
    expected = OmegaConf.load(BASELINE / "resolved_config.yaml")
    updates = {"data.train_files": [str(WORKSPACE / teacher["data_file"])],
               "trainer.experiment_name": run.name,
               "trainer.default_local_dir": str(run / "checkpoints"),
               "trainer.rollout_data_dir": str(run / "rollouts")}
    if steps == 1:
        updates.update({"trainer.total_training_steps": 1, "data.train_batch_size": batch,
                        "actor_rollout_ref.actor.ppo_mini_batch_size": batch,
                        # Hydra derives this inactive critic default from actor batch size.
                        "critic.ppo_mini_batch_size": batch,
                        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0})
    for key, value in updates.items():
        OmegaConf.update(expected, key, value)
    require(OmegaConf.to_container(config, resolve=True) == OmegaConf.to_container(expected, resolve=True),
            "Resolved configuration differs from the generalist beyond allowed dataset/run/smoke changes")
    return config


def validate_export(run, steps, gpu_count):
    checkpoint = run / f"checkpoints/global_step_{steps}/actor"
    export = run / f"hf/global_step_{steps}"
    require((checkpoint / "fsdp_config.json").is_file(), "Missing final FSDP checkpoint")
    for kind in ("model", "optim"):
        require(len(list(checkpoint.glob(f"{kind}_world_size_{gpu_count}_rank_*.pt"))) == gpu_count,
                f"Missing {kind} checkpoint shards for {gpu_count} GPUs")
    require(read_json(export / "sharedexpert_export_complete.json")["step"] == steps,
            "Export marker has the wrong step")
    require((export / "config.json").is_file(), "Missing HF config")
    shards = list(export.glob("*.safetensors"))
    require(bool(shards) and all(path.stat().st_size > 0 for path in shards), "Missing or empty HF weights")
    index = export / "model.safetensors.index.json"
    if index.is_file():
        require(set(read_json(index)["weight_map"].values()) == {path.name for path in shards},
                "HF index and exported weight shards do not match")
    return export


def smoke_weight_paths():
    paths = []
    require(not SMOKE_DIR.is_symlink(), "Smoke directory must not be a symlink")
    for subtree in ("checkpoints", "hf"):
        root = SMOKE_DIR / subtree
        require(not root.is_symlink(), "Smoke artifact roots must not be symlinks")
        for path in sorted(root.rglob("*")):
            require(not path.is_symlink(), f"Refusing symlink in smoke artifacts: {path}")
            if path.is_file() and path.suffix in (".pt", ".safetensors"):
                paths.append(path)
    return paths


def safe_smoke_path(relative, weight=False):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts, "Invalid smoke receipt path")
    if weight:
        require(relative.parts[0] in ("checkpoints", "hf") and relative.suffix in (".pt", ".safetensors"),
                "Smoke cleanup is restricted to checkpoint/HF weight files")
    path = SMOKE_DIR / relative
    require(path.resolve().is_relative_to(SMOKE_DIR.resolve()) and not path.is_symlink(),
            "Smoke receipt path escaped its own run directory")
    return path


def retained_smoke_evidence():
    weights = set(smoke_weight_paths())
    evidence = {}
    for path in sorted(SMOKE_DIR.rglob("*")):
        require(not path.is_symlink(), f"Refusing symlink in smoke evidence: {path}")
        if path.is_file() and path not in weights and path.name != "smoke_cleanup.json":
            evidence[str(path.relative_to(SMOKE_DIR))] = local.sha256(path)
    return evidence


def validate_cleanup_receipt(identity, allow_prepared=False):
    receipt = read_json(SMOKE_DIR / "smoke_cleanup.json")
    allowed = ("prepared", "complete") if allow_prepared else ("complete",)
    require(receipt.get("status") in allowed, "Smoke cleanup is incomplete; explicitly retry --cleanup-smoke")
    require(receipt["recipe_identity"] == identity, "Cleaned smoke recipe changed")
    require(receipt["protocol_sha256"] == local.sha256(SMOKE_DIR / "protocol.json"), "Cleaned smoke protocol changed")
    require(receipt["validation_sha256"] == local.sha256(SMOKE_DIR / "smoke_validation.json"),
            "Cleaned smoke validation receipt changed")
    require(receipt["retained_evidence_sha256"] == retained_smoke_evidence(), "Retained smoke evidence changed")
    weights = receipt["weights"]
    require(bool(weights) and len({item["path"] for item in weights}) == len(weights), "Invalid smoke weight archive")
    names = {item["path"] for item in weights}
    required = {f"checkpoints/global_step_1/actor/{kind}_world_size_2_rank_{rank}.pt"
                for kind in ("model", "optim") for rank in range(2)}
    require(required <= names and any(name.startswith("hf/global_step_1/") and name.endswith(".safetensors")
                                     for name in names), "Smoke archive must cover both model/optimizer ranks and HF weights")
    recorded = set()
    for item in weights:
        path = safe_smoke_path(item["path"], weight=True)
        require(item["size_bytes"] > 0 and len(item["sha256"]) == 64, "Invalid archived weight fingerprint")
        recorded.add(path)
        if receipt["status"] == "complete":
            require(not path.exists(), "Cleaned smoke weight unexpectedly reappeared")
        elif path.exists():
            require(path.stat().st_size == item["size_bytes"] and local.sha256(path) == item["sha256"],
                    "A smoke weight changed during interrupted cleanup")
    require(set(smoke_weight_paths()) <= recorded, "Unexpected unrecorded smoke weight")
    return receipt


def cleanup_smoke(identity):
    """Retire only this smoke's validated, reproducible large weight artifacts."""
    lock_key = hashlib.sha256(str(SMOKE_DIR.resolve()).encode()).hexdigest()[:24]
    with open(Path("/tmp") / ("sharedexpert-train-" + lock_key + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt_path = SMOKE_DIR / "smoke_cleanup.json"
        validate_smoke(identity, allow_prepared_cleanup=True)
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt["status"] == "complete":
                return receipt
        else:
            # Hash before deletion; a prepared receipt alone never authorizes formal training.
            weights = [{"path": str(path.relative_to(SMOKE_DIR)), "size_bytes": path.stat().st_size,
                        "sha256": local.sha256(path)} for path in smoke_weight_paths()]
            require(bool(weights), "No validated smoke weights are available for cleanup")
            receipt = {"status": "prepared", "recipe_identity": identity,
                       "protocol_sha256": local.sha256(SMOKE_DIR / "protocol.json"),
                       "validation_sha256": local.sha256(SMOKE_DIR / "smoke_validation.json"),
                       "retained_evidence_sha256": retained_smoke_evidence(), "weights": weights,
                       "total_weight_bytes": sum(item["size_bytes"] for item in weights),
                       "prepared_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
            write_json(receipt_path, receipt)
        for item in receipt["weights"]:
            safe_smoke_path(item["path"], weight=True).unlink(missing_ok=True)
        receipt.update(status="complete", completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write_json(receipt_path, receipt)
        validate_cleanup_receipt(identity)
        return receipt


def validate_smoke(identity, allow_prepared_cleanup=False):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    protocol = read_json(SMOKE_DIR / "protocol.json")
    for key, value in identity.items():
        require(protocol[key] == value, f"Smoke protocol no longer matches: {key}")
    require(protocol["data_sha256"] == local.sha256(DATA_DIR / "perception_v2_smoke.parquet"),
            "Smoke dataset changed after validation")
    groups = defaultdict(list)
    for path in sorted((SMOKE_DIR / "rollouts").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if int(row["step"]) != 1:
                continue
            require(bool(row.get("group_uid")), "Missing real rollout group UID")
            require(0 < int(row["prompt_tokens"]) <= 2048 and 0 < int(row["response_tokens"]) <= 8192,
                    "Smoke exceeded prompt/response token budget")
            groups[row["group_uid"]].append(float(row["score"]))
    require(len(groups) == 6 and all(len(scores) == 8 for scores in groups.values()),
            "Smoke must contain six groups of eight responses")
    require(all(math.isfinite(score) for scores in groups.values() for score in scores), "Nonfinite smoke reward")
    useful = sum(max(scores) - min(scores) > 1e-12 for scores in groups.values())
    require(useful > 0, "No nonuniform reward group in smoke; review smoke examples before formal training")
    events = EventAccumulator(str(SMOKE_DIR / "tensorboard"), size_guidance={"scalars": 0}).Reload()
    required = ("actor/pg_loss", "actor/grad_norm", "actor/lr", "rollout_corr/rollout_is_mean",
                "rollout_corr/rollout_is_min", "rollout_corr/rollout_is_max")
    scalars = {tag: [event.value for event in events.Scalars(tag) if event.step == 1]
               for tag in required if tag in events.Tags()["scalars"]}
    for tag in required:
        require(bool(scalars.get(tag)) and all(math.isfinite(v) for v in scalars[tag]), f"Missing/nonfinite {tag}")
    require(max(scalars["actor/grad_norm"]) > 0 and min(scalars["actor/lr"]) > 0,
            "Smoke must demonstrate nonzero gradient and learning rate")
    cleaned = (SMOKE_DIR / "smoke_cleanup.json").exists()
    if cleaned:
        validate_cleanup_receipt(identity, allow_prepared_cleanup)
        export = SMOKE_DIR / "hf/global_step_1"
    else:
        export = validate_export(SMOKE_DIR, 1, 2)
    resources = read_json(SMOKE_DIR / "resource_usage.json")
    require(resources["memory_events_delta"]["oom"] == resources["memory_events_delta"]["oom_kill"] == 0,
            "An OOM was observed during smoke")
    require(resources["working_set_peak_bytes"] > 0 and len(resources["gpu_peak_used_bytes"]) == 2,
            "Missing smoke resource measurements")
    report = {"passed": True, "step": 1, "prompts": 6, "responses": 48,
              "nonuniform_reward_groups": useful, "metrics": scalars, "hf_export": str(export),
              "protocol_sha256": local.sha256(SMOKE_DIR / "protocol.json"),
              "data_sha256": protocol["data_sha256"], "recipe_identity": identity,
              "validated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    if not cleaned:
        write_json(SMOKE_DIR / "smoke_validation.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ("dry-run", "check", "smoke", "execute", "merge-only", "cleanup-smoke"):
        mode.add_argument("--" + name, action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    p, teacher, baseline = load_protocol(gpus)
    smoke_mode = args.smoke or args.cleanup_smoke
    overrides, teacher, run, steps, batch = make_overrides(p, teacher, gpus, smoke_mode)
    identity, patch = recipe_identity(p, baseline)
    if args.smoke:
        require(not (SMOKE_DIR / "smoke_cleanup.json").exists(),
                "Smoke weights have been archived; use --execute or --cleanup-smoke without rewriting smoke evidence")
    if args.dry_run:
        print(json.dumps({"teacher": "perception_v2", "steps": steps, "prompts": steps * batch,
                          "responses": steps * batch * p["group_size"], "gpus": gpus,
                          "category_weights": teacher["category_weights"], "overrides": overrides,
                          "recipe_identity": identity}, ensure_ascii=False, indent=2))
        return 0
    env = dict(os.environ, PYTHONPATH=str(WORKSPACE) + ":" + str(WORKSPACE / p["repository"]),
               CUDA_VISIBLE_DEVICES="" if args.check or args.merge_only or args.cleanup_smoke else ",".join(gpus),
               PYTHONUNBUFFERED="1", VLLM_USE_V1="1", TOKENIZERS_PARALLELISM="false",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               WANDB_MODE="disabled", TENSORBOARD_DIR=str(run / "tensorboard"))
    os.environ.update(env)
    from preflight import check_imports
    from omegaconf import OmegaConf
    report = check_imports()
    report["dataset"] = validate_mixture(WORKSPACE / teacher["data_file"], smoke_mode)
    require(not report["errors"] and not report["dataset"]["errors"], json.dumps(report, ensure_ascii=False))
    from experiments.sharedexpert_c import rewards
    rewards.validate_runtime()
    config = compose_and_verify(overrides, p, teacher, run, steps, batch)
    if args.check:
        print(json.dumps({"passed": True, "cpu_only": True, "dataset": report["dataset"],
                          "configuration_matches_generalist": True, "recipe_identity": identity},
                         ensure_ascii=False, indent=2))
        return 0
    if args.cleanup_smoke:
        receipt = cleanup_smoke(identity)
        print(json.dumps({"status": receipt["status"], "retired_weight_bytes": receipt["total_weight_bytes"],
                          "receipt": str(SMOKE_DIR / "smoke_cleanup.json")}))
        return 0
    if args.execute:
        validate_smoke(identity)
    run.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(str(run.resolve()).encode()).hexdigest()[:24]
    with open(Path("/tmp") / ("sharedexpert-train-" + lock_key + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = {"protocol": p, "teacher": teacher, "gpus": gpus, "smoke": args.smoke,
                    "actual_steps": steps, "actual_batch_size": batch,
                    "rollout_replica_seeds": {str(rank): p["seed"] + rank for rank in range(len(gpus))},
                    "data_sha256": local.sha256(WORKSPACE / teacher["data_file"]), **identity}
        path = run / "protocol.json"
        require(not path.exists() or read_json(path) == protocol, "Refusing resume after protocol/data drift")
        resolved = OmegaConf.to_yaml(config, resolve=True)
        config_path = run / "resolved_config.yaml"
        require(not config_path.exists() or config_path.read_text() == resolved, "Refusing resolved config drift")
        write_json(path, protocol)
        config_path.write_text(resolved)
        (run / "repository.patch").write_text(patch)
        write_json(run / "preflight.json", report)
        latest = run / "checkpoints/latest_checkpointed_iteration.txt"
        completed = int(latest.read_text().strip()) if latest.exists() else 0
        require(0 <= completed <= steps, "Checkpoint step exceeds intended endpoint")
        if args.merge_only:
            require(completed == steps, "Cannot merge an unfinished training run")
        with local.monitor_cgroup(run):
            if completed < steps:
                report["gpu_snapshot"] = local.gpu_check(gpus)
                write_json(run / "preflight.json", report)
                with (run / "training.log").open("a", buffering=1) as log:
                    log.write(f"\nLaunch/resume at step {completed}, endpoint {steps}\n")
                    subprocess.run([sys.executable, str(TRAINING / "run.py"), "--config", str(config_path)],
                                   cwd=WORKSPACE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            local.merge_checkpoint(run, steps)
        exported = validate_export(run, steps, len(gpus))
        if args.smoke:
            validate_smoke(identity)
        else:
            write_json(run / "training_complete.json", {"teacher": "perception_v2", "step": steps,
                       "hf_model": str(exported), "protocol_sha256": local.sha256(path),
                       "data_sha256": protocol["data_sha256"],
                       "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        print(json.dumps({"teacher": "perception_v2", "step": steps, "hf_model": str(exported), "smoke": args.smoke}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
