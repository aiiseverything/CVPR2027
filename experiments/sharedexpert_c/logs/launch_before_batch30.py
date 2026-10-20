#!/usr/bin/env python3
"""Prepare, check, smoke-test or train one teacher with automatic resume/export.

Examples:
  python launch.py perception --gpus 0,1,2,3 --dry-run
  .venv-train/bin/python training/launch.py perception --gpus 0,1,2,3 --smoke
  .venv-train/bin/python training/launch.py perception --gpus 0,1,2,3 --execute

GPU selection is mandatory. Busy GPUs cause a refusal; no existing process is
killed and no existing Ray cluster is reused.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

TRAINING = Path(__file__).resolve().parent
WORKSPACE = TRAINING.parents[2]
EXPERIMENT = TRAINING.parent
sys.path[:0] = [str(WORKSPACE), str(WORKSPACE / "pre-lab/repo"), str(TRAINING)]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def monitor_cgroup(run_dir):
    """Measure the shared container peak across training, checkpoint and CPU merge."""
    root = Path("/sys/fs/cgroup")
    stop = threading.Event()
    report = {"scope": "whole shared container including other project services",
              "sample_interval_seconds": 0.5, "memory_peak_bytes": 0,
              "working_set_peak_bytes": 0,
              "memory_limit_bytes": (root / "memory.max").read_text().strip(),
              "cpu_max": (root / "cpu.max").read_text().strip(),
              "ray_object_store_bytes": 1073741824, "omp_mkl_openblas_threads": 1,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    events_before = {key: int(value) for key, value in
                     (line.split() for line in (root / "memory.events").read_text().splitlines())}
    gpu_handles = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        gpu_handles = {gpu: pynvml.nvmlDeviceGetHandleByIndex(int(gpu))
                       for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu.isdigit()}
        report["gpu_peak_used_bytes"] = dict.fromkeys(gpu_handles, 0)
    except Exception as exc:
        report["gpu_sampling_error"] = str(exc)
    def sample():
        while not stop.is_set():
            current = int((root / "memory.current").read_text())
            stats = {key: int(value) for key, value in
                     (line.split() for line in (root / "memory.stat").read_text().splitlines())}
            report["memory_peak_bytes"] = max(report["memory_peak_bytes"], current)
            report["working_set_peak_bytes"] = max(report["working_set_peak_bytes"],
                                                    current - stats.get("inactive_file", 0))
            for gpu, handle in gpu_handles.items():
                report["gpu_peak_used_bytes"][gpu] = max(report["gpu_peak_used_bytes"][gpu],
                                                        pynvml.nvmlDeviceGetMemoryInfo(handle).used)
            stop.wait(0.5)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    try:
        yield report
    finally:
        stop.set()
        thread.join(timeout=2)
        report["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report["memory_events_delta"] = {key: int(value) - events_before[key] for key, value in
                                          (line.split() for line in (root / "memory.events").read_text().splitlines())}
        (run_dir / "resource_usage.json").write_text(json.dumps(report, indent=2))


def make_overrides(protocol, teacher, gpu_count, run_dir, steps, batch):
    p, t = protocol, teacher
    values = {
        "data.train_files": [str(WORKSPACE / t["data_file"])], "data.val_files": [],
        "data.train_batch_size": batch, "data.shuffle": False, "data.seed": p["seed"],
        "data.max_prompt_length": 2048, "data.max_response_length": 8192,
        "data.truncation": "error", "data.filter_overlong_prompts": False,
        "data.dataloader_num_workers": 4, "data.trust_remote_code": False,
        "data.image_patch_size": 16, "data.return_multi_modal_inputs": True,
        "+data.apply_chat_template_kwargs.enable_thinking": True,
        "actor_rollout_ref.model.path": str(WORKSPACE / p["model"]),
        "actor_rollout_ref.model.trust_remote_code": False,
        "actor_rollout_ref.model.custom_chat_template": None,
        "actor_rollout_ref.model.custom_chat_template_file": None,
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.use_fused_kernels": False,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.model.lora_rank": 0,
        "actor_rollout_ref.actor.strategy": p["strategy"],
        "actor_rollout_ref.actor.fsdp_config.strategy": p["strategy"],
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.fsdp_config.offload_policy": False,
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "fp32",
        "+actor_rollout_ref.actor.fsdp_config.load_tied_weights_on_gpu": True,
        "actor_rollout_ref.actor.fsdp_config.dtype": "bfloat16",
        "actor_rollout_ref.actor.fsdp_config.seed": p["seed"],
        "actor_rollout_ref.actor.ppo_mini_batch_size": batch,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": p["microbatch_max_tokens"],
        "actor_rollout_ref.actor.freeze_vision_tower": False,
        "actor_rollout_ref.actor.policy_loss.loss_mode": "vanilla",
        "actor_rollout_ref.actor.clip_ratio": 0.2,
        "actor_rollout_ref.actor.clip_ratio_low": p["clip_ratio_low"],
        "actor_rollout_ref.actor.clip_ratio_high": p["clip_ratio_high"],
        "actor_rollout_ref.actor.clip_ratio_c": p["dual_clip"],
        "actor_rollout_ref.actor.grad_clip": p["grad_clip"],
        "actor_rollout_ref.actor.loss_agg_mode": "seq-mean-token-mean",
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.kl_loss_coef": 0.0,
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.calculate_entropy": False,
        "actor_rollout_ref.actor.use_torch_compile": False,
        "actor_rollout_ref.actor.optim.lr": p["learning_rate"],
        "actor_rollout_ref.actor.optim.weight_decay": p["weight_decay"],
        "actor_rollout_ref.actor.optim.lr_warmup_steps": min(p["warmup_steps"], steps - 1),
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.data_loader_seed": p["seed"],
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model", "optimizer", "extra"],
        "actor_rollout_ref.rollout.name": "vllm", "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.n": 8, "actor_rollout_ref.rollout.temperature": p["temperature"],
        "actor_rollout_ref.rollout.top_p": p["top_p"], "actor_rollout_ref.rollout.top_k": p["top_k"],
        "actor_rollout_ref.rollout.prompt_length": 2048, "actor_rollout_ref.rollout.response_length": 8192,
        "actor_rollout_ref.rollout.max_model_len": 10240,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": p["rollout_memory_utilization"],
        "actor_rollout_ref.rollout.max_num_seqs": p["rollout_max_num_seqs"],
        "+actor_rollout_ref.rollout.limit_images": 8,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 10240,
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": False,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": p["microbatch_max_tokens"],
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.logprobs_mode": "raw_logprobs",
        "actor_rollout_ref.rollout.agent.num_workers": 4,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "sharedexpert_single_turn",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(TRAINING / "config/agent_loops.yaml"),
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.seed": p["seed"],
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.pass_config.fuse_allreduce_rms": False,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.kernel_config.enable_flashinfer_autotune": False,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb": 0,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.min_pixels": 4096,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_kwargs.max_pixels": 786432,
        "algorithm.adv_estimator": "grpo", "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.use_kl_in_reward": False, "algorithm.kl_ctrl.kl_coef": 0.0,
        "algorithm.rollout_correction.rollout_is": "token",
        "algorithm.rollout_correction.rollout_is_threshold": 2.0,
        "algorithm.rollout_correction.rollout_rs": None,
        "reward_model.enable": False, "reward_model.use_reward_loop": False,
        "reward_manager.source": "importlib", "reward_manager.name": "SharedExpertRewardManager",
        "reward_manager.module.path": str(TRAINING / "reward_manager.py"),
        "custom_reward_function.path": str(EXPERIMENT / "rewards.py"),
        "custom_reward_function.name": "compute_score",
        "trainer.project_name": "SharedExpert-C-Teachers", "trainer.experiment_name": run_dir.name,
        "trainer.n_gpus_per_node": gpu_count, "trainer.nnodes": 1,
        "trainer.total_epochs": 300, "trainer.total_training_steps": steps,
        "trainer.save_freq": p["save_every"], "trainer.max_actor_ckpt_to_keep": p["keep_checkpoints"],
        "trainer.resume_mode": "auto", "trainer.val_before_train": False,
        "trainer.test_freq": -1, "trainer.default_local_dir": str(run_dir / "checkpoints"),
        "trainer.rollout_data_dir": str(run_dir / "rollouts"),
        "trainer.logger": ["console", "tensorboard"], "trainer.use_legacy_worker_impl": "enable",
        "+trainer.reuse_rollout_log_probs": False,
        "ray_kwargs.ray_init.num_cpus": 24,
        "+ray_kwargs.ray_init.object_store_memory": 1073741824,
        "+ray_kwargs.ray_init.address": "local",
    }
    return [key + "=" + json.dumps(value, ensure_ascii=False) for key, value in values.items()]


def gpu_check(gpus):
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"
    ], text=True)
    known = {}
    for line in query.splitlines():
        idx, uid, used = [part.strip() for part in line.split(",")]
        known[idx] = (uid, int(used))
    for gpu in gpus:
        if gpu not in known:
            raise RuntimeError(f"GPU {gpu} not present")
        if known[gpu][1] > 1024:
            raise RuntimeError(f"GPU {gpu} is busy ({known[gpu][1]} MiB used); no process was stopped")
    apps = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"
    ], text=True)
    selected_uuids = {known[gpu][0] for gpu in gpus}
    for line in apps.splitlines():
        uid, pid = [part.strip() for part in line.split(",", 1)]
        if uid in selected_uuids:
            raise RuntimeError(f"Selected GPU has an existing compute process PID {pid}; no process was stopped")
    return {gpu: {"uuid": known[gpu][0], "used_mib": known[gpu][1]} for gpu in gpus}


def merge_checkpoint(run_dir, steps):
    source = run_dir / f"checkpoints/global_step_{steps}/actor"
    target = run_dir / f"hf/global_step_{steps}"
    marker = target / "sharedexpert_export_complete.json"
    if marker.is_file():
        return target
    if not (source / "fsdp_config.json").is_file():
        raise RuntimeError(f"Final step checkpoint is absent/incomplete: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    # A fresh directory avoids destroying an existing export after interruption.
    temporary = target.parent / (target.name + ".merging." + str(os.getpid()))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    subprocess.run([
        sys.executable, "-m", "verl.model_merger", "merge", "--backend", "fsdp",
        "--local_dir", str(source), "--target_dir", str(temporary), "--use_cpu_initialization"
    ], cwd=WORKSPACE / "pre-lab/repo", env=env, check=True)
    if not (temporary / "config.json").is_file() or not list(temporary.glob("*.safetensors")):
        raise RuntimeError(f"HF export validation failed: {temporary}")
    (temporary / "sharedexpert_export_complete.json").write_text(json.dumps({"step": steps, "source": str(source)}))
    if target.exists():
        raise RuntimeError(f"Refusing to replace unverified pre-existing export: {target}")
    temporary.rename(target)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("teacher", choices=["perception", "reasoning", "generalist"])
    parser.add_argument("--gpus", required=True, help="Explicit comma-separated physical GPU indices")
    parser.add_argument("--smoke-data", type=Path, help="Optional prepared parquet used only by --smoke")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="One step, 4 prompts, separate output directory")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    if args.smoke_data is not None and not args.smoke:
        parser.error("--smoke-data can only be used with --smoke")
    p = json.loads((TRAINING / "config/common.json").read_text())
    t = json.loads((TRAINING / f"config/{args.teacher}.json").read_text())
    if args.smoke_data is not None:
        t["data_file"] = str(args.smoke_data.resolve())
    gpus = args.gpus.split(",")
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(g.isdigit() for g in gpus):
        parser.error("This frozen protocol requires four distinct GPU indices")
    steps, batch = (1, 4) if args.smoke else (p["steps"], p["batch_size"])
    assert (p["steps"], p["batch_size"], p["group_size"]) == (300, 40, 8)
    run_dir = EXPERIMENT / "runs" / (args.teacher + ("_smoke" if args.smoke else ""))
    overrides = make_overrides(p, t, len(gpus), run_dir, steps, batch)
    if args.dry_run:
        print(json.dumps({"teacher": args.teacher, "steps": steps, "prompts": steps * batch,
                          "responses": steps * batch * 8, "gpus": gpus, "overrides": overrides}, indent=2))
        return
    env = dict(os.environ)
    env.update({"PYTHONPATH": str(WORKSPACE) + ":" + str(WORKSPACE / p["repository"]),
                "CUDA_VISIBLE_DEVICES": ",".join(gpus), "PYTHONUNBUFFERED": "1",
                "VLLM_USE_V1": "1", "TOKENIZERS_PARALLELISM": "false",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                "WANDB_MODE": "disabled", "TENSORBOARD_DIR": str(run_dir / "tensorboard")})
    # Imported modules and Ray descendants must inherit the same module paths.
    os.environ.update(env)
    if args.merge_only:
        print(merge_checkpoint(run_dir, steps))
        return
    from preflight import check_dataset, check_imports
    report = check_imports()
    report["repository_commit"] = subprocess.check_output(
        ["git", "-C", str(WORKSPACE / p["repository"]), "rev-parse", "HEAD"], text=True
    ).strip()
    if report["repository_commit"] != p["repository_commit"]:
        report["errors"].append("Training repository HEAD differs from frozen protocol")
    repository_diff = subprocess.check_output(
        ["git", "-C", str(WORKSPACE / p["repository"]), "diff", "HEAD", "--"], text=True
    )
    report["repository_diff_sha256"] = hashlib.sha256(repository_diff.encode()).hexdigest()
    report["dataset"] = check_dataset(WORKSPACE / t["data_file"], None if args.smoke else p["steps"] * p["batch_size"])
    if args.smoke and report["dataset"].get("rows", 0) < batch:
        report["dataset"]["errors"].append(f"Smoke dataset needs at least {batch} rows")
    if report["errors"] or report["dataset"]["errors"]:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(2)
    if not (EXPERIMENT / "rewards.py").is_file():
        raise RuntimeError("Missing experiments/sharedexpert_c/rewards.py")
    from experiments.sharedexpert_c import rewards
    rewards.validate_runtime()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(WORKSPACE / p["repository"] / "verl/trainer/config"), version_base=None):
        cfg = compose(config_name="ppo_trainer", overrides=overrides)
    if args.check:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(str(run_dir.resolve()).encode()).hexdigest()[:24]
    with open(Path("/tmp") / ("sharedexpert-train-" + lock_key + ".lock"), "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = {"protocol": p, "teacher": t, "gpus": gpus, "smoke": args.smoke,
                    "rollout_replica_seeds": {str(rank): p["seed"] + rank for rank in range(len(gpus))},
                    "data_sha256": sha256(WORKSPACE / t["data_file"]),
                    "reward_sha256": sha256(EXPERIMENT / "rewards.py"),
                    "repository_diff_sha256": report["repository_diff_sha256"],
                    "training_code_sha256": {name: sha256(TRAINING / name) for name in
                                              ("launch.py", "run.py", "agent_loop.py", "reward_manager.py")}}
        protocol["training_environment_sha256"] = {
            name: sha256(EXPERIMENT / "logs" / name) for name in
            ("training-environment.freeze.txt", "flash-attention-build.json", "flash-attention-validation.json")
            if (EXPERIMENT / "logs" / name).is_file()
        }
        protocol_path = run_dir / "protocol.json"
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
            raise RuntimeError("Protocol/data/reward changed since this run. Use a separately named experiment.")
        protocol_path.write_text(json.dumps(protocol, ensure_ascii=False, indent=2))
        (run_dir / "repository.patch").write_text(repository_diff)
        config_path = run_dir / "resolved_config.yaml"
        resolved = OmegaConf.to_yaml(cfg, resolve=True)
        if config_path.exists() and config_path.read_text() != resolved:
            raise RuntimeError("Resolved config differs from the original run; refusing silent resume drift")
        config_path.write_text(resolved)
        report["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (run_dir / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        latest = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
        completed = int(latest.read_text().strip()) if latest.exists() else 0
        if completed > steps:
            raise RuntimeError(f"Checkpoint step {completed} exceeds intended endpoint {steps}")
        with monitor_cgroup(run_dir):
            if completed < steps:
                report["gpu_snapshot"] = gpu_check(gpus)
                (run_dir / "preflight.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
                with open(run_dir / "training.log", "a", buffering=1) as log:
                    log.write(f"\nLaunch/resume at step {completed}, endpoint {steps}\n")
                    subprocess.run([sys.executable, str(TRAINING / "run.py"), "--config", str(config_path)],
                                   cwd=WORKSPACE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            exported = merge_checkpoint(run_dir, steps)
        print(json.dumps({"teacher": args.teacher, "step": steps, "hf_model": str(exported), "smoke": args.smoke}))


if __name__ == "__main__":
    main()
