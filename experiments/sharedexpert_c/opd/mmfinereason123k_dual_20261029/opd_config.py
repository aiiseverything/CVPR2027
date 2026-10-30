"""Two independently initialized, sampled-token OPD runs on the curated intersection."""
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
EXPERIMENT = ROOT / "experiments/sharedexpert_c"
CURATION = EXPERIMENT / "dataset_curation/mmfinereason123k_20261029"
REPO = ROOT / "pre-lab/repo"
PYTHON = EXPERIMENT / ".venv-train/bin/python"
ORIGINALS = {"student": ROOT / "pre-lab/models/Qwen3.5-2B",
             "perception_v1": EXPERIMENT / "runs/perception/hf/global_step_100",
             "reasoning": EXPERIMENT / "runs/reasoning/hf/global_step_100"}
VIEWS = {k: HERE / "model_inputs" / k for k in ORIGINALS}
RUNS = {k: EXPERIMENT / "runs" / f"mmfinereason_opd_{k}_20261029" for k in ["perception_v1", "reasoning"]}
DATASET = HERE / "training_6300.parquet"
SETTINGS = {
    "student": "Qwen/Qwen3.5-2B", "teachers": ["perception_v1", "reasoning"],
    "initial_student_weights": str(ORIGINALS["student"]),
    "independent_initialization": True, "global_batch_size": 63, "optimizer_steps": 100,
    "rollouts_per_prompt": 1, "rollouts_per_run": 6300, "gpus": list(range(7)),
    "loss": "mean_sequences(mean_valid_response_tokens(stopgrad(logp_student-logp_fixed_teacher)*logp_student))",
    "loss_aggregation": "seq-mean-token-mean",
    "full_logit_distillation": False, "alpha": 1.0, "is_clip": None,
    "rollout_importance_weights": False, "advantage_normalization": False,
    "reward_loss": False, "additional_kl_loss": False, "entropy_coefficient": 0.0,
    "teacher_update": False, "teacher_input": "identical original image, question and student response prefix; no answer hints",
    "learning_rate": 1e-6, "warmup_steps": 10, "weight_decay": 0.0, "grad_clip": 1.0,
    "temperature": 1.0, "top_p": 1.0, "top_k": -1, "seed": 20261029,
    "max_prompt_tokens": 8192, "max_response_tokens": 8192, "thinking": True,
    "save_every": 25, "keep_checkpoints": 2, "precision": "fp32 master weights with bfloat16 forward",
    "image_size": {"shortest_edge": 4096, "longest_edge": 786432},
    "data_order": "same seeded permutation of full accepted set for both runs; first 6300 exposures, deterministic new permutations if cycling is needed",
    "gpu_smoke": False,
    "curation_completion_scope": "user_requested_early_stop_export",
    "authorization": "On 2026-10-30 the user stopped further screening and requested immediate formal OPD on the completed dual-correct set: batchsize=63 x 100 steps once for perception V1 and once for reasoning; CPU checks then formal training, no GPU smoke.",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(8<<20), b""):
            h.update(b)
    return h.hexdigest()


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n")
    temp.replace(path)


def baseline():
    spec = importlib.util.spec_from_file_location("sharedexpert_baseline_launcher", EXPERIMENT / "training/launch_100.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_model_views():
    """Keep original weights untouched; local processor settings match curation exactly."""
    result = {}
    for name, source in ORIGINALS.items():
        target = VIEWS[name]
        target.mkdir(parents=True, exist_ok=True)
        records = []
        for path in sorted(source.iterdir()):
            if not path.is_file() or path.suffix not in {".json", ".safetensors", ".jinja", ".txt"}:
                continue
            dest = target / path.name
            if path.name in {"preprocessor_config.json", "processor_config.json"}:
                value = json.loads(path.read_text())
                image_config = value.get("image_processor", value)
                image_config["size"] = SETTINGS["image_size"]
                for k, v in [("min_pixels",4096),("max_pixels",786432)]:
                    if k in image_config:
                        image_config[k] = v
                if dest.is_symlink():
                    assert dest.resolve() == path.resolve()
                    dest.unlink()
                if dest.exists():
                    assert json.loads(dest.read_text()) == value
                else:
                    atomic_json(dest, value)
            elif dest.is_symlink():
                assert dest.resolve() == path.resolve()
            elif not dest.exists():
                dest.symlink_to(path.resolve())
            else:
                raise ValueError(f"Unexpected non-symlink model file: {dest}")
            records.append({"name": path.name, "source": str(path), "source_size": path.stat().st_size,
                            "source_mtime_ns": path.stat().st_mtime_ns, "view": str(dest)})
        result[name] = {"source": str(source), "view": str(target), "files": records,
                        "change": "image processor pixel limits only; unchanged model weights/tokenizer/chat template"}
    atomic_json(HERE / "model_views.json", result)
    return result


def overrides(teacher):
    p = json.loads((EXPERIMENT / "training/config/common.json").read_text())
    p.update(model=str(VIEWS["student"]), steps=100, batch_size=63, group_size=1,
             prompt_length=8192, response_length=8192, microbatch_max_tokens=16384,
             seed=SETTINGS["seed"], rollout_max_num_seqs=9, learning_rate=1e-6,
             save_every=25, keep_checkpoints=2)
    values = {s.split("=",1)[0]:json.loads(s.split("=",1)[1]) for s in
              baseline().make_overrides(p, {"data_file": str(DATASET)}, 7, RUNS[teacher], 100, 63)}
    values.update({
        "actor_rollout_ref.model.external_lib": "experiments.sharedexpert_c.opd.mmfinereason123k_dual_20261029.memory_policy",
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.fsdp_config.offload_policy": False,
        "actor_rollout_ref.ref.fsdp_config.param_offload": False,
        "actor_rollout_ref.ref.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.ref.fsdp_config.offload_policy": False,
        "actor_rollout_ref.model.enable_activation_offload": False,
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb": 0,
        "data.max_prompt_length": 8192, "data.dataloader_num_workers": 2,
        "data.custom_cls.path": str(HERE / "matched_inputs.py"), "data.custom_cls.name": "MatchedImageDataset",
        "actor_rollout_ref.actor.policy_loss.loss_mode": "vopd",
        "actor_rollout_ref.actor.loss_agg_mode": SETTINGS["loss_aggregation"],
        "actor_rollout_ref.actor.self_distillation.full_logit_distillation": False,
        "actor_rollout_ref.actor.self_distillation.distillation_topk": None,
        "actor_rollout_ref.actor.self_distillation.alpha": 1.0,
        "actor_rollout_ref.actor.self_distillation.gamma": 1.0,
        "actor_rollout_ref.actor.self_distillation.is_clip": None,
        "actor_rollout_ref.actor.self_distillation.teacher_model_source": "fixed",
        "actor_rollout_ref.actor.self_distillation.teacher_model_path": str(VIEWS[teacher]),
        "actor_rollout_ref.actor.self_distillation.teacher_always_on": True,
        "actor_rollout_ref.actor.self_distillation.teacher_image_key": "teacher_images",
        "actor_rollout_ref.actor.self_distillation.teacher_prompt_mode": None,
        "actor_rollout_ref.actor.self_distillation.teacher_update_rate": 0.0,
        "actor_rollout_ref.actor.self_distillation.fallback_to_policy_loss_on_missing_teacher": False,
        "actor_rollout_ref.actor.self_distillation.include_environment_feedback": False,
        "actor_rollout_ref.actor.self_distillation.max_reprompt_len": 8192,
        # HF only accepts left/right here; CPU checks and the strict agent
        # reject over-budget inputs before either teacher prompt can truncate.
        "actor_rollout_ref.actor.self_distillation.reprompt_truncation": "right",
        "actor_rollout_ref.actor.self_distillation.log_prob_dump_dir": None,
        "actor_rollout_ref.ref.strategy": "fsdp2", "actor_rollout_ref.ref.fsdp_config.strategy": "fsdp2",
        "actor_rollout_ref.ref.fsdp_config.model_dtype": "bfloat16",
        "actor_rollout_ref.ref.fsdp_config.dtype": "bfloat16",
        "+actor_rollout_ref.ref.fsdp_config.load_tied_weights_on_gpu": True,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.n": 1,
        "actor_rollout_ref.rollout.prompt_length": 8192,
        "actor_rollout_ref.rollout.response_length": 8192,
        "actor_rollout_ref.rollout.max_model_len": 16384,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 16384,
        "actor_rollout_ref.rollout.max_num_seqs": 9,
        "actor_rollout_ref.rollout.agent.num_workers": 7,
        "+actor_rollout_ref.rollout.agent.agent_loop_manager_class": "experiments.sharedexpert_c.opd.mmfinereason123k_dual_20261029.seven_gpu_rollout.SevenGPUAgentLoopManager",
        "actor_rollout_ref.rollout.agent.default_agent_loop": "mfr_matched_single_turn",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(HERE / "agent_loops.yaml"),
        "+actor_rollout_ref.rollout.limit_images": 1,
        "algorithm.norm_adv_by_std_in_grpo": False,
        "algorithm.rollout_correction.rollout_is": None,
        "algorithm.rollout_correction.rollout_rs": None,
        "trainer.total_epochs": 1, "trainer.total_training_steps": 100,
        "+trainer.reuse_rollout_log_probs": True,
        "trainer.project_name": "MMFineReason-SingleToken-OPD",
        # The repository reserves 3 logical CPU slots per GPU in its placement
        # group, in addition to the driver and agent workers. This is scheduling
        # capacity; OMP/MKL/OpenBLAS remain one thread under the 8-core cgroup.
        "ray_kwargs.ray_init.num_cpus": 32,
    })
    return [key+"="+json.dumps(value, ensure_ascii=False) for key,value in values.items()]


def compose_config(teacher):
    sys.path[:0] = [str(ROOT), str(REPO)]
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(config_dir=str(REPO / "verl/trainer/config"), version_base=None):
        return compose(config_name="ppo_trainer", overrides=overrides(teacher))


def validate_config(cfg):
    assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 63
    assert cfg.trainer.total_training_steps == 100 and cfg.actor_rollout_ref.rollout.n == 1
    assert cfg.trainer.n_gpus_per_node == 7 and 63 % 7 == 0
    assert cfg.actor_rollout_ref.rollout.agent.num_workers == cfg.trainer.n_gpus_per_node == 7
    assert cfg.data.train_batch_size % cfg.actor_rollout_ref.rollout.agent.num_workers == 0
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size == cfg.actor_rollout_ref.rollout.data_parallel_size == cfg.actor_rollout_ref.rollout.pipeline_model_parallel_size == 1
    assert cfg.actor_rollout_ref.rollout.agent.agent_loop_manager_class == "experiments.sharedexpert_c.opd.mmfinereason123k_dual_20261029.seven_gpu_rollout.SevenGPUAgentLoopManager"
    assert cfg.actor_rollout_ref.model.external_lib == "experiments.sharedexpert_c.opd.mmfinereason123k_dual_20261029.memory_policy"
    for model in (cfg.actor_rollout_ref.actor, cfg.actor_rollout_ref.ref):
        assert not any(model.fsdp_config[k] for k in ('param_offload', 'optimizer_offload', 'offload_policy'))
    assert not cfg.actor_rollout_ref.model.enable_activation_offload
    assert cfg.actor_rollout_ref.rollout.engine_kwargs.vllm.cpu_offload_gb == 0
    assert 'swap_space' not in cfg.actor_rollout_ref.rollout.engine_kwargs.vllm
    assert cfg.ray_kwargs.ray_init.num_cpus >= (
        3 * cfg.trainer.n_gpus_per_node + 1 + cfg.actor_rollout_ref.rollout.agent.num_workers
    ), "Insufficient Ray logical CPU slots for placement group, driver and agents"
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "vopd"
    sd = cfg.actor_rollout_ref.actor.self_distillation
    assert sd.teacher_model_source == "fixed" and sd.teacher_always_on and sd.teacher_image_key == "teacher_images"
    assert sd.alpha == 1 and not sd.full_logit_distillation and sd.distillation_topk is None
    assert sd.is_clip is None and sd.teacher_update_rate == 0 and sd.teacher_prompt_mode is None
    assert not sd.fallback_to_policy_loss_on_missing_teacher and not sd.include_environment_feedback
    assert cfg.actor_rollout_ref.actor.loss_agg_mode == SETTINGS["loss_aggregation"]
    assert cfg.algorithm.rollout_correction.rollout_is is None and cfg.algorithm.rollout_correction.rollout_rs is None
    assert not cfg.algorithm.use_kl_in_reward and not cfg.actor_rollout_ref.actor.use_kl_loss
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0
    assert cfg.data.max_prompt_length == cfg.actor_rollout_ref.rollout.prompt_length == 8192
    assert cfg.data.max_response_length == cfg.actor_rollout_ref.rollout.response_length == 8192
    assert not cfg.data.shuffle and not cfg.trainer.val_before_train
    return True
