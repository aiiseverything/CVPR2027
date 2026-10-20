#!/usr/bin/env python3
"""Validate the actual composed config, processor truncation and reward plumbing.

No model weights or CUDA allocation. This supplements, but cannot replace, a real
GPU forward/backward, checkpoint and export smoke test.
"""

import asyncio
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
TRAINING = Path(__file__).resolve().parent
WORKSPACE = TRAINING.parents[2]
sys.path[:0] = [str(WORKSPACE), str(WORKSPACE / "pre-lab/repo"), str(TRAINING)]


async def check_processor(cfg):
    from PIL import Image
    from transformers import AutoProcessor
    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from experiments.sharedexpert_c.training.agent_loop import SharedExpertSingleTurn

    processor = AutoProcessor.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
    loop = SharedExpertSingleTurn(
        trainer_config=DictConfigWrap(cfg), server_manager=None, tokenizer=processor.tokenizer,
        processor=processor, dataset_cls=RLHFDataset, dataset_config=cfg.data,
    )
    image = Image.new("RGB", (1024, 768), color=(180, 100, 40))
    def messages(text):
        return [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": text}]}]
    short = await loop.apply_chat_template(messages("Count objects. Put the answer in \\boxed{}."), images=[image])
    long = await loop.apply_chat_template(
        messages("Observe this image. " + "a lengthy irrelevant phrase " * 1800 + " Put the answer in \\boxed{}."),
        images=[image],
    )
    image_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert len(long) <= 2048
    assert short.count(image_id) > 0 and long.count(image_id) == short.count(image_id)
    assert long[-8:] == short[-8:], "Truncation damaged the assistant/thinking prefix"
    assert "\\boxed{}" in processor.tokenizer.decode(long), "Final format instruction was removed"
    class ProbeServer:
        async def generate(self, **kwargs):
            return SimpleNamespace(token_ids=[processor.tokenizer.eos_token_id], log_probs=None, routed_experts=None)
    loop.server_manager = ProbeServer()
    multi_image_reports = {}
    for image_count in (3, 8):
        output = await loop.run({}, raw_prompt=[{"role": "user", "content": [
            *[{"type": "image", "image": image, "max_pixels": 786432} for _ in range(image_count)],
            {"type": "text", "text": "Observe the images. " * 800 + " Put answer in \\boxed{}."},
        ]}])
        assert len(output.prompt_ids) <= 2048
        assert 0 < output.prompt_ids.count(image_id) <= 1536
        assert output.prompt_ids[-8:] == short[-8:]
        multi_image_reports[image_count] = {"prompt_tokens": len(output.prompt_ids),
                                           "image_tokens": output.prompt_ids.count(image_id)}
    return {"short_prompt_tokens": len(short), "truncated_prompt_tokens": len(long),
            "image_tokens_preserved": long.count(image_id), "multi_image": multi_image_reports}


def check_reward_plumbing(cfg):
    import numpy as np
    import torch
    from transformers import AutoTokenizer
    from verl import DataProto
    from experiments.sharedexpert_c import rewards
    from experiments.sharedexpert_c.training.reward_manager import SharedExpertRewardManager

    tok = AutoTokenizer.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True)
    prompt = tok.apply_chat_template([{"role": "user", "content": "Count objects. Put answer in \\boxed{}."}],
                                     tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
    good = tok.encode("There are two. </think>\n\\boxed{2}", add_special_tokens=False) + [tok.eos_token_id]
    incomplete = tok.encode("There are two. \\boxed{2}", add_special_tokens=False)
    width = max(len(good), len(incomplete))
    response = torch.tensor([good + [tok.pad_token_id] * (width-len(good)),
                             incomplete + [tok.pad_token_id] * (width-len(incomplete))])
    attention = torch.tensor([[1] * (len(prompt)+len(good)) + [0] * (width-len(good)),
                              [1] * (len(prompt)+len(incomplete)) + [0] * (width-len(incomplete))])
    gt = json.dumps({"task": "exact", "answers": ["2"], "format": "boxed"})
    data = DataProto.from_dict(
        tensors={"prompts": torch.tensor([prompt, prompt]), "responses": response, "attention_mask": attention},
        non_tensors={"data_source": np.array(["smoke", "smoke"], dtype=object),
                     "reward_model": np.array([{"ground_truth": gt}, {"ground_truth": gt}], dtype=object),
                     "extra_info": np.array([{}, {}], dtype=object)},
    )
    result = SharedExpertRewardManager(tok, compute_score=rewards.compute_score)(data)
    scores = result["reward_tensor"].sum(-1).tolist()
    assert scores == [1.0, 0.0], scores
    rewards.validate_runtime()
    return {"complete_correct": scores[0], "truncated_reasoning_only": scores[1]}


def check_logprob_without_entropy(cfg):
    """Execute the real worker method with a CPU fake model and inspect its contract."""
    from contextlib import nullcontext
    import torch
    from omegaconf import OmegaConf
    from verl import DataProto
    from verl.trainer.ppo.ray_trainer import should_reuse_rollout_log_probs_as_old_log_probs

    source = WORKSPACE / "pre-lab/repo/verl/workers/fsdp_workers.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActorRolloutRefWorker")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "compute_log_prob")
    method.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    scope = {"DataProto": DataProto}
    exec(compile(module, str(source), "exec"), scope)
    requested = []
    def compute_log_prob(data, calculate_entropy):
        requested.append(calculate_entropy)
        result = {"log_probs": torch.zeros(2, 4)}
        if calculate_entropy:
            result["entropys"] = torch.ones(2, 4)
        return result
    worker = SimpleNamespace(_is_actor=True, _is_offload_param=False,
                             config=cfg.actor_rollout_ref, tokenizer=SimpleNamespace(pad_token_id=0),
                             actor=SimpleNamespace(compute_log_prob=compute_log_prob),
                             ulysses_sharding_manager=nullcontext(), world_size=1)
    output = scope["compute_log_prob"](worker, DataProto.from_dict(tensors={"responses": torch.zeros(2, 4)}))
    assert requested == [False] and "entropys" not in output.batch
    responses_per_step = cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n
    batch = DataProto.from_dict(tensors={"rollout_log_probs": torch.zeros(responses_per_step, 4)})
    assert not should_reuse_rollout_log_probs_as_old_log_probs(cfg, batch)
    with __import__("omegaconf").open_dict(cfg):
        cfg.trainer.reuse_rollout_log_probs = True
    assert should_reuse_rollout_log_probs_as_old_log_probs(cfg, batch)
    cfg.trainer.reuse_rollout_log_probs = False
    return {"worker_entropy_requested": requested[0], "entropy_tensor_returned": False,
            "old_logprobs_recomputed": True, "rollout_is": cfg.algorithm.rollout_correction.rollout_is}


def main():
    # Import registers the dataclass config groups used by ppo_trainer.
    import verl.trainer.main_ppo  # noqa: F401
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from launch import make_overrides

    p = json.loads((TRAINING / "config/common.json").read_text())
    reports = {}
    with initialize_config_dir(config_dir=str(WORKSPACE / p["repository"] / "verl/trainer/config"), version_base=None):
        for name in ["perception", "reasoning", "generalist"]:
            teacher = json.loads((TRAINING / f"config/{name}.json").read_text())
            cfg = compose(config_name="ppo_trainer", overrides=make_overrides(
                p, teacher, 4, TRAINING.parent / "runs" / name, p["steps"], p["batch_size"]
            ))
            OmegaConf.resolve(cfg)
            assert cfg.algorithm.adv_estimator == "grpo"
            assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"
            assert not cfg.actor_rollout_ref.actor.self_distillation.full_logit_distillation
            assert cfg.actor_rollout_ref.actor.self_distillation.distillation_topk is None
            assert not cfg.actor_rollout_ref.actor.use_kl_loss
            assert not cfg.algorithm.use_kl_in_reward
            assert cfg.actor_rollout_ref.actor.entropy_coeff == 0
            assert not cfg.actor_rollout_ref.actor.calculate_entropy
            assert not cfg.trainer.reuse_rollout_log_probs
            assert cfg.actor_rollout_ref.rollout.n == 8
            assert cfg.trainer.total_training_steps == 300 and cfg.data.train_batch_size == 30
            assert cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 30
            assert cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n // 4 == 60
            assert cfg.data.train_batch_size * cfg.actor_rollout_ref.rollout.n % 4 == 0
            assert cfg.data.max_prompt_length == 2048 and cfg.data.max_response_length == 8192
            from verl.utils.config import omega_conf_to_dataclass
            actor = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
            assert actor.fsdp_config.load_tied_weights_on_gpu
            assert not actor.fsdp_config.param_offload
            assert not actor.fsdp_config.optimizer_offload
            assert not actor.fsdp_config.offload_policy
            assert cfg.actor_rollout_ref.rollout.gpu_memory_utilization == 0.4
            # The pinned Qwen3.5 linear-attention forward has no cu_seqlens input.
            # Never pack two examples into one recurrent stream.
            assert not actor.use_dynamic_bsz and actor.ppo_micro_batch_size_per_gpu == 1
            assert not cfg.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
            assert cfg.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu == 1
            assert actor.fsdp_config.seed == p["seed"]
            reports[name] = {"hydra_compose": "passed", "prepared_rows": 12000,
                             "prompts": 9000, "responses": 72000, "responses_per_rank": 60}
    reports["processor"] = asyncio.run(check_processor(cfg))
    reports["reward_manager"] = check_reward_plumbing(cfg)
    reports["logprob_contract"] = check_logprob_without_entropy(cfg)
    # Exercise the real FSDP wrapper selector with transformers5's set contract,
    # replacing only GPU sharding itself so this remains a CPU validation.
    import torch
    from verl.utils import fsdp_utils
    from unittest.mock import patch
    wrapped = []
    tiny = torch.nn.Sequential(torch.nn.Linear(2, 2))
    tiny._no_split_modules = {"Linear"}
    tiny.config = SimpleNamespace(tie_word_embeddings=True)
    with patch.object(fsdp_utils, "fully_shard", lambda module, **kwargs: wrapped.append(type(module).__name__)):
        fsdp_utils.apply_fsdp2(tiny, {}, {"wrap_policy": {}})
    assert wrapped == ["Linear", "Sequential"], wrapped
    reports["fsdp2_transformers5_wrap_selector"] = {"passed": True, "wrapped": wrapped}
    # Execute the actual final engine-argument assignment (after engine_kwargs),
    # without starting a server, to verify independent deterministic replica RNGs.
    server_tree = ast.parse((WORKSPACE / p["repository"] /
                            "verl/workers/rollout/vllm_rollout/vllm_async_server.py").read_text())
    seed_assignments = [node for node in ast.walk(server_tree)
                        if isinstance(node, ast.Assign) and len(node.targets) == 1
                        and isinstance(node.targets[0], ast.Subscript)
                        and ast.unparse(node.targets[0]) == "args['seed']"]
    assert len(seed_assignments) == 1
    seed_code = compile(ast.Module(body=seed_assignments, type_ignores=[]), "actual_rollout_seed", "exec")
    replica_seeds = []
    for rank in range(4):
        namespace = {"args": {"seed": p["seed"]}, "self": SimpleNamespace(replica_rank=rank)}
        exec(seed_code, namespace)
        replica_seeds.append(namespace["args"]["seed"])
    assert replica_seeds == [20261019, 20261020, 20261021, 20261022]
    reports["independent_rollout_replica_seeds"] = replica_seeds
    reports["gpu_smoke"] = "not_performed_by_this_script"
    path = TRAINING.parent / "logs/cpu-validation.json"
    path.write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2))


if __name__ == "__main__":
    main()
