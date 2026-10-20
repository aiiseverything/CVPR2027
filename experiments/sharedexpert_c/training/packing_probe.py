#!/usr/bin/env python3
"""Isolated, real verl/Qwen3.5/FSDP2 packing probe; never changes training files.

Prepare on CPU first. Run torchrun only after the smoke owner releases GPUs:
  .venv-train/bin/python training/packing_probe.py --prepare-only
  CUDA_VISIBLE_DEVICES=3,4,5,6 OMP_NUM_THREADS=1 .venv-train/bin/torchrun \
    --standalone --nproc-per-node=4 training/packing_probe.py --execute

Paths above are relative to experiments/sharedexpert_c. All outputs stay under
logs/packing_probe. A saved real rollout is retokenized, never generated anew.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

# Set before importing torch, numpy, tokenizer, FLA, or model code.
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

TRAINING = Path(__file__).resolve().parent
EXPERIMENT = TRAINING.parent
ROOT = EXPERIMENT.parents[1]
REPO = ROOT / "pre-lab/repo"
sys.path[:0] = [str(ROOT), str(REPO)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=EXPERIMENT / "runs/perception_smoke/resolved_config.yaml")
    ap.add_argument("--data", type=Path, default=EXPERIMENT / "data_pipeline/prepared/smoke.parquet")
    ap.add_argument("--rollouts", type=Path)
    ap.add_argument("--output", type=Path, default=EXPERIMENT / "logs/packing_probe")
    ap.add_argument("--prepare-only", action="store_true")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--forward-repeats", type=int, default=3)
    ap.add_argument("--skip-gradients", action="store_true", help="Diagnostic only; cannot establish gradient equivalence")
    ap.add_argument("--segmented-linear-control", action="store_true",
                    help="After native controls, test a process-only segmented linear decoder when causal contamination exceeds measured controls")
    args = ap.parse_args()
    if args.prepare_only == args.execute:
        ap.error("Choose exactly one of --prepare-only or --execute")
    if args.forward_repeats < 2:
        ap.error("At least two repeats are required to measure same-input numerical noise")
    args.output = args.output.resolve()
    if not args.output.is_relative_to((EXPERIMENT / "logs").resolve()):
        ap.error("Probe outputs must remain under the experiment logs directory")
    return args


async def prepare_samples(args, cfg):
    """Use the actual dataset, rollout template, MM feature and mRoPE methods."""
    import torch
    import pyarrow.parquet as pq
    from verl.utils import hf_processor
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, DictConfigWrap
    from experiments.sharedexpert_c.training.agent_loop import SharedExpertSingleTurn

    processor = hf_processor(cfg.actor_rollout_ref.model.path, local_files_only=True)
    if processor is None:
        raise RuntimeError("The actual verl processor could not be initialized")
    tokenizer = processor.tokenizer
    rows = pq.read_table(args.data).to_pylist()
    if args.rollouts is None:
        candidates = sorted((EXPERIMENT / "runs").glob("*/rollouts/1.jsonl"))
        if not candidates:
            raise FileNotFoundError("No real completed smoke rollout; pass --rollouts explicitly")
        args.rollouts = candidates[0]
    generations = [json.loads(line) for line in args.rollouts.read_text().splitlines() if line]
    samples, provenance = [], []
    # These two real examples have unequal lengths and distinct images/content.
    for source in ("vqav2", "docvqa"):
        row = next(row for row in rows if row["extra_info"]["source"] == source)
        prompt_text = row["prompt"][0]["content"].replace("<image>", "").strip()
        matches = [r for r in generations if r.get("audit_source") == "sharedexpert_" + source
                   and prompt_text in r["input"] and not r.get("truncated")]
        if not matches:
            raise ValueError(f"No matching non-truncated real rollout for {source}")
        generation = min(matches, key=lambda r: r["response_tokens"])
        response_ids = tokenizer.encode(generation["output"], add_special_tokens=False)
        eos_appended = not response_ids or response_ids[-1] != tokenizer.eos_token_id
        if eos_appended:
            response_ids.append(tokenizer.eos_token_id)
        class SavedResponseServer:
            async def generate(self, **kwargs):
                return SimpleNamespace(token_ids=response_ids, log_probs=None, routed_experts=None)
        loop = SharedExpertSingleTurn(
            trainer_config=DictConfigWrap(cfg), server_manager=SavedResponseServer(),
            tokenizer=tokenizer, processor=processor, dataset_cls=RLHFDataset,
            dataset_config=cfg.data,
        )
        dataset_stub = SimpleNamespace(prompt_key="prompt", image_key="images", video_key="videos", processor=processor)
        messages = RLHFDataset._build_messages(dataset_stub, copy.deepcopy(row))
        output = await loop.run({}, raw_prompt=messages)
        assert len(output.prompt_ids) == generation["prompt_tokens"], (source, len(output.prompt_ids), generation["prompt_tokens"])
        ids = torch.tensor([output.prompt_ids + response_ids], dtype=torch.long)
        helper = SimpleNamespace(processor=processor, tokenizer=tokenizer)
        multimodal = AgentLoopWorker._compute_multi_modal_inputs(helper, output, ids)
        samples.append({"prompt_ids": output.prompt_ids, "response_ids": response_ids, "multi_modal_inputs": multimodal})
        provenance.append({
            "source": source, "source_id": row["extra_info"]["source_id"],
            "record_id": row["extra_info"].get("record_id"),
            "image_paths": [image["path"] for image in row["images"]],
            "image_file_sha256": [sha(image["path"]) for image in row["images"]],
            "prompt_tokens": len(output.prompt_ids), "response_tokens": len(response_ids),
            "saved_rollout_response_tokens": generation["response_tokens"],
            "reconstructed_token_count_matches_saved": len(response_ids) == generation["response_tokens"],
            "eos_token_id": tokenizer.eos_token_id, "eos_appended": bool(eos_appended),
            "response_retokenized_from_real_saved_text": True,
            "response_text_sha256": hashlib.sha256(generation["output"].encode()).hexdigest(),
            "prompt_truncated": output.extra_fields["prompt_truncated"],
        })
    pwidth = max(len(s["prompt_ids"]) for s in samples)
    rwidth = max(len(s["response_ids"]) for s in samples)
    tensors = {key: [] for key in ("input_ids", "attention_mask", "position_ids", "responses", "response_mask")}
    mmis = []
    for sample in samples:
        pids, rids = sample["prompt_ids"], sample["response_ids"]
        left, right = pwidth - len(pids), rwidth - len(rids)
        response = rids + [tokenizer.pad_token_id] * right
        ids = torch.tensor([[tokenizer.pad_token_id] * left + pids + response], dtype=torch.long)
        attention = torch.tensor([[0] * left + [1] * (len(pids) + len(rids)) + [0] * right], dtype=torch.long)
        mmi = copy.deepcopy(sample["multi_modal_inputs"])
        position = AgentLoopWorker._compute_position_ids(SimpleNamespace(processor=processor), ids, attention, mmi)
        assert position[0, 0][attention[0].bool()].tolist() == list(range(len(pids) + len(rids)))
        values = {"input_ids": ids, "attention_mask": attention, "position_ids": position,
                  "responses": torch.tensor([response]), "response_mask": torch.tensor([[1] * len(rids) + [0] * right])}
        for key, value in values.items():
            tensors[key].append(value)
        mmis.append(mmi)
    batch = {key: torch.cat(values, dim=0) for key, values in tensors.items()}
    batch["multi_modal_inputs"] = mmis
    assert batch["position_ids"].shape == (2, 4, pwidth + rwidth)
    for index, sample in enumerate(samples):
        assert batch["input_ids"][index][batch["attention_mask"][index].bool()].tolist() == sample["prompt_ids"] + sample["response_ids"]
    # Preserve the entire second sequence and the first sequence's lengths,
    # image tokens, image features, masks and positions. Only non-EOS response
    # token order in the first sequence changes; EOS stays in place.
    mutated = {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in batch.items()}
    n = len(samples[0]["response_ids"]) - 1
    original = mutated["responses"][0, :n].clone()
    mutated["responses"][0, :n] = original.flip(0)
    mutated["input_ids"][0, pwidth:pwidth + n] = original.flip(0)
    assert not torch.equal(mutated["input_ids"][0], batch["input_ids"][0])
    assert torch.equal(mutated["input_ids"][1], batch["input_ids"][1])
    for key in ("attention_mask", "position_ids", "response_mask"):
        assert torch.equal(mutated[key], batch[key])
    mutated_ba = {key: value.clone() if torch.is_tensor(value) else copy.deepcopy(value) for key, value in batch.items()}
    n_second = len(samples[1]["response_ids"]) - 1
    second_original = mutated_ba["responses"][1, :n_second].clone()
    mutated_ba["responses"][1, :n_second] = second_original.flip(0)
    mutated_ba["input_ids"][1, pwidth:pwidth + n_second] = second_original.flip(0)
    assert torch.equal(mutated_ba["input_ids"][0], batch["input_ids"][0])
    for key in ("attention_mask", "position_ids", "response_mask"):
        assert torch.equal(mutated_ba[key], batch[key])
    report = {
        "sources": provenance, "config": str(args.config.resolve()), "config_sha256": sha(args.config),
        "data": str(args.data.resolve()), "data_sha256": sha(args.data),
        "rollouts": str(args.rollouts.resolve()), "rollouts_sha256": sha(args.rollouts),
        "probe_source_sha256": sha(__file__), "prompt_width": pwidth, "response_width": rwidth,
        "unpadded_tokens": int(batch["attention_mask"].sum()), "padded_tokens": int(batch["input_ids"].numel()),
        "valid_response_tokens": int(batch["response_mask"].sum()),
        "mutation": "AB reverses A non-EOS response tokens while fixing B; BA reverses B while fixing A. All lengths/layout/masks/images/positions preserved",
        "construction": "Actual RLHFDataset._build_messages, SharedExpertSingleTurn.run, AgentLoopWorker MM/mRoPE methods",
    }
    torch.save({"batch": batch, "mutated": mutated, "mutated_ba": mutated_ba, "provenance": report}, args.output / "inputs.pt")
    write_json(args.output / "inputs.json", report)
    return report


def cpu_prepare(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from omegaconf import OmegaConf
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cfg = OmegaConf.load(args.config)
    report = asyncio.run(prepare_samples(args, cfg))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def differences(a, b, mask):
    import torch
    a, b = a[mask.bool()].double(), b[mask.bool()].double()
    delta = a - b
    return {"n": a.numel(), "max_abs": float(delta.abs().max()),
            "mean_abs": float(delta.abs().mean()), "rms": float(delta.square().mean().sqrt()),
            "relative_l2": float(delta.norm() / b.norm().clamp_min(1e-30)),
            "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all())}


def execute(args):
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoModelForImageTextToText
    from torch.distributed.device_mesh import init_device_mesh
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict, get_shard_placement_fn, MixedPrecisionPolicy
    from verl.utils.model import update_model_config
    from verl.utils import hf_tokenizer
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.utils.ulysses import set_ulysses_sequence_parallel_group
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf_qwen

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != 4:
        raise ValueError("Use four ranks to match the actual teacher FSDP2 configuration")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    device = torch.device("cuda", local_rank)
    set_ulysses_sequence_parallel_group(None)
    cfg = OmegaConf.load(args.config)
    actor_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    assert actor_cfg.strategy == "fsdp2"
    assert not actor_cfg.fsdp_config.param_offload and not actor_cfg.fsdp_config.optimizer_offload
    assert not actor_cfg.fsdp_config.offload_policy
    assert cfg.actor_rollout_ref.model.enable_gradient_checkpointing
    assert not cfg.actor_rollout_ref.model.use_fused_kernels
    payload = torch.load(args.output / "inputs.pt", map_location="cpu", weights_only=False)
    if payload["provenance"]["config_sha256"] != sha(args.config):
        raise ValueError("Config changed after CPU preparation; regenerate probe inputs explicitly")
    def move(value):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list):
            return [move(v) for v in value]
        if isinstance(value, dict):
            return {k: move(v) for k, v in value.items()}
        return value
    batch, mutated, mutated_ba = move(payload["batch"]), move(payload["mutated"]), move(payload["mutated_ba"])
    del payload
    torch.manual_seed(20261019)
    torch.cuda.manual_seed_all(20261019)
    tokenizer = hf_tokenizer(cfg.actor_rollout_ref.model.path, local_files_only=True)
    model_cfg = AutoConfig.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True,
                                          attn_implementation="flash_attention_2")
    update_model_config(model_cfg, override_config_kwargs={"bos_token_id": tokenizer.bos_token_id,
                        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id})
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.actor_rollout_ref.model.path, config=model_cfg, torch_dtype=torch.float32,
        attn_implementation="flash_attention_2", device_map={"": device}, local_files_only=True,
    )
    apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1, use_fused_kernels=False,
                      use_prefix_grouper=False, use_tiled_mlp=False)
    model.to(torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    nonzero_dropout = [(name, module.p) for name, module in model.named_modules()
                       if isinstance(module, torch.nn.Dropout) and module.p != 0]
    if nonzero_dropout or getattr(model_cfg.text_config, "attention_dropout", 0):
        raise ValueError(f"Dropout would invalidate exact same-input comparisons: {nonzero_dropout}")
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))
    full_state = model.state_dict()
    apply_fsdp2(model, {
        "mesh": mesh, "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True),
        "offload_policy": None, "reshard_after_forward": actor_cfg.fsdp_config.reshard_after_forward,
        "shard_placement_fn": get_shard_placement_fn(world),
    }, actor_cfg.fsdp_config)
    fsdp2_load_full_state_dict(model, full_state, mesh, None)
    del full_state
    torch.cuda.empty_cache()
    model.train()
    actor = DataParallelPPOActor(actor_cfg, model, actor_optimizer=None)
    actor.use_fused_kernels = False
    actor.use_prefix_grouper = False
    modes = ("independent", "packed_ab", "packed_ba", "padded_ab", "padded_ba", "packed_mutated_ab", "padded_mutated_ab",
             "packed_mutated_ba", "padded_mutated_ba")
    response_mask_cpu = batch["response_mask"].cpu()
    original_decoder_forward = hf_qwen.Qwen3_5DecoderLayer.forward

    def segmented_decoder_forward(self, hidden_states, position_embeddings, attention_mask=None,
                                  position_ids=None, past_key_values=None, **kwargs):
        # All boundary information travels through the real forward arguments,
        # including gradient-checkpoint recomputation. No external stream state.
        if (self.layer_type != "linear_attention" or hidden_states.shape[0] != 1
                or attention_mask is not None or position_ids is None):
            return original_decoder_forward(self, hidden_states, position_embeddings,
                                            attention_mask, position_ids, past_key_values, **kwargs)
        if position_ids.ndim != 2:
            raise ValueError(f"Expected text positions [1,L], got {position_ids.shape}")
        starts = torch.where(position_ids[0] == 0)[0].tolist()
        if len(starts) <= 1:
            return original_decoder_forward(self, hidden_states, position_embeddings,
                                            attention_mask, position_ids, past_key_values, **kwargs)
        if starts[0] != 0 or past_key_values is not None:
            raise ValueError("Segmented control only supports reset-at-zero, uncached training streams")
        stops = starts[1:] + [hidden_states.shape[1]]
        # This intentionally segments the whole linear decoder block (including
        # its MLP). The outer FSDP/checkpoint wrapper still runs once. The full
        # attention decoder and its native varlen FlashAttention remain intact.
        if any(value.shape[-2] != hidden_states.shape[1] for value in position_embeddings):
            raise ValueError("Unexpected rotary sequence axis in segmented control")
        chunks = [original_decoder_forward(self, hidden_states[:, begin:end],
                                           tuple(value[..., begin:end, :] for value in position_embeddings),
                                           None, position_ids[:, begin:end], None, **kwargs)
                  for begin, end in zip(starts, stops)]
        return torch.cat(chunks, dim=1)

    def select(data, ids):
        return {key: [value[i] for i in ids] if key == "multi_modal_inputs" else value[ids]
                for key, value in data.items()}

    def forward(mode, backward=False):
        segmented = mode.startswith("segmented_")
        hf_qwen.Qwen3_5DecoderLayer.forward = segmented_decoder_forward if segmented else original_decoder_forward
        mode = mode.removeprefix("segmented_")
        data = (mutated_ba if mode.endswith("ba") else mutated) if "mutated" in mode else batch
        actor.use_remove_padding = not mode.startswith("padded")
        order = [1, 0] if mode.endswith("ba") else [0, 1]
        groups = [[0], [1]] if mode == "independent" else [order]
        result = [None, None]
        loss_sum = 0.0
        for ids in groups:
            micro = select(data, ids)
            result_dict = actor._forward_micro_batch(micro, temperature=1.0, calculate_entropy=False)
            logps = result_dict["log_probs"]
            loss = -(logps * micro["response_mask"]).sum() / batch["response_mask"].sum()
            if backward:
                loss.backward()
            loss_sum += float(loss.detach())
            for offset, idx in enumerate(ids):
                result[idx] = logps[offset].detach().float().cpu()
            del result_dict, logps, loss
        return torch.stack(result), loss_sum

    def timed(mode, backward=False):
        dist.barrier()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        if backward:
            prediction, loss = forward(mode, True)
        else:
            with torch.no_grad():
                prediction, loss = forward(mode, False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        metrics = torch.tensor([elapsed, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                               device=device, dtype=torch.float64)
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        return prediction, loss, {"max_rank_seconds": float(metrics[0]),
                "max_rank_peak_allocated_bytes": int(metrics[1]), "max_rank_peak_reserved_bytes": int(metrics[2])}

    source_files = [REPO / "verl/workers/actor/dp_actor.py", REPO / "verl/models/transformers/qwen3_5.py",
                    REPO / "verl/models/transformers/monkey_patch.py", REPO / "verl/utils/fsdp_utils.py"]
    source_files.append(Path(hf_qwen.__file__))
    import inspect
    linear_layers = [m for m in model.modules() if m.__class__.__name__.endswith("Qwen3_5GatedDeltaNet")]
    first_linear = linear_layers[0]
    def kernel_name(function):
        return None if function is None else f"{function.__module__}.{function.__qualname__}"
    report = {"status": "running", "world_size": world, "replicated_inputs_on_all_ranks": True,
              "executed_probe_source_sha256": sha(__file__),
              "linear_attention": {"layer_count": len(linear_layers), "forward_signature": str(inspect.signature(first_linear.forward)),
                  "causal_conv1d_fn": kernel_name(first_linear.causal_conv1d_fn),
                  "chunk_gated_delta_rule": kernel_name(first_linear.chunk_gated_delta_rule)},
              "unique_examples": 2, "configuration": "FSDP2 FP32 master / BF16 forward / FP32 reduction / FA2 / nonreentrant gradient checkpointing / no CPU offload",
              "actor_entry": "DataParallelPPOActor._forward_micro_batch", "source_sha256": {str(p): sha(p) for p in source_files},
              "input_provenance": json.loads((args.output / "inputs.json").read_text()),
              "forward": {}, "gradients": {},
              "limits": ["Short saved real multimodal responses, not full 8192-token optimizer training",
                         "NLL on identical response tokens tests forward/backward equivalence; no PPO optimizer update",
                         "Ranks replicate the same two samples for a controlled FSDP comparison; do not count rank duplicates as unique throughput",
                         "Timing includes the real actor path and small response-logprob CPU transfers; warmup is excluded",
                         "No arbitrary pass tolerance: compare same-input repeats, padded controls and fixed-layout causal intervention"]}
    report["limits"].append("Padded probe width is the local 781+250 tokens; production dynamic batching currently retains global padded width and uses approximate valid-token budgeting, so this does not validate production B2/dynamic memory safety")
    outputs = {}
    def measure_forward(mode):
        # Compile/warm every layout before collecting throughput and noise.
        timed(mode)
        repeats = [timed(mode) for _ in range(args.forward_repeats)]
        outputs[mode] = repeats[0][0]
        report["forward"][mode] = {
            "loss": repeats[0][1], "timings": [r[2] for r in repeats],
            "median_max_rank_seconds": statistics.median(r[2]["max_rank_seconds"] for r in repeats),
            "same_input_repeat_differences": [differences(r[0], repeats[0][0], response_mask_cpu) for r in repeats[1:]],
            "vs_independent": differences(repeats[0][0], outputs["independent"], response_mask_cpu),
            "per_example_vs_independent": [differences(repeats[0][0][i], outputs["independent"][i], response_mask_cpu[i]) for i in range(2)],
        }
        if rank == 0:
            write_json(args.output / "report.json", report)
            print(json.dumps({"completed_forward": mode, **report["forward"][mode]}, ensure_ascii=False), flush=True)
    for mode in modes:
        measure_forward(mode)
    report["causal_intervention"] = {
        "fixed_second_sequence_packed": differences(outputs["packed_mutated_ab"][1], outputs["packed_ab"][1], response_mask_cpu[1]),
        "fixed_second_sequence_padded_control": differences(outputs["padded_mutated_ab"][1], outputs["padded_ab"][1], response_mask_cpu[1]),
        "fixed_second_sequence_packed_ba": differences(outputs["packed_mutated_ba"][0], outputs["packed_ba"][0], response_mask_cpu[0]),
        "fixed_second_sequence_padded_control_ba": differences(outputs["padded_mutated_ba"][0], outputs["padded_ba"][0], response_mask_cpu[0]),
        "packed_order_exchange": differences(outputs["packed_ba"], outputs["packed_ab"], response_mask_cpu),
        "padded_order_exchange": differences(outputs["padded_ba"], outputs["padded_ab"], response_mask_cpu),
    }
    packed_noise = max(r["max_abs"] for mode in ("packed_ab", "packed_ba") for r in report["forward"][mode]["same_input_repeat_differences"])
    padded_causal = max(report["causal_intervention"][key]["max_abs"] for key in ("fixed_second_sequence_padded_control", "fixed_second_sequence_padded_control_ba"))
    observed = max(report["causal_intervention"][key]["max_abs"] for key in ("fixed_second_sequence_packed", "fixed_second_sequence_packed_ba"))
    causal_exceeds_controls = observed > max(packed_noise, padded_causal)
    report["segmented_linear_control"] = {"requested": args.segmented_linear_control,
        "native_fixed_second_change_exceeds_measured_controls": causal_exceeds_controls,
        "native_causal_max_abs": observed, "packed_repeat_noise_max_abs": packed_noise,
        "padded_causal_max_abs": padded_causal, "executed": False}
    segmented_modes = []
    if args.segmented_linear_control and causal_exceeds_controls:
        segmented_modes = ["segmented_packed_ab", "segmented_packed_ba"]
        for mode in segmented_modes + ["segmented_packed_mutated_ab", "segmented_packed_mutated_ba"]:
            measure_forward(mode)
        report["segmented_linear_control"].update({"executed": True,
            "scope": "Process-only Qwen3_5DecoderLayer.forward; entire linear-attention decoder segmented by text position resets; original full-attention packing untouched",
            "boundary_detection_cost_included": "Per-linear-layer GPU-to-CPU split-position extraction is intentionally included in measured runtime",
            "fixed_second_sequence": differences(outputs["segmented_packed_mutated_ab"][1], outputs["segmented_packed_ab"][1], response_mask_cpu[1]),
            "fixed_second_sequence_ba": differences(outputs["segmented_packed_mutated_ba"][0], outputs["segmented_packed_ba"][0], response_mask_cpu[0]),
            "order_exchange": differences(outputs["segmented_packed_ba"], outputs["segmented_packed_ab"], response_mask_cpu),
        })
    if rank == 0:
        torch.save({"response_mask": response_mask_cpu, "log_probs": outputs}, args.output / "response_log_probs.pt")
        write_json(args.output / "report.json", report)

    def local_gradient(parameter):
        grad = parameter.grad
        return grad.to_local() if hasattr(grad, "to_local") else grad

    def snapshot_gradients():
        return {name: local_gradient(parameter).detach().clone() for name, parameter in model.named_parameters() if parameter.grad is not None}

    def gradient_difference(reference):
        accum = torch.zeros(5, dtype=torch.float64, device=device)
        maximum = torch.zeros((), dtype=torch.float64, device=device)
        missing = []
        # Chunking avoids another full-model fp64 copy on GPU/CPU.
        for name, parameter in model.named_parameters():
            if parameter.grad is None and name not in reference:
                continue
            if parameter.grad is None or name not in reference:
                missing.append(name)
                continue
            actual, expected = local_gradient(parameter).detach().reshape(-1), reference[name].reshape(-1)
            for start in range(0, actual.numel(), 2**20):
                a, b = actual[start:start + 2**20].double(), expected[start:start + 2**20].double()
                delta = a - b
                accum += torch.stack((delta.square().sum(), b.square().sum(), a.square().sum(), (a*b).sum(),
                                      torch.tensor(a.numel(), dtype=torch.float64, device=device)))
                maximum = torch.maximum(maximum, delta.abs().max())
        dist.all_reduce(accum)
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
        return {"max_abs": float(maximum), "relative_l2": float((accum[0]/accum[1].clamp_min(1e-300)).sqrt()),
                "cosine": float(accum[3]/(accum[1]*accum[2]).sqrt().clamp_min(1e-300)),
                "reference_l2": float(accum[1].sqrt()), "actual_l2": float(accum[2].sqrt()),
                "gradient_elements_across_shards": int(accum[4]), "missing_parameter_gradients": missing,
                "finite": bool(torch.isfinite(accum).all() and torch.isfinite(maximum))}

    if not args.skip_gradients:
        reference = None
        for mode in ["independent", "packed_ab", "packed_ba", "padded_ab", "padded_ba"] + segmented_modes:
            model.zero_grad(set_to_none=True)
            # Also warm backward kernels, then clear them; no optimizer step.
            timed(mode, backward=True)
            model.zero_grad(set_to_none=True)
            prediction, loss, metrics = timed(mode, backward=True)
            if reference is None:
                reference = snapshot_gradients()
                report["reference_gradient_gpu_bytes_per_rank"] = sum(g.numel()*g.element_size() for g in reference.values())
            report["gradients"][mode] = {"loss": loss, "timing": metrics,
                    "vs_independent": gradient_difference(reference),
                    "forward_vs_independent": differences(prediction, outputs["independent"], response_mask_cpu)}
            mode_reference = reference if mode == "independent" else snapshot_gradients()
            model.zero_grad(set_to_none=True)
            repeat_prediction, repeat_loss, repeat_metrics = timed(mode, backward=True)
            report["gradients"][mode]["same_input_repeat"] = {
                "loss": repeat_loss, "timing": repeat_metrics,
                "gradient_noise_vs_same_mode_first": gradient_difference(mode_reference),
                "vs_independent": gradient_difference(reference),
                "forward_noise_vs_same_mode_first": differences(repeat_prediction, prediction, response_mask_cpu),
            }
            if mode != "independent":
                del mode_reference
            if rank == 0:
                write_json(args.output / "report.json", report)
                print(json.dumps({"completed_gradient": mode, **report["gradients"][mode]}, ensure_ascii=False), flush=True)
        del reference
    report["status"] = "complete" if not args.skip_gradients else "forward_only_complete"
    report["gradient_memory_note"] = "Initial independent run has no reference clone; independent repeat and each later mode's first run hold one baseline clone. Each later mode's repeat holds two clones. Compare matched-overhead runs only. All are sharded GPU clones, absent from forward-only timing."
    report["model_weights_unchanged"] = "No optimizer instantiated or parameter update performed"
    if rank == 0:
        write_json(args.output / "report.json", report)
        print("PACKING_PROBE_COMPLETE", flush=True)
    model.zero_grad(set_to_none=True)
    hf_qwen.Qwen3_5DecoderLayer.forward = original_decoder_forward
    dist.barrier()
    dist.destroy_process_group()


def main():
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        cpu_prepare(args)
    else:
        if not (args.output / "inputs.pt").exists():
            raise SystemExit("Run --prepare-only before allocating GPUs")
        execute(args)


if __name__ == "__main__":
    main()
