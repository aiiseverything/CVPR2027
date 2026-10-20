#!/usr/bin/env python3
"""Isolated FP32-versus-BF16 CPU pixel transport differential probe.

This script never edits the training implementation or starts Ray services.
First prepare with --prepare-only. Execute only when the root scheduler grants
the requested GPUs; two FSDP2 ranks use saved real images and responses, one
sequence at a time, with no optimizer and no production monkey patches.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

TRAINING = Path(__file__).resolve().parent
EXPERIMENT = TRAINING.parent
ROOT = EXPERIMENT.parents[1]
REPO = ROOT / "pre-lab/repo"
sys.path[:0] = [str(ROOT), str(REPO)]
PIXEL_KEYS = ("pixel_values", "pixel_values_videos")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def args_parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--source-inputs", type=Path, default=EXPERIMENT / "logs/packing_probe/inputs.pt")
    parser.add_argument("--config", type=Path, default=EXPERIMENT / "runs/perception_smoke/resolved_config.yaml")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "logs/pixel_dtype_probe")
    parser.add_argument("--gradient-repeats", type=int, default=3)
    parser.add_argument("--host-working-limit-gib", type=float, default=54.0)
    args = parser.parse_args()
    if args.prepare_only == args.execute:
        parser.error("Choose exactly one of --prepare-only or --execute")
    if args.gradient_repeats < 3:
        parser.error("At least three gradient runs per input dtype measure reference repeat noise")
    if not 1 < args.host_working_limit_gib < 60:
        parser.error("Host working-set guard must stay below the formal training guard")
    args.output = args.output.resolve()
    if not args.output.is_relative_to((EXPERIMENT / "logs").resolve()):
        parser.error("Independent probe outputs must remain under experiment/logs")
    return args


def host_sample():
    root = Path("/sys/fs/cgroup")
    stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
    current = int((root / "memory.current").read_text())
    return {"current_bytes": current, "working_set_bytes": current - int(stats.get("inactive_file", 0)),
            "anon_bytes": int(stats.get("anon", 0)),
            "events": {k: int(v) for k, v in (line.split() for line in (root / "memory.events").read_text().splitlines())}}


def start_guard(args, rank):
    """Abort only this probe rank; torchrun then cleans its sibling rank."""
    stop = threading.Event()
    state = {"scope": "shared container", "before": host_sample(), "peak_working_set_bytes": 0,
             "peak_current_bytes": 0, "guard_limit_gib": args.host_working_limit_gib}
    def watch():
        while not stop.is_set():
            sample = host_sample()
            state["peak_working_set_bytes"] = max(state["peak_working_set_bytes"], sample["working_set_bytes"])
            state["peak_current_bytes"] = max(state["peak_current_bytes"], sample["current_bytes"])
            if sample["working_set_bytes"] > args.host_working_limit_gib * 2**30:
                state.update(aborted=True, reason="Independent probe host guard", last=sample)
                write_json(args.output / f"resource_rank{rank}.json", state)
                os._exit(88)
            stop.wait(0.25)
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    return stop, thread, state


def prepare(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    import ray.cloudpickle as cloudpickle
    from verl import DataProto
    torch.set_num_threads(1)
    payload = torch.load(args.source_inputs, map_location="cpu", weights_only=False)
    provenance = payload["provenance"]
    if provenance["config_sha256"] != sha(args.config):
        raise ValueError("Saved real input config differs from the frozen smoke config")
    original = payload["batch"]
    del payload
    converted = {key: value for key, value in original.items() if key != "multi_modal_inputs"}
    converted["multi_modal_inputs"] = []
    pixel_report = []
    for index, mmi in enumerate(original["multi_modal_inputs"]):
        output = dict(mmi)
        for key in PIXEL_KEYS:
            if key not in output:
                continue
            pixel = output[key]
            assert pixel.dtype == torch.float32 and pixel.device.type == "cpu"
            assert not pixel.requires_grad and torch.isfinite(pixel).all()
            output[key] = pixel.to(torch.bfloat16)
            pixel_report.append({"sample": index, "key": key, "shape": list(pixel.shape),
                                 "fp32_bytes": pixel.numel() * pixel.element_size(),
                                 "bf16_bytes": output[key].numel() * output[key].element_size()})
        converted["multi_modal_inputs"].append(output)
    if not pixel_report:
        raise ValueError("Probe must contain real pixel_values")
    serialization = {}
    for mode, batch in (("fp32", original), ("bf16", converted)):
        mmi = np.empty(len(batch["multi_modal_inputs"]), dtype=object)
        mmi[:] = batch["multi_modal_inputs"]
        data = DataProto.from_dict(tensors={k: v for k, v in batch.items() if torch.is_tensor(v)},
                                   non_tensors={"multi_modal_inputs": mmi})
        encoded = cloudpickle.dumps(data, protocol=5)
        restored = cloudpickle.loads(encoded)
        checks = []
        for first, second in zip(mmi, restored.non_tensor_batch["multi_modal_inputs"]):
            assert first.keys() == second.keys()
            for key, tensor in first.items():
                assert torch.is_tensor(tensor) and torch.equal(tensor, second[key])
                assert tensor.dtype == second[key].dtype
                checks.append(key)
        for key, tensor in data.batch.items():
            assert torch.equal(tensor, restored.batch[key])
        serialization[mode] = {"serialized_bytes": len(encoded), "tensor_keys_checked": sorted(set(checks)),
                               "exact_roundtrip": True, "mechanism": "ray.cloudpickle protocol5; DataProto actual __getstate__/__setstate__; no Ray cluster"}
        del encoded, restored, data
    torch.save({"fp32": original, "bf16": converted, "provenance": provenance}, args.output / "inputs.pt")
    source_files = [Path(__file__), REPO / "verl/workers/actor/dp_actor.py", REPO / "verl/utils/fsdp_utils.py",
                    REPO / "verl/models/transformers/monkey_patch.py", REPO / "verl/models/transformers/qwen3_5.py"]
    report = {"status": "cpu_prepared", "config_sha256": sha(args.config), "source_inputs_sha256": sha(args.source_inputs),
              "prepared_inputs_sha256": sha(args.output / "inputs.pt"), "source_sha256": {str(p): sha(p) for p in source_files},
              "pixel_tensors": pixel_report, "serialization": serialization, "input_provenance": provenance,
              "limits": ["Transport roundtrip uses Ray cloudpickle, without a running Ray object store",
                         "Real saved VQA/DocVQA inputs, including all response tokens; no new generation",
                         "Only pixel_values and pixel_values_videos may change dtype; IDs, masks, grids unchanged"]}
    write_json(args.output / "cpu_report.json", report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


def execute(args):
    rank, local_rank, world = (int(os.environ[k]) for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    if world != 2:
        raise ValueError("This independent probe is sized for exactly two ranks")
    report = json.loads((args.output / "cpu_report.json").read_text())
    if report["config_sha256"] != sha(args.config) or report["prepared_inputs_sha256"] != sha(args.output / "inputs.pt"):
        raise ValueError("Prepared inputs/config drifted")
    for path, expected in report["source_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Probe or training source drifted: {path}")
    stop, guard_thread, resource = start_guard(args, rank)
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
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=20))
    set_ulysses_sequence_parallel_group(None)
    cfg = OmegaConf.load(args.config)
    actor_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    assert actor_cfg.strategy == "fsdp2" and actor_cfg.fsdp_config.dtype == "bfloat16"
    assert not actor_cfg.fsdp_config.offload_policy and not actor_cfg.fsdp_config.param_offload
    assert cfg.actor_rollout_ref.model.enable_gradient_checkpointing
    payload = torch.load(args.output / "inputs.pt", map_location="cpu", weights_only=False)
    batches = {mode: payload[mode] for mode in ("fp32", "bf16")}
    for mode in batches:
        # Match the actual actor: ordinary tensors move to GPU while MM inputs
        # remain on CPU until the FSDP root input transfer/cast.
        batches[mode] = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batches[mode].items()}
    conversion_checks = []
    for first, second in zip(batches["fp32"]["multi_modal_inputs"], batches["bf16"]["multi_modal_inputs"]):
        for key in PIXEL_KEYS:
            if key in first:
                gpu_first = first[key].to(device).to(torch.bfloat16)
                gpu_second = second[key].to(device)
                conversion_checks.append(bool(torch.equal(gpu_first, gpu_second)))
                del gpu_first, gpu_second
    assert conversion_checks and all(conversion_checks), "CPU/GPU BF16 conversion must be bitwise identical"
    del payload
    torch.manual_seed(20261019)
    torch.cuda.manual_seed_all(20261019)
    tokenizer = hf_tokenizer(cfg.actor_rollout_ref.model.path, local_files_only=True)
    model_cfg = AutoConfig.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True,
                                           attn_implementation="flash_attention_2")
    update_model_config(model_cfg, override_config_kwargs={"bos_token_id": tokenizer.bos_token_id,
                        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id})
    model = AutoModelForImageTextToText.from_pretrained(cfg.actor_rollout_ref.model.path, config=model_cfg,
        torch_dtype=torch.float32, attn_implementation="flash_attention_2", device_map={"": device}, local_files_only=True)
    apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1, use_fused_kernels=False,
                      use_prefix_grouper=False, use_tiled_mlp=False)
    model.to(torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert not [(n, m.p) for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.p]
    assert not getattr(model_cfg.text_config, "attention_dropout", 0)
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))
    full_state = model.state_dict()
    apply_fsdp2(model, {"mesh": mesh,
        "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=True),
        "offload_policy": None, "reshard_after_forward": actor_cfg.fsdp_config.reshard_after_forward,
        "shard_placement_fn": get_shard_placement_fn(world)}, actor_cfg.fsdp_config)
    fsdp2_load_full_state_dict(model, full_state, mesh, None)
    del full_state
    torch.cuda.empty_cache()
    model.train()
    actor = DataParallelPPOActor(actor_cfg, model, actor_optimizer=None)
    actor.use_remove_padding, actor.use_fused_kernels, actor.use_prefix_grouper = True, False, False
    mask = batches["fp32"]["response_mask"].bool().cpu()
    def one_pass(mode, backward):
        model.zero_grad(set_to_none=True)
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        batch, outputs, total_loss = batches[mode], [], 0.0
        for index in range(len(batch["multi_modal_inputs"])):
            micro = {k: [v[index]] if k == "multi_modal_inputs" else v[index:index+1] for k, v in batch.items()}
            with torch.set_grad_enabled(backward):
                result = actor._forward_micro_batch(micro, temperature=1.0, calculate_entropy=False)
                logps = result["log_probs"]
                loss = -(logps * micro["response_mask"]).sum() / batch["response_mask"].sum()
                if backward:
                    loss.backward()
            outputs.append(logps.detach().float().cpu())
            total_loss += float(loss.detach())
            del result, logps, loss
        torch.cuda.synchronize()
        metrics = torch.tensor([time.perf_counter()-start, torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                               device=device, dtype=torch.float64)
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        return torch.cat(outputs), {"loss": total_loss, "max_rank_seconds": float(metrics[0]),
            "max_rank_peak_allocated_bytes": int(metrics[1]), "max_rank_peak_reserved_bytes": int(metrics[2])}
    def logp_difference(actual, expected):
        a, b = actual[mask].double(), expected[mask].double()
        delta = a-b
        return {"max_abs": float(delta.abs().max()), "rms": float(delta.square().mean().sqrt()),
                "finite": bool(torch.isfinite(a).all() and torch.isfinite(b).all()), "tokens": a.numel()}
    def local_grad(parameter):
        return parameter.grad.to_local() if hasattr(parameter.grad, "to_local") else parameter.grad
    def gradient_difference(reference):
        totals = torch.zeros((2, 3), dtype=torch.float64, device=device)
        maxima = torch.zeros(2, dtype=torch.float64, device=device)
        missing = []
        for name, parameter in model.named_parameters():
            if parameter.grad is None and name not in reference:
                continue
            if parameter.grad is None or name not in reference:
                missing.append(name)
                continue
            actual, expected = local_grad(parameter).detach().reshape(-1), reference[name].reshape(-1)
            groups = (0, 1) if ".visual." in name else (0,)
            for offset in range(0, actual.numel(), 2**18):
                a, b = actual[offset:offset+2**18].double(), expected[offset:offset+2**18].double()
                delta = a-b
                values = torch.stack((delta.square().sum(), b.square().sum(), torch.tensor(a.numel(), device=device)))
                for group in groups:
                    totals[group] += values
                    maxima[group] = torch.maximum(maxima[group], delta.abs().max())
        dist.all_reduce(totals)
        dist.all_reduce(maxima, op=dist.ReduceOp.MAX)
        result = {}
        for index, name in enumerate(("full", "vision")):
            result[name] = {"max_abs": float(maxima[index]),
                "relative_l2": float((totals[index,0]/totals[index,1].clamp_min(1e-300)).sqrt()),
                "reference_l2": float(totals[index,1].sqrt()), "elements": int(totals[index,2]),
                "finite": bool(torch.isfinite(totals[index]).all() and torch.isfinite(maxima[index]))}
        result["missing_gradients"] = missing
        return result
    report.update(status="running", world_size=world, gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_gpu_bf16_bitwise_equal=conversion_checks, single_sequence_per_forward=True,
        configuration="FSDP2 FP32 master / BF16 forward / FP32 reduction / no offload / no optimizer",
        gradients={}, forward={}, reference_storage="One sharded full-gradient clone on GPU; never full gradients on CPU")
    one_pass("fp32", False)
    forward_reference = None
    for mode in ("fp32", "bf16"):
        report["forward"][mode] = []
        for repeat in range(3):
            prediction, metrics = one_pass(mode, False)
            if forward_reference is None:
                forward_reference = prediction
            report["forward"][mode].append({**metrics, "vs_reference": logp_difference(prediction, forward_reference)})
    one_pass("fp32", True)  # backward warmup; clear before the reference
    reference = None
    for mode in ("fp32", "bf16"):
        report["gradients"][mode] = []
        for repeat in range(args.gradient_repeats):
            prediction, metrics = one_pass(mode, True)
            if reference is None:
                reference = {name: local_grad(p).detach().clone() for name, p in model.named_parameters() if p.grad is not None}
                report["reference_gradient_gpu_bytes_per_rank"] = sum(g.numel()*g.element_size() for g in reference.values())
            report["gradients"][mode].append({**metrics, "vs_reference": gradient_difference(reference),
                "forward_vs_reference": logp_difference(prediction, forward_reference)})
            if rank == 0:
                write_json(args.output / "report.json", report)
                print(json.dumps({"completed_gradient": mode, "repeat": repeat,
                                  "metrics": report["gradients"][mode][-1]}), flush=True)
    del reference
    model.zero_grad(set_to_none=True)
    report["noise_comparison"] = {}
    for group in ("full", "vision"):
        metrics = {}
        for metric in ("max_abs", "relative_l2"):
            noise = max(x["vs_reference"][group][metric] for x in report["gradients"]["fp32"][1:])
            observed = max(x["vs_reference"][group][metric] for x in report["gradients"]["bf16"])
            metrics[metric] = {"reference_repeat_noise_max": noise, "bf16_difference_max": observed,
                               "within_observed_noise": observed <= noise}
        report["noise_comparison"][group] = metrics
    report["status"] = "complete"
    report["limits"] += ["Two FSDP ranks rather than production four; single-sequence real short inputs",
        "Full and vision parameter gradients of identical NLL; does not test PPO optimizer or 8192-token peak",
        "Gradient tolerance is reported against measured same-input noise, not a fabricated fixed threshold",
        "All ranks replicate the same two inputs to isolate transport precision; no throughput claim"]
    dist.barrier()
    stop.set()
    guard_thread.join(timeout=2)
    resource.update(after=host_sample(), aborted=False)
    write_json(args.output / f"resource_rank{rank}.json", resource)
    if rank == 0:
        write_json(args.output / "report.json", report)
        print("PIXEL_DTYPE_PROBE_COMPLETE", flush=True)
    dist.destroy_process_group()


def main():
    args = args_parse()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        prepare(args)
    else:
        execute(args)


if __name__ == "__main__":
    main()
