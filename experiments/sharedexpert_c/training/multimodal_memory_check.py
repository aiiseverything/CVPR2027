#!/usr/bin/env python3
"""Independent R/G warm-Adam memory check; CPU preparation never starts GPUs.

Run --prepare once, --check-cpu to verify inputs, --supervise only on idle GPUs
3-6, then --verify. A valid completed receipt makes --supervise a CPU-only no-op.
Only this script's dedicated logs directory is written. No training is resumed,
no rollout service is started, and pressure inputs never enter training data.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import ctypes
from datetime import datetime, timezone
import fcntl
import functools
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "0"
TRAINING = Path(__file__).resolve().parent
EXPERIMENT = TRAINING.parent
ROOT = EXPERIMENT.parents[1]
REPO = ROOT / "pre-lab/repo"
OUTPUT_ROOT = EXPERIMENT / "logs/multimodal_memory_check"
CANONICAL = EXPERIMENT / "logs/unused_topk_probe"
sys.path[:0] = [str(ROOT), str(REPO)]
# Reuse the already validated low-memory hashing, atomic JSON and cgroup reads.
from experiments.sharedexpert_c.training.unused_topk_probe import sha, save, host

CASES = (
    ("eight_images", "virl39k:Processed-b497da3d-cb40-443b-a9d6-1fea88faaf05", 8),
    ("three_images_max_visual", "virl39k:Processed-4c784a40-cd45-4968-a9de-a5bfe696888c", 3),
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ("prepare", "check-cpu", "supervise", "verify", "execute", "self-test-cleanup", "self-test-receipt", "dummy-tree"):
        modes.add_argument("--" + mode, action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--gpus", default="3,4,5,6")
    parser.add_argument("--host-limit-gib", type=float, default=54)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(OUTPUT_ROOT.resolve()):
        parser.error("Artifacts must stay in the dedicated multimodal_memory_check directory")
    if args.gpus != "3,4,5,6":
        parser.error("Only the experiment's physical GPUs 3,4,5,6 are supported")
    if not 0 < args.host_limit_gib <= 54:
        parser.error("The independent host guard may not exceed 54 GiB")
    if args.attempt:
        args.attempt = args.attempt.resolve()
        if not args.attempt.is_relative_to(args.output / "attempts"):
            parser.error("Attempt directory must be inside output/attempts")
    if args.execute and not args.attempt:
        parser.error("--execute is an internal torchrun mode requiring --attempt")
    return args


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    try:
        root = Path("/proc") / str(pid)
        stat = root.joinpath("stat").read_text().rsplit(")", 1)[1].split()
        return {"pid": pid, "start_ticks": int(stat[19]), "pgrp": int(stat[2]),
                "session": int(stat[3]), "state": stat[0]}
    except FileNotFoundError:
        return None


def parent_death_signal(expected_parent):
    """Called only in the freshly forked child, before exec; closes the fork race."""
    if ctypes.CDLL(None, use_errno=True).prctl(1, signal.SIGTERM, 0, 0, 0) != 0:
        os._exit(127)
    if os.getppid() != expected_parent:
        os._exit(126)


def spawn_owned(command, **kwargs):
    parent = os.getpid()
    return subprocess.Popen(command, start_new_session=True,
                            preexec_fn=functools.partial(parent_death_signal, parent), **kwargs)


def stop_owned(child, timeout=10):
    """Signal only the new session created by spawn_owned, then reap its leader."""
    if child is None:
        return
    if child.poll() is None:
        identity = process_identity(child.pid)
        if identity and identity["pgrp"] != child.pid:
            raise RuntimeError("Owned child no longer leads the expected process group")
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)


def model_weight_identity(model):
    # Avoid re-reading 8.8 GiB of weights during the live perception training.
    # These are explicitly metadata identities, not content hashes.
    result = {}
    for path in sorted(model.glob("*.safetensors")):
        stat = path.stat()
        result[str(path.resolve())] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino}
    if not result:
        raise RuntimeError("No local model safetensors found")
    return result


def verify_canonical():
    path = EXPERIMENT / "logs/unused_topk_transition_validation.json"
    receipt = json.loads(path.read_text())
    if receipt.get("passed") is not True:
        raise RuntimeError("Canonical unused-top-k validation has not passed")
    for relative, expected in receipt["validated_artifacts"].items():
        target = (EXPERIMENT / relative).resolve()
        if not target.is_relative_to(EXPERIMENT) or sha(target) != expected:
            raise RuntimeError(f"Canonical artifact changed: {relative}")
    report = json.loads((CANONICAL / "report.json").read_text())
    supervisor = json.loads((CANONICAL / "supervisor.json").read_text())
    if not (report.get("passed") is True and supervisor.get("status") == "complete"
            and supervisor.get("returncode") == 0 and not supervisor.get("guard_triggered")
            and all(supervisor["memory_events_delta"][key] == 0 for key in ("oom", "oom_kill"))):
        raise RuntimeError("Canonical four-rank validation did not exit successfully")
    return report


def verify_cpu(output):
    report = json.loads((output / "cpu_report.json").read_text())
    if report.get("status") != "cpu_prepared" or report.get("gpu_started") is not False:
        raise RuntimeError("CPU preparation is incomplete")
    for path, expected in report["source_sha256"].items():
        if sha(path) != expected:
            raise RuntimeError(f"Prepared source/data identity changed: {path}")
    if sha(output / "inputs.pt") != report["inputs_sha256"]:
        raise RuntimeError("Prepared multimodal inputs changed")
    if sha(output / "resolved_config.yaml") != report["resolved_config_sha256"]:
        raise RuntimeError("Prepared effective config changed")
    if model_weight_identity(Path(report["model_path"])) != report["model_weight_metadata"]:
        raise RuntimeError("Model weight metadata changed after preparation")
    return report


def check_rank_report(result, cpu_sha, rank):
    def require(condition, message):
        if not condition:
            raise RuntimeError(f"Rank {rank}: {message}")
    require(result.get("status") == "complete" and result.get("passed") is True
            and result.get("rank") == rank and result.get("world_size") == 4
            and result.get("cpu_report_sha256") == cpu_sha, "incomplete result or mismatched identity")
    updates = result.get("updates", [])
    require(len(updates) == 2, "two optimizer updates are required")
    for step, update in enumerate(updates, 1):
        require(update.get("update") == step and update.get("learning_rate") == (0.0 if step == 1 else 1e-7), "unexpected update or learning rate")
        metrics = update.get("metrics", {})
        require({"actor/grad_norm", "actor/pg_loss", "actor/kl_loss"}.issubset(metrics)
                and all(isinstance(value, (int, float)) and math.isfinite(value) for value in metrics.values())
                and metrics["actor/grad_norm"] > 0, "non-finite or missing PPO/gradient metrics")
        calls = update.get("forward_calls", [])
        require(len(calls) == 2 and [call.get("case") for call in calls] == [case[0] for case in CASES], "two ordered single-sequence forwards are required")
        require(all(call.get("full_logps") is False and call.get("topk") is None and call.get("entropy") is False for call in calls), "unused outputs or entropy enabled")
        accumulated = calls[1].get("gradient_before", {})
        require(accumulated.get("parameters", 0) > 0 and accumulated.get("bytes", 0) > 0, "second microbatch lacks accumulated gradients")
        require(calls[0].get("gradient_before") == {"parameters": 0, "bytes": 0}, "update did not start with cleared gradients")
        after = update.get("adam_after", {})
        require(after.get("all_fp32_cuda") is True and after.get("parameters") == accumulated["parameters"]
                and after.get("bytes") == 2 * accumulated["bytes"]
                and after.get("step_min") == after.get("step_max") == step, "Adam states do not cover the FP32 gradients")
        states_before = [update.get("adam_before", {}), *[call.get("adam_before", {}) for call in calls]]
        if step == 1:
            require(all(state.get("parameters") == 0 and state.get("bytes") == 0 for state in states_before), "first update did not begin with empty Adam states")
        else:
            require(all(state.get("parameters") == after["parameters"] and state.get("bytes") == after["bytes"]
                        and state.get("all_fp32_cuda") is True and state.get("step_min") == state.get("step_max") == 1
                        for state in states_before), "warm Adam and gradients were not concurrent on update 2")
    kernels = result.get("flash_attention_calls", {})
    require(kernels and all("deterministicFalse" in key and isinstance(value, int) and value > 0 for key, value in kernels.items())
            and any("head64" in key for key in kernels) and any("head256" in key for key in kernels), "production FA2 kernel evidence missing")


def verify_result_records(output, receipt, cpu_sha):
    """Check immutable result files and the actual four-rank pressure evidence."""
    if receipt.get("passed") is not True or receipt.get("cpu_report_sha256") != cpu_sha:
        raise RuntimeError("No matching successful multimodal memory receipt")
    report_relative = Path(receipt["report"])
    if len(report_relative.parts) != 3 or report_relative.parts[0] != "attempts" or report_relative.name != "report.json":
        raise RuntimeError("Report must belong to one local attempt")
    attempt = report_relative.parent
    if receipt["supervisor"] != str(attempt / "supervisor.json"):
        raise RuntimeError("Report and supervisor must belong to the same attempt")
    names = ("report.json", "supervisor.json", "gpu.log", "resources.jsonl", "rank0.json", "rank1.json", "rank2.json", "rank3.json")
    expected_artifacts = {str(attempt / name) for name in names}
    if set(receipt.get("validated_artifacts", {})) != expected_artifacts:
        raise RuntimeError("Result manifest must contain exactly report, supervisor, four ranks, log and resource samples")
    for relative, expected in receipt["validated_artifacts"].items():
        target = (output / relative).resolve()
        if not target.is_relative_to(output) or sha(target) != expected:
            raise RuntimeError(f"Memory-check result changed: {relative}")
    result = json.loads((output / receipt["report"]).read_text())
    supervisor = json.loads((output / receipt["supervisor"]).read_text())
    check_rank_report(result, cpu_sha, 0)
    all_ranks = result.get("all_ranks", [])
    if not (len(all_ranks) == 4 and all(type(item.get("rank")) is int and item.get("passed") is True for item in all_ranks)
            and {item["rank"] for item in all_ranks} == {0, 1, 2, 3}):
        raise RuntimeError("Report does not confirm all four distinct ranks")
    for rank in range(4):
        rank_result = json.loads((output / attempt / f"rank{rank}.json").read_text())
        check_rank_report(rank_result, cpu_sha, rank)
        entry = next(item for item in all_ranks if item["rank"] == rank)
        if entry.get("gradient_norm") != rank_result["updates"][1]["metrics"]["actor/grad_norm"]:
            raise RuntimeError(f"Rank {rank} reduction receipt does not match its report")
    if not (supervisor.get("returncode") == 0 and supervisor.get("cpu_report_sha256") == cpu_sha
            and supervisor.get("status") == "complete" and supervisor.get("gpus_released") is True
            and not supervisor.get("guard_triggered") and not supervisor.get("interrupted_signal")
            and all(supervisor["memory_events_delta"][key] == 0 for key in ("oom", "oom_kill"))):
        raise RuntimeError("The actual four-rank memory check is incomplete or failed")
    return result


def verify_receipt(output):
    prepared = verify_cpu(output)
    receipt = json.loads((output / "receipt.json").read_text())
    result = verify_result_records(output, receipt, sha(output / "cpu_report.json"))
    return {"passed": True, "receipt": str(output / "receipt.json"),
            "receipt_sha256": sha(output / "receipt.json"), "coverage": prepared["cases"],
            "limitations": result["limitations"]}


async def prepare_inputs(cfg, rows, response_ids):
    import torch
    from verl.utils import hf_processor
    from verl.utils.dataset.rl_dataset import RLHFDataset
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker, DictConfigWrap
    from experiments.sharedexpert_c.training.agent_loop import SharedExpertSingleTurn
    processor = hf_processor(cfg.actor_rollout_ref.model.path, local_files_only=True)
    tokenizer = processor.tokenizer
    image_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    class SavedResponse:
        async def generate(self, **kwargs):
            return SimpleNamespace(token_ids=response_ids, log_probs=None, routed_experts=None)
    loop = SharedExpertSingleTurn(trainer_config=DictConfigWrap(cfg), server_manager=SavedResponse(),
        tokenizer=tokenizer, processor=processor, dataset_cls=RLHFDataset, dataset_config=cfg.data)
    stub = SimpleNamespace(prompt_key="prompt", image_key="images", video_key="videos", processor=processor)
    helper = SimpleNamespace(processor=processor, tokenizer=tokenizer)
    actual_samples, stress_samples, provenance = [], [], []
    for label, record_id, image_count in CASES:
        row = rows[record_id]["row"]
        messages = RLHFDataset._build_messages(stub, copy.deepcopy(row))
        actual = await loop.run({}, raw_prompt=messages)
        actual_ids = torch.tensor([actual.prompt_ids + response_ids])
        mmi = AgentLoopWorker._compute_multi_modal_inputs(helper, actual, actual_ids)
        visual_tokens = actual.prompt_ids.count(image_id)
        assert visual_tokens == int(mmi["image_grid_thw"].prod(dim=-1).sum()) // 4
        assert len(mmi["image_grid_thw"]) == image_count
        assert not actual.extra_fields["prompt_truncated"]
        def pressure_messages(count):
            value = copy.deepcopy(messages)
            content = next(m["content"] for m in value if m["role"] == "user")
            assert isinstance(content, list)
            content.append({"type": "text", "text": "\nAdditional memory-test text:" + " detail" * count})
            return value
        def expected_length(value):
            text = processor.apply_chat_template(value, tools=loop.tool_schemas,
                add_generation_prompt=True, tokenize=False, enable_thinking=True)
            ids = tokenizer.encode(text, add_special_tokens=False)
            assert ids.count(image_id) == image_count
            return len(ids) + visual_tokens - image_count
        assert expected_length(messages) == len(actual.prompt_ids)
        low, high = 0, 4096
        while low < high:
            middle = (low + high + 1) // 2
            if expected_length(pressure_messages(middle)) <= 2048:
                low = middle
            else:
                high = middle - 1
        pressure = pressure_messages(low)
        # Original message blocks and image order remain exactly intact.
        for original, changed in zip(messages, pressure, strict=True):
            assert changed["role"] == original["role"]
            expected = changed["content"][:-1] if original["role"] == "user" else changed["content"]
            assert expected == original["content"]
        stressed = await loop.run({}, raw_prompt=pressure)
        assert 2040 <= len(stressed.prompt_ids) <= 2048
        assert not stressed.extra_fields["prompt_truncated"]
        assert len(stressed.prompt_ids) == expected_length(pressure)
        assert stressed.prompt_ids.count(image_id) == visual_tokens
        stress_ids = torch.tensor([stressed.prompt_ids + response_ids])
        stress_mmi = AgentLoopWorker._compute_multi_modal_inputs(helper, stressed, stress_ids)
        assert mmi.keys() == stress_mmi.keys()
        # TF5 also returns text-length-dependent mm_token_type_ids. The actual
        # helper reconstructs those from each padded input during mRoPE creation.
        image_keys = set(mmi) - {"mm_token_type_ids"}
        assert all(torch.equal(mmi[key], stress_mmi[key]) for key in image_keys)
        # Reuse identical image tensors, retaining separate text metadata.
        for key in image_keys:
            stress_mmi[key] = mmi[key]
        actual_samples.append({"prompt_ids": actual.prompt_ids, "multi_modal_inputs": mmi})
        stress_samples.append({"prompt_ids": stressed.prompt_ids, "multi_modal_inputs": stress_mmi})
        provenance.append({"case": label, "record_id": record_id,
            "prepared_locations": rows[record_id]["locations"], "images": image_count,
            "image_file_sha256": {item["path"]: sha(item["path"]) for item in row["images"]},
            "actual_prompt_tokens": len(actual.prompt_ids), "stress_prompt_tokens": len(stressed.prompt_ids),
            "response_tokens": len(response_ids), "visual_tokens": visual_tokens,
            "image_grid_thw": mmi["image_grid_thw"].tolist(),
            "pixel_values_shape": list(mmi["pixel_values"].shape),
            "pixel_values_dtype": str(mmi["pixel_values"].dtype),
            "pixel_values_bytes": mmi["pixel_values"].numel() * mmi["pixel_values"].element_size(),
            "added_ordinary_text_repetitions": low, "original_blocks_preserved": True,
            "actual_and_pressure_image_tensors_equal": True, "prompt_truncated": False})
    def build(samples):
        width = max(len(sample["prompt_ids"]) for sample in samples)
        tensors = {key: [] for key in ("input_ids", "attention_mask", "position_ids", "responses", "response_mask")}
        mmis = []
        for sample in samples:
            left = width - len(sample["prompt_ids"])
            ids = torch.tensor([[tokenizer.pad_token_id] * left + sample["prompt_ids"] + response_ids])
            mask = torch.tensor([[0] * left + [1] * (len(sample["prompt_ids"]) + len(response_ids))])
            mmi = dict(sample["multi_modal_inputs"])
            pos = AgentLoopWorker._compute_position_ids(helper, ids, mask, mmi)
            mmis.append(mmi)
            assert pos.shape == (1, 4, width + len(response_ids))
            assert pos[0, 0][mask[0].bool()].tolist() == list(range(len(sample["prompt_ids"]) + len(response_ids)))
            assert ids[0][mask[0].bool()].tolist() == sample["prompt_ids"] + response_ids
            values = dict(input_ids=ids, attention_mask=mask, position_ids=pos,
                          responses=torch.tensor([response_ids]), response_mask=torch.ones((1, len(response_ids)), dtype=torch.long))
            for key, value in values.items():
                tensors[key].append(value)
        batch = {key: torch.cat(values) for key, values in tensors.items()}
        batch["multi_modal_inputs"] = mmis
        return batch
    return {"actual_batch": build(actual_samples), "stress_batch": build(stress_samples)}, provenance


def prepare(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    import pyarrow.parquet as pq
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from experiments.sharedexpert_c.training.launch import make_overrides
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if (args.output / "receipt.json").exists():
        raise RuntimeError("A completed receipt exists; use a new dedicated output subdirectory")
    args.output.mkdir(parents=True, exist_ok=True)
    canonical = verify_canonical()
    protocol = json.loads((TRAINING / "config/common.json").read_text())
    configs = {}
    for role in ("reasoning", "generalist"):
        teacher = json.loads((TRAINING / f"config/{role}.json").read_text())
        with initialize_config_dir(config_dir=str(REPO / "verl/trainer/config"), version_base=None):
            cfg = compose(config_name="ppo_trainer", overrides=make_overrides(protocol, teacher, 4,
                EXPERIMENT / "runs" / role, protocol["steps"], protocol["batch_size"]))
        old = OmegaConf.load(CANONICAL / f"resolved_after_{role}.yaml")
        assert OmegaConf.to_container(cfg, resolve=True) == OmegaConf.to_container(old, resolve=True)
        configs[role] = cfg
    assert OmegaConf.to_container(configs["reasoning"].actor_rollout_ref, resolve=True) == OmegaConf.to_container(configs["generalist"].actor_rollout_ref, resolve=True)
    cfg = configs["reasoning"]
    (args.output / "resolved_config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    wanted = {case[1] for case in CASES}
    rows = {}
    for role in ("reasoning", "generalist"):
        index = 0
        parquet = EXPERIMENT / f"data_pipeline/prepared/{role}.parquet"
        for batch in pq.ParquetFile(parquet).iter_batches(batch_size=64, columns=["prompt", "images", "extra_info"], use_threads=False):
            for row in batch.to_pylist():
                index += 1
                if index > 9000:
                    break
                record_id = row["extra_info"]["record_id"]
                if record_id in wanted:
                    entry = rows.setdefault(record_id, {"row": row, "locations": []})
                    assert entry["row"]["prompt"] == row["prompt"] and entry["row"]["images"] == row["images"]
                    entry["locations"].append({"teacher": role, "row_1based": index, "step_1based": (index - 1) // 30 + 1})
            if index >= 9000:
                break
    assert set(rows) == wanted
    assert {"teacher": "reasoning", "row_1based": 2467, "step_1based": 83} in rows[CASES[1][1]]["locations"]
    assert {x["teacher"] for x in rows[CASES[0][1]]["locations"]} == {"reasoning", "generalist"}
    original = torch.load(CANONICAL / "inputs.pt", map_location="cpu", weights_only=False)
    response_ids = original["long"]["responses"][0].tolist()
    assert len(response_ids) == 8192
    del original
    gc.collect()
    inputs, cases = asyncio.run(prepare_inputs(cfg, rows, response_ids))
    torch.save(inputs, args.output / "inputs.pt")
    files = {Path(p) for p in canonical["source_sha256"]}
    files.update({Path(__file__), TRAINING / "config/common.json", TRAINING / "config/reasoning.json",
        TRAINING / "config/generalist.json", EXPERIMENT / "logs/unused_topk_transition_validation.json",
        CANONICAL / "report.json", CANONICAL / "supervisor.json", CANONICAL / "inputs.pt",
        CANONICAL / "resolved_after_reasoning.yaml", CANONICAL / "resolved_after_generalist.yaml",
        EXPERIMENT / "logs/visual_batch30_budget_audit.json"})
    model_path = Path(cfg.actor_rollout_ref.model.path).resolve()
    files.update(model_path / name for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "preprocessor_config.json"))
    files.update(REPO / name for name in ("verl/trainer/ppo/core_algos.py", "verl/models/transformers/monkey_patch.py",
        "verl/models/transformers/qwen3_5.py", "verl/experimental/agent_loop/agent_loop.py"))
    files.update(EXPERIMENT / f"data_pipeline/prepared/{role}.parquet" for role in ("reasoning", "generalist"))
    files.update(Path(path) for case in cases for path in case["image_file_sha256"])
    assert not torch.cuda.is_initialized()
    report = {"status": "cpu_prepared", "created_at": timestamp(), "gpu_started": False,
        "torch_cuda_initialized": torch.cuda.is_initialized(),
        "shared_teachers": ["reasoning", "generalist"], "cases": cases,
        "inputs_sha256": sha(args.output / "inputs.pt"), "resolved_config_sha256": sha(args.output / "resolved_config.yaml"),
        "source_sha256": {str(path.resolve()): sha(path) for path in sorted(files)},
        "model_path": str(model_path), "model_weight_metadata": model_weight_identity(model_path),
        "model_weight_identity_scope": "File size, mtime_ns and inode; weight bytes not rehashed during live training",
        "response_provenance": canonical["long_provenance"],
        "pressure_scope": "Synthetic ordinary-text additions and transplanted saved 8192 response; not correctness/reward or new-rollout evidence",
        "gpu_plan": {"gpus": [3, 4, 5, 6], "updates": 2, "mini_per_rank": 2, "micro_per_rank": 1,
                     "sample_order": [case[0] for case in CASES], "flash_attention_deterministic": False,
                     "first_learning_rate": 0.0, "second_learning_rate": 1e-7}}
    save(args.output / "cpu_report.json", report)
    verify_cpu(args.output)
    print(json.dumps({"status": report["status"], "cases": cases, "output": str(args.output)}, indent=2), flush=True)


def execute(args):
    import numpy as np
    import torch
    import torch.distributed as dist
    from datetime import timedelta
    from omegaconf import OmegaConf
    from transformers import AutoConfig, AutoModelForImageTextToText
    from torch.distributed.device_mesh import init_device_mesh
    import flash_attn.flash_attn_interface as fa
    from verl import DataProto
    from verl.models.transformers.monkey_patch import apply_monkey_patch
    from verl.utils import hf_tokenizer
    from verl.utils.config import omega_conf_to_dataclass
    from verl.utils.fsdp_utils import apply_fsdp2, fsdp2_load_full_state_dict, get_shard_placement_fn, MixedPrecisionPolicy
    from verl.utils.model import update_model_config
    from verl.utils.ulysses import set_ulysses_sequence_parallel_group
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.config.optimizer import build_optimizer
    rank, local_rank, world = [int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE")]
    assert world == 4 and os.environ["FLASH_ATTENTION_DETERMINISTIC"] == "0"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30))
    set_ulysses_sequence_parallel_group(None)
    prepared = verify_cpu(args.output)
    cfg = OmegaConf.load(args.output / "resolved_config.yaml")
    cfg.actor_rollout_ref.actor.ppo_mini_batch_size = 2  # Probe-only normalization, two micro=1 calls per rank.
    actor_cfg = omega_conf_to_dataclass(cfg.actor_rollout_ref.actor)
    assert actor_cfg.strategy == "fsdp2" and actor_cfg.ppo_micro_batch_size_per_gpu == 1
    assert not actor_cfg.use_dynamic_bsz and not actor_cfg.use_kl_loss and not actor_cfg.calculate_entropy
    assert actor_cfg.entropy_coeff == 0 and actor_cfg.policy_loss.loss_mode == "vanilla"
    assert not actor_cfg.self_distillation.full_logit_distillation and actor_cfg.self_distillation.distillation_topk is None
    assert not actor_cfg.fsdp_config.param_offload and not actor_cfg.fsdp_config.optimizer_offload and not actor_cfg.fsdp_config.offload_policy
    assert cfg.actor_rollout_ref.model.enable_gradient_checkpointing and not cfg.actor_rollout_ref.model.use_fused_kernels
    fa_calls = {}
    for cls in (fa.FlashAttnFunc, fa.FlashAttnVarlenFunc):
        original, signature = cls.forward, inspect.signature(cls.forward)
        def wrap(function, signature, label):
            @functools.wraps(function)
            def tracked(*a, **kw):
                values = signature.bind(*a, **kw)
                values.apply_defaults()
                assert values.arguments["deterministic"] is False
                key = f"{label}:head{values.arguments['q'].shape[-1]}:causal{values.arguments['causal']}:deterministicFalse"
                fa_calls[key] = fa_calls.get(key, 0) + 1
                return function(*a, **kw)
            return tracked
        cls.forward = staticmethod(wrap(original, signature, cls.__name__))
    torch.manual_seed(20261019)
    torch.cuda.manual_seed_all(20261019)
    tokenizer = hf_tokenizer(cfg.actor_rollout_ref.model.path, local_files_only=True)
    model_cfg = AutoConfig.from_pretrained(cfg.actor_rollout_ref.model.path, local_files_only=True, attn_implementation="flash_attention_2")
    update_model_config(model_cfg, override_config_kwargs={"bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id, "pad_token_id": tokenizer.pad_token_id})
    model = AutoModelForImageTextToText.from_pretrained(cfg.actor_rollout_ref.model.path, config=model_cfg,
        torch_dtype=torch.float32, attn_implementation="flash_attention_2", device_map={"": device}, local_files_only=True)
    apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1, use_fused_kernels=False,
                      use_prefix_grouper=False, use_tiled_mlp=False)
    model.to(torch.float32)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))
    full = model.state_dict()
    apply_fsdp2(model, {"mesh": mesh, "mp_policy": MixedPrecisionPolicy(param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32, cast_forward_inputs=True), "offload_policy": None,
        "reshard_after_forward": actor_cfg.fsdp_config.reshard_after_forward,
        "shard_placement_fn": get_shard_placement_fn(world)}, actor_cfg.fsdp_config)
    fsdp2_load_full_state_dict(model, full, mesh, None)
    del full
    torch.cuda.empty_cache()
    optimizer = build_optimizer(model.parameters(), actor_cfg.optim)
    actor = DataParallelPPOActor(actor_cfg, model, actor_optimizer=optimizer)
    payload = torch.load(args.output / "inputs.pt", map_location="cpu", weights_only=False)
    batch = payload["stress_batch"]
    del payload
    tensors = {key: value for key, value in batch.items() if torch.is_tensor(value)}
    tensors["old_log_probs"] = torch.zeros_like(tensors["responses"], dtype=torch.float32)
    tensors["advantages"] = torch.ones_like(tensors["old_log_probs"])
    tensors["rollout_is_weights"] = torch.ones_like(tensors["old_log_probs"])
    mm = np.empty(2, dtype=object)
    mm[:] = batch["multi_modal_inputs"]
    data = DataProto.from_dict(tensors=tensors, non_tensors={"multi_modal_inputs": mm},
                              meta_info={"temperature": 1.0, "pad_token_id": tokenizer.pad_token_id})
    def local(tensor):
        return tensor.to_local() if hasattr(tensor, "to_local") else tensor
    def gradients():
        values = [local(p.grad) for p in model.parameters() if p.grad is not None]
        return {"parameters": len(values), "bytes": sum(t.numel() * t.element_size() for t in values)}
    def adam_state():
        result = {"parameters": 0, "bytes": 0, "all_fp32_cuda": True, "step_min": None, "step_max": None}
        steps = []
        for state in optimizer.state.values():
            if not state:
                continue
            result["parameters"] += 1
            steps.append(float(state["step"]))
            for key in ("exp_avg", "exp_avg_sq"):
                value = local(state[key])
                result["bytes"] += value.numel() * value.element_size()
                result["all_fp32_cuda"] &= value.dtype == torch.float32 and value.device.type == "cuda"
        if steps:
            result.update(step_min=min(steps), step_max=max(steps))
        return result
    calls = []
    original_forward = actor._forward_micro_batch
    def tracked_forward(*a, **kw):
        assert kw.get("return_all_logps") is False and kw.get("distill_topk") is None and kw.get("calculate_entropy") is False
        calls.append({"case": CASES[len(calls) % 2][0], "gradient_before": gradients(),
                      "adam_before": adam_state(), "full_logps": False, "topk": None, "entropy": False})
        return original_forward(*a, **kw)
    actor._forward_micro_batch = tracked_forward
    report = {"status": "running", "passed": False, "rank": rank, "world_size": world,
        "cpu_report_sha256": sha(args.output / "cpu_report.json"), "updates": [],
        "limitations": ["Synthetic text padding and transplanted real response; not reward/quality evidence",
                        "Two single-sequence microbatches per rank, not full 30-prompt Ray aggregation",
                        "No rollout engines or simultaneous vLLM/actor memory; formal monitoring remains required",
                        "Actual image layouts checked: 8 images and 3 images; total image tensor budget is recorded",
                        "Base model weight identity uses file metadata, not newly computed full-weight content hashes"]}
    for update in (1, 2):
        learning_rate = 0.0 if update == 1 else 1e-7
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        before = adam_state()
        if update == 2:
            assert before["parameters"] > 0 and before["all_fp32_cuda"] and before["step_min"] == before["step_max"] == 1
        offset = len(calls)
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        metrics = {key: float(value) for key, value in actor.update_policy(data).items()}
        torch.cuda.synchronize()
        assert all(np.isfinite(value) for value in metrics.values()) and metrics["actor/grad_norm"] > 0
        current_calls = calls[offset:]
        after = adam_state()
        assert len(current_calls) == 2 and current_calls[1]["gradient_before"]["bytes"] > 0
        assert after["parameters"] == current_calls[1]["gradient_before"]["parameters"]
        assert after["all_fp32_cuda"] and after["step_min"] == after["step_max"] == update
        if update == 2:
            assert current_calls[1]["adam_before"]["parameters"] == current_calls[1]["gradient_before"]["parameters"]
        report["updates"].append({"update": update, "learning_rate": learning_rate, "metrics": metrics,
            "seconds": time.perf_counter() - started, "adam_before": before, "adam_after": after,
            "forward_calls": current_calls})
        if rank == 0:
            save(args.attempt / "report.json", report)
            print(json.dumps({"update": update, "metrics": metrics, "adam_bytes": after["bytes"]}), flush=True)
    assert any("head64" in key for key in fa_calls) and any("head256" in key for key in fa_calls)
    report.update(status="complete", passed=True, flash_attention_calls=fa_calls,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
    save(args.attempt / f"rank{rank}.json", report)
    ranks = [None] * world
    dist.all_gather_object(ranks, {"rank": rank, "passed": True, "gradient_norm": report["updates"][-1]["metrics"]["actor/grad_norm"]})
    if rank == 0:
        report["all_ranks"] = ranks
        save(args.attempt / "report.json", report)
        print("MULTIMODAL_MEMORY_CHECK_COMPLETE", flush=True)
    dist.barrier()
    dist.destroy_process_group()


def supervise(args):
    prepared = verify_cpu(args.output)
    if (args.output / "receipt.json").exists():
        print(json.dumps(dict(verify_receipt(args.output), skipped_existing_success=True)), flush=True)
        return
    lockname = "sharedexpert-mm-check-" + hashlib.sha256(str(args.output).encode()).hexdigest() + ".lock"
    with (Path(tempfile.gettempdir()) / lockname).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import pynvml
        pynvml.nvmlInit()
        handles = {gpu: pynvml.nvmlDeviceGetHandleByIndex(gpu) for gpu in (3, 4, 5, 6)}
        if any(pynvml.nvmlDeviceGetComputeRunningProcesses(handle) for handle in handles.values()):
            raise RuntimeError("Memory-check GPUs are occupied; no process was stopped")
        start = host()
        if start["working_set_bytes"] >= 20 * 2**30:
            raise RuntimeError("Host working set is too high for independent model initialization")
        attempt = args.output / "attempts" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:8])
        attempt.mkdir(parents=True)
        cmd = [str(EXPERIMENT / ".venv-train/bin/torchrun"), "--standalone", "--nproc-per-node=4",
               str(Path(__file__).resolve()), "--execute", "--output", str(args.output), "--attempt", str(attempt)]
        summary = {"status": "running", "started_at": timestamp(), "command": cmd,
            "supervisor_identity": process_identity(os.getpid()), "host_before": start,
            "peak_working_set_bytes": start["working_set_bytes"], "peak_current_bytes": start["current_bytes"],
            "gpu_peak_used_bytes": {str(gpu): 0 for gpu in handles}, "guard_limit_gib": args.host_limit_gib,
            "cpu_report_sha256": sha(args.output / "cpu_report.json")}
        interrupted = []
        previous = {sig: signal.signal(sig, lambda signum, frame: interrupted.append(signum)) for sig in (signal.SIGTERM, signal.SIGINT)}
        child = None
        try:
            with (attempt / "gpu.log").open("w", buffering=1) as log, (attempt / "resources.jsonl").open("w", buffering=1) as samples:
                child = spawn_owned(cmd, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpus),
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                summary["torchrun_identity"] = process_identity(child.pid)
                save(attempt / "supervisor.json", summary)
                save(args.output / "active_attempt.json", {"attempt": str(attempt), "supervisor": summary["supervisor_identity"], "torchrun": summary["torchrun_identity"]})
                while child.poll() is None:
                    if interrupted:
                        summary["interrupted_signal"] = interrupted[0]
                        raise RuntimeError(f"Supervisor received signal {interrupted[0]}")
                    sample = dict(host(), time=time.time(), gpu_used_bytes={str(gpu): pynvml.nvmlDeviceGetMemoryInfo(handle).used for gpu, handle in handles.items()})
                    samples.write(json.dumps(sample) + "\n")
                    for key in ("working_set_bytes", "current_bytes"):
                        summary["peak_" + key] = max(summary["peak_" + key], sample[key])
                    for key, value in sample["gpu_used_bytes"].items():
                        summary["gpu_peak_used_bytes"][key] = max(summary["gpu_peak_used_bytes"][key], value)
                    if sample["working_set_bytes"] > args.host_limit_gib * 2**30:
                        summary["guard_triggered"] = True
                        raise RuntimeError("Independent 54 GiB working-set guard triggered")
                    time.sleep(.5)
        except BaseException as exc:
            summary["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            stop_owned(child)
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and any(pynvml.nvmlDeviceGetComputeRunningProcesses(handle) for handle in handles.values()):
                time.sleep(.25)
            summary.update(finished_at=timestamp(), returncode=None if child is None else child.returncode,
                host_after=host(), gpus_released=all(not pynvml.nvmlDeviceGetComputeRunningProcesses(handle) for handle in handles.values()))
            summary["memory_events_delta"] = {key: summary["host_after"]["events"][key] - value for key, value in start["events"].items()}
            summary["status"] = "complete" if child is not None and child.returncode == 0 and "error" not in summary and not interrupted else "failed"
            if interrupted:
                summary["interrupted_signal"] = interrupted[0]
            save(attempt / "supervisor.json", summary)
        if summary["status"] != "complete" or not summary["gpus_released"] or any(summary["memory_events_delta"][key] != 0 for key in ("oom", "oom_kill")):
            raise RuntimeError("Independent four-rank check did not exit cleanly without memory faults")
        report = json.loads((attempt / "report.json").read_text())
        if report.get("passed") is not True or len(report.get("all_ranks", [])) != 4:
            raise RuntimeError("GPU report has not validated all four ranks")
        files = [attempt / name for name in ("report.json", "supervisor.json", "gpu.log", "resources.jsonl", "rank0.json", "rank1.json", "rank2.json", "rank3.json")]
        receipt = {"passed": True, "validated_at": timestamp(), "cpu_report_sha256": sha(args.output / "cpu_report.json"),
            "report": str((attempt / "report.json").relative_to(args.output)),
            "supervisor": str((attempt / "supervisor.json").relative_to(args.output)),
            "validated_artifacts": {str(path.relative_to(args.output)): sha(path) for path in files},
            "shared_teachers": prepared["shared_teachers"], "limitations": report["limitations"]}
        save(args.output / "receipt.json", receipt)
        print(json.dumps(verify_receipt(args.output), indent=2), flush=True)


def dummy_tree(args):
    """CPU-only process-tree fixture; SIGTERM cleans a separately sessioned child."""
    child = spawn_owned([sys.executable, "-c", "import time; time.sleep(120)"], stdin=subprocess.DEVNULL)
    save(args.output / "dummy_tree.json", {"parent": process_identity(os.getpid()), "child": process_identity(child.pid)})
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stopped.append(signum))
    try:
        while not stopped:
            time.sleep(.05)
    finally:
        stop_owned(child)


def self_test_cleanup(args):
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    # A pure-Python fixture tests process-group TERM and the parent-death signal.
    # It imports no torch, NVML or model and cannot start a GPU worker.
    for label, terminate_parent in (("forwarded_term", False), ("parent_death_sigkill", True)):
        with tempfile.TemporaryDirectory(dir=args.output, prefix="cleanup_") as folder:
            directory = Path(folder)
            worker = [sys.executable, str(Path(__file__).resolve()), "--dummy-tree", "--output", str(directory)]
            if terminate_parent:
                code = "import subprocess,time,os,signal,ctypes; p=os.getpid(); c=subprocess.Popen(" + repr(worker) + ",start_new_session=True,preexec_fn=lambda: ctypes.CDLL(None).prctl(1,signal.SIGTERM,0,0,0)); time.sleep(120)"
                child = spawn_owned([sys.executable, "-c", code])
            else:
                child = spawn_owned(worker)
            ready = directory / "dummy_tree.json"
            deadline = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < deadline:
                time.sleep(.05)
            assert ready.exists(), "CPU process fixture did not start"
            identity = json.loads(ready.read_text())
            if terminate_parent:
                os.kill(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            else:
                stop_owned(child)
            deadline = time.monotonic() + 10
            alive = []
            while time.monotonic() < deadline:
                alive = [item for item in identity.values() if (current := process_identity(item["pid"])) and current["start_ticks"] == item["start_ticks"] and current["state"] != "Z"]
                if not alive:
                    break
                time.sleep(.05)
            assert not alive, alive
            results.append({"case": label, "passed": True, "gpu_used": False})
    save(args.output / "cleanup_cpu_checks.json", {"passed": True, "at": timestamp(), "cases": results})
    print(json.dumps(results), flush=True)


def self_test_receipt(args):
    """Exercise real receipt parsing using temporary synthetic CPU fixtures."""
    args.output.mkdir(parents=True, exist_ok=True)
    cpu_sha = "cpu-only-synthetic-fixture-not-gpu-evidence"
    empty = dict(parameters=0, bytes=0, all_fp32_cuda=True, step_min=None, step_max=None)
    warm = dict(parameters=723, bytes=9078531072, all_fp32_cuda=True, step_min=1, step_max=1)
    zero_grad = dict(parameters=0, bytes=0)
    full_grad = dict(parameters=723, bytes=4539265536)
    updates = []
    for step in (1, 2):
        before = empty if step == 1 else warm
        after = dict(warm, step_min=step, step_max=step)
        updates.append({"update": step, "learning_rate": 0.0 if step == 1 else 1e-7,
            "metrics": {"actor/grad_norm": 1.0, "actor/pg_loss": -1.0, "actor/kl_loss": 0.0},
            "adam_before": copy.deepcopy(before), "adam_after": after,
            "forward_calls": [{"case": case[0], "gradient_before": copy.deepcopy(zero_grad if i == 0 else full_grad),
                "adam_before": copy.deepcopy(before), "full_logps": False, "topk": None, "entropy": False}
                for i, case in enumerate(CASES)]})
    base = {"status": "complete", "passed": True, "rank": 0, "world_size": 4,
        "cpu_report_sha256": cpu_sha, "updates": updates,
        "flash_attention_calls": {"head64:deterministicFalse": 1, "head256:deterministicFalse": 1},
        "all_ranks": [{"rank": rank, "passed": True, "gradient_norm": 1.0} for rank in range(4)]}
    base_supervisor = {"status": "complete", "returncode": 0, "gpus_released": True,
        "cpu_report_sha256": cpu_sha, "memory_events_delta": {"oom": 0, "oom_kill": 0}}
    cases = ["valid", "missing_manifest_rank", "wrong_report_path", "different_supervisor_attempt",
             "missing_rank", "duplicate_rank", "failed_rank", "rank_cpu_mismatch", "rank_missing_update",
             "nonfinite_metric", "missing_warm_adam", "missing_second_micro_gradients", "adam_not_fp32_cuda",
             "deterministic_kernel", "supervisor_failed", "host_oom", "gpu_not_released", "artifact_changed"]
    outcomes = []
    for name in cases:
        with tempfile.TemporaryDirectory(dir=args.output, prefix="receipt_cpu_fixture_") as folder:
            output = Path(folder).resolve()
            attempt = output / "attempts/synthetic_fixture"
            attempt.mkdir(parents=True)
            main = copy.deepcopy(base)
            ranks = [dict(copy.deepcopy(base), rank=rank) for rank in range(4)]
            supervisor = copy.deepcopy(base_supervisor)
            if name == "missing_rank": main["all_ranks"].pop()
            if name == "duplicate_rank": main["all_ranks"][3]["rank"] = 2
            if name == "failed_rank": main["all_ranks"][3]["passed"] = False
            if name == "rank_cpu_mismatch": ranks[2]["cpu_report_sha256"] = "wrong"
            if name == "rank_missing_update": ranks[1]["updates"].pop()
            if name == "nonfinite_metric": ranks[2]["updates"][1]["metrics"]["actor/grad_norm"] = math.nan
            if name == "missing_warm_adam": ranks[2]["updates"][1]["adam_before"] = copy.deepcopy(empty)
            if name == "missing_second_micro_gradients": ranks[3]["updates"][1]["forward_calls"][1]["gradient_before"] = copy.deepcopy(zero_grad)
            if name == "adam_not_fp32_cuda": ranks[1]["updates"][1]["adam_after"]["all_fp32_cuda"] = False
            if name == "deterministic_kernel": ranks[2]["flash_attention_calls"] = {"head64:deterministicTrue": 1}
            if name == "supervisor_failed": supervisor["returncode"] = 1
            if name == "host_oom": supervisor["memory_events_delta"]["oom"] = 1
            if name == "gpu_not_released": supervisor["gpus_released"] = False
            save(attempt / "report.json", main)
            save(attempt / "supervisor.json", supervisor)
            for rank, value in enumerate(ranks): save(attempt / f"rank{rank}.json", value)
            for filename in ("gpu.log", "resources.jsonl"): (attempt / filename).write_text("CPU fixture only\n")
            receipt = {"passed": True, "cpu_report_sha256": cpu_sha,
                "report": "attempts/synthetic_fixture/report.json", "supervisor": "attempts/synthetic_fixture/supervisor.json",
                "validated_artifacts": {str(path.relative_to(output)): sha(path) for path in attempt.iterdir()}}
            if name == "missing_manifest_rank": receipt["validated_artifacts"].pop("attempts/synthetic_fixture/rank3.json")
            if name == "wrong_report_path": receipt["report"] = "../report.json"
            if name == "different_supervisor_attempt": receipt["supervisor"] = "attempts/other/supervisor.json"
            if name == "artifact_changed": (attempt / "gpu.log").write_text("changed after hashing\n")
            rejected = False
            try:
                verify_result_records(output, receipt, cpu_sha)
            except (RuntimeError, KeyError, ValueError, OSError):
                rejected = True
            assert rejected == (name != "valid"), (name, rejected)
            outcomes.append({"case": name, "passed": True, "rejected": rejected})
    save(args.output / "receipt_cpu_checks.json", {"passed": True, "at": timestamp(), "gpu_started": False,
        "scope": "Temporary synthetic CPU fixtures only; not a GPU receipt", "cases": outcomes})
    print(json.dumps({"passed": True, "cases": len(outcomes), "gpu_started": False}), flush=True)


if __name__ == "__main__":
    args = arguments()
    if args.prepare:
        prepare(args)
    elif args.check_cpu:
        report = verify_cpu(args.output)
        print(json.dumps({"status": report["status"], "gpu_started": False, "cases": report["cases"]}, indent=2))
    elif args.supervise:
        supervise(args)
    elif args.verify:
        print(json.dumps(verify_receipt(args.output), indent=2))
    elif args.execute:
        # Elastic ranks may be in distinct process groups. If torchrun itself
        # dies unexpectedly, each rank must also exit rather than retain a GPU.
        parent_death_signal(os.getppid())
        execute(args)
    elif args.self_test_cleanup:
        self_test_cleanup(args)
    elif args.self_test_receipt:
        self_test_receipt(args)
    else:
        dummy_tree(args)
