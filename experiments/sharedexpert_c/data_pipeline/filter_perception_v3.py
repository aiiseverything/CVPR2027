#!/usr/bin/env python3
"""Screen fixed-base-model perception candidates, eight completions per question.

This is dataset inference, not training. Completed groups are immutable on resume.
The final parquet is published only after all six exact quotas are satisfied.
"""
from __future__ import annotations

import argparse
from collections import Counter
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
sys.path.insert(0, str(HERE))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def record_identity(record):
    keys = ("id", "source", "source_id", "split", "question", "question_sha256",
            "image_sha256", "task", "answers", "bbox", "perception_category", "selection_rule")
    return digest({k: record[k] for k in keys if k in record})


def record_seed(record_id, seed):
    return int(hashlib.sha256(f"{seed}:{record_id}".encode()).hexdigest()[:8], 16)


def read_journal(path):
    """Ignore only an uncommitted final partial line left by a terminated writer."""
    if not path.exists():
        return {}
    records, end = {}, 0
    with path.open("rb") as f:
        for raw in f:
            if not raw.endswith(b"\n"):
                break
            item = json.loads(raw)
            rid = item["id"]
            if rid in records:
                raise ValueError(f"Duplicate committed group: {rid}")
            records[rid] = item
            end += len(raw)
    if path.stat().st_size != end:
        with path.open("r+b") as f:
            f.truncate(end)
    return records


def load_candidates(pool_dirs):
    import prepare_perception_v2 as v2
    records, seen, evidence = [], {}, []
    for directory in pool_dirs:
        manifest_path = directory / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        record_path = directory / "candidates.records.jsonl"
        if (manifest.get("artifact_kind") != "candidate_pool" or
                manifest.get("difficulty_filtered") is not False or
                manifest.get("training_ready") is not False):
            raise ValueError(f"Invalid candidate-pool publication: {directory}")
        for name, key in (("candidates.records.jsonl", "records_sha256"),
                          ("candidates.parquet", "data_sha256"), ("audit.json", "audit_sha256")):
            if file_sha(directory / name) != manifest.get(key):
                raise ValueError(f"Candidate-pool artifact changed: {directory / name}")
        if (manifest.get("reward_sha256") != file_sha(EXPERIMENT / "rewards.py") or
                manifest.get("original_rules_sha256") != file_sha(HERE / "prepare_perception_v2.py")):
            raise ValueError("Candidate rules/reward source changed")
        if manifest.get("target_category_weights") != {k: v / 100 for k, v in v2.WEIGHTS.items()}:
            raise ValueError("Candidate target mixture differs from original")
        pool_records = [json.loads(line) for line in record_path.read_text().splitlines() if line.strip()]
        if (len(pool_records) != manifest.get("n") or
                dict(Counter(r["perception_category"] for r in pool_records)) != manifest.get("category_counts")):
            raise ValueError("Candidate-pool counts do not match manifest")
        evidence.append({"directory": str(directory.resolve()),
                         "manifest_sha256": file_sha(manifest_path),
                         "records_sha256": file_sha(record_path), "manifest": manifest})
        for record in pool_records:
            rid, identity = record["id"], record_identity(record)
            if rid in seen:
                if seen[rid] != identity:
                    raise ValueError(f"Candidate changed between pools: {rid}")
                continue
            if record["split"] != "train":
                raise ValueError(f"Non-train candidate: {rid}")
            seen[rid] = identity
            records.append(record)
    return records, evidence


def model_identity(model):
    wanted = sorted(p for p in model.iterdir() if p.is_file() and
                    (p.suffix in {".json", ".safetensors", ".jinja"} or p.name == "merges.txt"))
    if not any(p.suffix == ".safetensors" for p in wanted):
        raise ValueError("No model weights found")
    return {p.name: {"bytes": p.stat().st_size, "sha256": file_sha(p)} for p in wanted}


def messages_for(record, protocol):
    import prepare_perception_v2 as v2
    row = v2.package(record, 0, protocol)
    images = iter(row["images"])
    messages = []
    for message in row["prompt"]:
        content = []
        for segment in re.split("(<image>)", message["content"]):
            if segment == "<image>":
                item = dict(next(images))
                item["image"] = item.pop("path")
                content.append({"type": "image", **item})
            elif segment:
                content.append({"type": "text", "text": segment})
        messages.append({"role": message["role"], "content": content})
    return messages


def prepare_prompt(record, protocol, processor):
    """Match the train agent's Qwen resize, template, tokenization and dedup."""
    from qwen_vl_utils import process_vision_info
    import prepare_perception_v2 as v2
    for path, expected in zip(record["images"], record["image_sha256"], strict=True):
        if v2.pixel_hash(path) != expected:
            raise ValueError(f"Candidate image changed: {record['id']} {path}")
    messages = messages_for(record, protocol)
    images, videos = process_vision_info(messages, image_patch_size=processor.image_processor.patch_size)
    if videos:
        raise ValueError("Unexpected video in perception candidate")
    text = processor.apply_chat_template(messages, tools=[], add_generation_prompt=True,
                                         tokenize=False, enable_thinking=protocol["thinking"])
    batch = processor(text=[text], images=images, return_tensors="pt", do_sample_frames=False)
    full_ids = batch["input_ids"].squeeze(0).tolist()
    del batch
    # Screening excludes over-budget questions rather than altering their meaning.
    if len(full_ids) > protocol["prompt_length"]:
        return None, {"prompt_tokens": len(full_ids), "reason": "prompt_over_budget"}
    ids = full_ids
    if "Qwen2VLImageProcessor" in type(processor.image_processor).__name__:
        visual = {processor.image_token_id, processor.video_token_id}
        ids = [token for i, token in enumerate(ids)
               if i == 0 or not (token in visual and ids[i - 1] in visual)]
    return {"prompt_token_ids": ids, "multi_modal_data": {"image": images}}, {
        "prompt_tokens": len(full_ids), "prompt_token_ids_sha256": digest(full_ids),
        "expected_prompt_token_ids": full_ids,
        "first_resize_wh": [list(image.size) for image in images],
    }


def category_status(records, committed, target):
    import prepare_perception_v2 as v2
    counts = Counter()
    accepted = []
    for record in records:
        receipt = committed.get(record["id"])
        if receipt and receipt.get("difficulty_filter", {}).get("accepted"):
            counts[record["perception_category"]] += 1
            if record["perception_category"] == "existence_confusion":
                counts["existence_" + v2.consensus(record)[0]] += 1
            accepted.append({**record, "difficulty_filter": receipt["difficulty_filter"]})
    quotas = {key: target * weight // 100 for key, weight in v2.WEIGHTS.items()}
    remaining = {key: max(0, quota - counts[key]) for key, quota in quotas.items()}
    half = quotas["existence_confusion"] // 2
    remaining["existence_yes"] = max(0, half - counts["existence_yes"])
    remaining["existence_no"] = max(0, half - counts["existence_no"])
    return accepted, dict(counts), quotas, remaining


def publish(records, committed, out, protocol, run_fingerprint, target, seed, pool_evidence):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import prepare_perception_v2 as v2
    from perception_v3_filtering import select_records
    accepted, counts, quotas, remaining = category_status(records, committed, target)
    if any(remaining.values()):
        return False
    selected = select_records(accepted, target, seed, balance_existence=True)
    payloads = []
    for i, record in enumerate(selected):
        row = v2.package(record, i, protocol)
        row["ability"] = "perception_v3"
        row["extra_info"].update(selection_version="perception-v3-base-8-keep-2-to-6-v1",
                                 screening_n_correct=record["difficulty_filter"]["n_correct"],
                                 screening_n_completions=8,
                                 screening_run_sha256=run_fingerprint,
                                 screening_candidate_sha256=record_identity(record))
        payloads.append(row)
    destination = out / "dataset"
    destination.mkdir(exist_ok=True)
    # The final manifest is the publication marker; unfinished files are never advertised.
    parquet_path = destination / "perception_v3.parquet"
    tmp = parquet_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(payloads), tmp)
    tmp.replace(parquet_path)
    records_path = destination / "perception_v3.records.jsonl"
    temp_records = records_path.with_suffix(".jsonl.tmp")
    temp_records.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected))
    temp_records.replace(records_path)
    atomic_json(destination / "manifest.json", {
        "status": "complete", "records": len(selected), "category_counts": quotas,
        "screening": "Exactly one committed group of eight; 2 <= correct <= 6 for every row",
        "run_sha256": run_fingerprint, "parquet_sha256": file_sha(parquet_path),
        "records_sha256": file_sha(records_path), "groups_sha256": file_sha(out / "groups.jsonl"),
        "all_accepted_counts": counts, "pool_evidence": pool_evidence,
        "training_reward_changed": False,
        "limitation": "Screening is measured at fixed base weights; future training groups can differ.",
    })
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, default=HERE / "prepared/perception_v3_filter")
    parser.add_argument("--model", type=Path, default=EXPERIMENT / "models/Qwen3.5-4B")
    parser.add_argument("--protocol", type=Path, default=EXPERIMENT / "training/config/common_local_100.json")
    parser.add_argument("--target", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-prompts", type=int, default=12)
    parser.add_argument("--prepare-only", action="store_true", help="CPU checks only, no model inference")
    args = parser.parse_args()
    if args.target <= 0 or args.target % 40 or args.batch_prompts < 1:
        raise ValueError("Target must be a positive multiple of40; batch must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
                 "HF_HUB_DISABLE_TELEMETRY", "VLLM_NO_USAGE_STATS"):
        os.environ.setdefault(name, "1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fs = os.statvfs(args.output_dir)
    if fs.f_files and fs.f_favail < 1024:
        raise RuntimeError("Insufficient free filesystem inodes for screening outputs")
    lock = (args.output_dir / ".lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    from perception_v3_filtering import accept_group
    import prepare_perception_v2 as v2
    protocol = json.loads(args.protocol.read_text())
    if protocol["group_size"] != 8:
        raise ValueError("Original training group size must be eight")
    records, pool_evidence = load_candidates(args.pool_dir)
    if any(p["manifest"]["protocol_sha256"] != file_sha(args.protocol) for p in pool_evidence):
        raise ValueError("Candidate pool was prepared against another training protocol")
    if not records:
        raise ValueError("No candidates")
    model = args.model.resolve()
    print(json.dumps({"stage": "cpu_preflight", "candidates": len(records), "model": str(model)}), flush=True)
    run_spec = {
        "version": "perception-v3-screen-v1", "model_files": model_identity(model),
        "sampling": {k: protocol[k] for k in ("temperature", "top_p", "top_k", "thinking",
                      "prompt_length", "response_length", "min_image_pixels", "max_image_pixels", "max_total_image_pixels")},
        "group_size": 8, "seed": args.seed, "target": args.target,
        "correctness": {"bbox": "IoU>=0.5", "vqa": "score>=1-1e-9", "anls": "score>=1-1e-9"},
        "reward_source_sha256": file_sha(EXPERIMENT / "rewards.py"),
        "filter_source_sha256": file_sha(HERE / "perception_v3_filtering.py"),
        "driver_source_sha256": file_sha(Path(__file__)),
        "original_rules_sha256": file_sha(HERE / "prepare_perception_v2.py"),
        "dependencies": {name: importlib.metadata.version(name) for name in
                         ("torch", "transformers", "vllm", "qwen-vl-utils", "pyarrow")},
        "bad_words": ["<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|video_pad|>"],
        "preprocessing": "train agent two-stage processor; Qwen image token dedup; legacy vLLM pixel kwargs",
    }
    fingerprint = digest(run_spec)
    run_path = args.output_dir / "run.json"
    if (run_path.exists() and json.loads(run_path.read_text())["run_sha256"] != fingerprint and
            (args.output_dir / "groups.jsonl").exists() and (args.output_dir / "groups.jsonl").stat().st_size):
        raise ValueError("Frozen model/sampling/scoring changed; use a different output directory")
    atomic_json(run_path, {"run_sha256": fingerprint, "spec": run_spec, "model_path": str(model)})
    history_path = args.output_dir / "candidate_pools.json"
    if history_path.exists():
        previous_pools = json.loads(history_path.read_text())
        current = {p["directory"]: p for p in pool_evidence}
        if any(current.get(p["directory"]) != p for p in previous_pools):
            raise ValueError("Previous frozen candidate pool changed or was omitted; append new --pool-dir entries")
    atomic_json(args.output_dir / "candidate_pools.json", pool_evidence)
    committed = read_journal(args.output_dir / "groups.jsonl")
    identities = {r["id"]: record_identity(r) for r in records}
    record_by_id = {r["id"]: r for r in records}
    for rid, receipt in committed.items():
        if identities.get(rid) != receipt["candidate_sha256"] or receipt["run_sha256"] != fingerprint:
            raise ValueError(f"Cached result has incompatible candidate/protocol: {rid}")
        if receipt.get("completions") is not None:
            recalculated = accept_group(record_by_id[rid], receipt["completions"])
            if recalculated != receipt["difficulty_filter"]:
                raise ValueError(f"Cached scores failed recomputation: {rid}")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    representatives = {}
    for record in records:
        representatives.setdefault(record["perception_category"], record)
    cpu_checks = []
    for record in representatives.values():
        prompt, metadata = prepare_prompt(record, protocol, processor)
        metadata.pop("expected_prompt_token_ids", None)
        cpu_checks.append({"id": record["id"], "category": record["perception_category"], **metadata})
    atomic_json(args.output_dir / "cpu_preflight.json", {
        "status": "passed", "gpu_inference_performed": False, "run_sha256": fingerprint,
        "candidate_count": len(records), "representative_input_checks": cpu_checks,
        "processor": type(processor).__name__, "image_processor": type(processor.image_processor).__name__,
        "model_path": str(model), "free_disk_bytes": __import__("shutil").disk_usage(args.output_dir).free,
    })
    if args.prepare_only:
        print("CPU preflight complete; no GPU inference performed", flush=True)
        return
    if publish(records, committed, args.output_dir, protocol, fingerprint, args.target, args.seed, pool_evidence):
        print("Exact filtered dataset already complete", flush=True)
        return
    gpu_lock = open(f"/tmp/sharedexpert-perception-v3-gpu-{args.gpu}.lock", "a")
    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid",
                                       "--format=csv,noheader"], text=True).strip()
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                         "--format=csv,noheader"], text=True)
    if any(line.startswith(gpu_uuid) for line in processes.splitlines()):
        raise RuntimeError("Selected GPU has an existing compute process")
    from vllm import LLM, SamplingParams
    llm = LLM(model=str(model), tensor_parallel_size=1, dtype="bfloat16", max_model_len=10240,
              gpu_memory_utilization=0.70, max_num_seqs=120, max_num_batched_tokens=10240,
              seed=args.seed, limit_mm_per_prompt={"image": 1}, mm_processor_cache_gb=0,
              mm_processor_kwargs={"min_pixels": protocol["min_image_pixels"],
                                   "max_pixels": protocol["max_image_pixels"]},
              compilation_config={"pass_config": {"fuse_allreduce_rms": False}},
              kernel_config={"enable_flashinfer_autotune": False}, enable_prefix_caching=True)
    start = time.monotonic()
    with (args.output_dir / "groups.jsonl").open("a") as journal:
        while True:
            _, counts, quotas, remaining = category_status(records, committed, args.target)
            if not any(remaining.values()):
                break
            pending = []
            scheduled = Counter()
            for record in records:
                if record["id"] in committed:
                    continue
                category = record["perception_category"]
                key = "existence_" + v2.consensus(record)[0] if category == "existence_confusion" else category
                if remaining[key] <= 0:
                    continue
                # Round-robin candidates in the pool keep task coverage in each batch.
                if scheduled[key] >= remaining[key]:
                    continue
                pending.append(record)
                scheduled[key] += 1
                if len(pending) == args.batch_prompts:
                    break
            if not pending:
                break
            inputs, sampling, prepared = [], [], []
            for record in pending:
                prompt, metadata = prepare_prompt(record, protocol, processor)
                if prompt is None:
                    receipt = {"id": record["id"], "candidate_sha256": record_identity(record),
                               "run_sha256": fingerprint, "input_rejection": metadata,
                               "difficulty_filter": {"accepted": False, "rejection_reason": metadata["reason"]}}
                    journal.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                    journal.flush()
                    committed[record["id"]] = receipt
                    continue
                seed = record_seed(record["id"], args.seed)
                inputs.append(prompt)
                sampling.append(SamplingParams(n=8, temperature=protocol["temperature"],
                    top_p=protocol["top_p"], top_k=protocol["top_k"], max_tokens=protocol["response_length"],
                    seed=seed, repetition_penalty=1.0, skip_special_tokens=False,
                    bad_words=run_spec["bad_words"]))
                prepared.append((record, metadata, seed))
            if inputs:
                outputs = llm.generate(inputs, sampling, use_tqdm=False)
                if len(outputs) != len(prepared):
                    raise RuntimeError("Incomplete inference batch; no missing responses counted as errors")
                for output, (record, metadata, seed) in zip(outputs, prepared, strict=True):
                    expected_ids = metadata.pop("expected_prompt_token_ids")
                    if list(output.prompt_token_ids) != expected_ids:
                        raise RuntimeError(f"Training/inference tokenization mismatch: {record['id']}")
                    completions = [{"index": completion.index,
                                    "text": processor.tokenizer.decode(completion.token_ids, skip_special_tokens=False),
                                    "finish_reason": completion.finish_reason,
                                    "token_ids": list(completion.token_ids)} for completion in output.outputs]
                    assessed = accept_group(record, completions)
                    if assessed["n_correct"] is None:
                        raise RuntimeError(f"Uncommitted incomplete group {record['id']}: {assessed['rejection_reason']}")
                    receipt = {"id": record["id"], "candidate_sha256": record_identity(record),
                               "run_sha256": fingerprint, "sampling_seed": seed,
                               "category": record["perception_category"], "input": metadata,
                               "completions": completions, "difficulty_filter": assessed}
                    journal.write(json.dumps(receipt, ensure_ascii=False) + "\n")
                    journal.flush()
                    committed[record["id"]] = receipt
                os.fsync(journal.fileno())
            accepted, counts, quotas, remaining = category_status(records, committed, args.target)
            status = {"status": "screening", "screened": len(committed), "available_candidates": len(records),
                      "accepted": len(accepted), "accepted_counts": counts, "quotas": quotas,
                      "remaining": remaining, "elapsed_seconds": round(time.monotonic() - start, 1),
                      "run_sha256": fingerprint}
            atomic_json(args.output_dir / "status.json", status)
            print(json.dumps(status), flush=True)
    done = publish(records, committed, args.output_dir, protocol, fingerprint, args.target, args.seed, pool_evidence)
    accepted, counts, quotas, remaining = category_status(records, committed, args.target)
    atomic_json(args.output_dir / "status.json", {
        "status": "complete" if done else "needs_more_candidates", "screened": len(committed),
        "accepted": len(accepted), "accepted_counts": counts, "quotas": quotas, "remaining": remaining,
        "run_sha256": fingerprint, "dataset": str(args.output_dir / "dataset") if done else None})
    print(json.dumps({"complete": done, "remaining": remaining}), flush=True)


if __name__ == "__main__":
    main()
