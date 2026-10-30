#!/usr/bin/env python3
"""Train the fully screened perception v3 mixture using the frozen v2 recipe.

CPU configuration/data/dependency/resource checks precede formal training.
There is no preliminary GPU training run or smoke-receipt prerequisite.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True

TRAINING = Path(__file__).resolve().parent
EXPERIMENT = TRAINING.parent
WORKSPACE = TRAINING.parents[2]
PIPELINE = EXPERIMENT / "data_pipeline"
sys.path[:0] = [str(TRAINING), str(PIPELINE), str(WORKSPACE), str(WORKSPACE / "pre-lab/repo")]
import launch_local as local
import perception_v3_filtering as filtering

BASELINE = EXPERIMENT / "runs/perception_v2"
RUN_DIR = EXPERIMENT / "runs/perception_v3"
DEFAULT_MODEL = EXPERIMENT / "models/Qwen3.5-4B"
DEFAULT_DATASET = PIPELINE / "prepared/perception_v3_filter/dataset"
STORAGE_REFERENCE = TRAINING / "config/perception_v3_storage_reference.json"
SHARED_FILES = ("launch.py", "launch_local.py", "run.py", "agent_loop.py", "reward_manager.py")
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "thinking", "prompt_length", "response_length",
                 "min_image_pixels", "max_image_pixels", "max_total_image_pixels")


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


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def load_protocol(gpus, dataset_dir=DEFAULT_DATASET, model=DEFAULT_MODEL):
    baseline = read_json(BASELINE / "protocol.json")
    common = read_json(TRAINING / "config/common_local_100.json")
    require(common == baseline["protocol"], "Common training protocol changed since perception_v2")
    require(gpus == baseline["gpus"] == ["0", "1"], "Formal training requires the original GPUs 0,1")
    require((common["steps"], common["batch_size"], common["group_size"]) == (100, 30, 8),
            "Formal budget must remain 100 steps, batch30, eight completions")
    model = Path(model).resolve()
    require(model.is_dir() and model.name == Path(common["model"]).name,
            "The relocated original Qwen3.5-4B model is required")
    protocol = {**common, "model": str(model)}
    teacher = read_json(TRAINING / "config/perception_v3.json")
    require(teacher["teacher"] == "perception_v3" and teacher["category_weights"] ==
            {key: weight / 100 for key, weight in filtering.WEIGHTS.items()}, "Wrong perception_v3 mixture")
    teacher["data_file"] = str(Path(dataset_dir).resolve() / "perception_v3.parquet")
    return protocol, teacher, baseline


def make_overrides(protocol, teacher, gpus):
    return local.make_overrides(protocol, teacher, len(gpus), RUN_DIR, 100, 30)


def relocate_baseline_paths(value, old_model, new_model):
    """Only recognized workspace aliases and the original model path may move."""
    if isinstance(value, dict):
        return {key: relocate_baseline_paths(item, old_model, new_model) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate_baseline_paths(item, old_model, new_model) for item in value]
    if not isinstance(value, str):
        return value
    if value == old_model or value.startswith(old_model + "/"):
        return new_model + value[len(old_model):]
    for old_root in ("/autodl-fs/data/CVPR2027", "/root/autodl-fs/CVPR2027"):
        if value == old_root or value.startswith(old_root + "/"):
            return str(WORKSPACE) + value[len(old_root):]
    return value


def compose_and_verify(protocol, teacher, gpus):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(WORKSPACE / protocol["repository"] / "verl/trainer/config"),
                               version_base=None):
        config = compose(config_name="ppo_trainer", overrides=make_overrides(protocol, teacher, gpus))
    original = read_json(BASELINE / "protocol.json")["protocol"]
    expected = OmegaConf.create(relocate_baseline_paths(
        OmegaConf.to_container(OmegaConf.load(BASELINE / "resolved_config.yaml"), resolve=True),
        original["model"], protocol["model"]))
    for key, value in {
        "data.train_files": [teacher["data_file"]], "trainer.experiment_name": RUN_DIR.name,
        "trainer.default_local_dir": str(RUN_DIR / "checkpoints"),
        "trainer.rollout_data_dir": str(RUN_DIR / "rollouts"),
    }.items():
        OmegaConf.update(expected, key, value)
    actual_tree, expected_tree = (OmegaConf.to_container(value, resolve=True) for value in (config, expected))
    require(actual_tree == expected_tree,
            "Resolved configuration differs from perception_v2 beyond dataset/run/model/workspace paths")
    return config


def recipe_identity(protocol, baseline):
    shared = {name: local.sha256(TRAINING / name) for name in SHARED_FILES}
    require(shared == {name: baseline["training_code_sha256"][name] for name in SHARED_FILES},
            "Shared trainer changed since perception_v2")
    reward = local.sha256(EXPERIMENT / "rewards.py")
    require(reward == baseline["reward_sha256"], "Training reward changed since perception_v2")
    repository = WORKSPACE / protocol["repository"]
    head = subprocess.check_output(["git", "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    require(head == protocol["repository_commit"], "Training repository HEAD changed")
    patch = subprocess.check_output(["git", "-C", str(repository), "diff", "HEAD", "--"], text=True)
    patch_hash = hashlib.sha256(patch.encode()).hexdigest()
    require(patch_hash == baseline["repository_diff_sha256"], "Training repository patch changed")
    return {"reward_sha256": reward, "repository_diff_sha256": patch_hash,
            "training_code_sha256": {**shared, "launch_perception_v3.py": local.sha256(__file__)},
            "baseline_protocol_sha256": local.sha256(BASELINE / "protocol.json"),
            "baseline_resolved_config_sha256": local.sha256(BASELINE / "resolved_config.yaml")}, patch


def journal_prefixes(path, lengths):
    """Hash all requested immutable journal prefixes in one streaming pass."""
    result, position, count, previous_byte = {}, 0, 0, b""
    checksum = hashlib.sha256()
    require(all(type(length) is int and length >= 0 for length in lengths), "Invalid execution journal byte count")
    with Path(path).open("rb") as journal:
        for length in sorted(set(lengths)):
            while position < length:
                block = journal.read(min(8 << 20, length - position))
                require(bool(block), "Execution journal prefix exceeds current journal size")
                checksum.update(block)
                count += block.count(b"\n")
                position += len(block)
                previous_byte = block[-1:]
            require(length == 0 or previous_byte == b"\n", "Execution journal prefix ends inside an uncommitted group")
            result[length] = {"sha256": checksum.copy().hexdigest(), "count": count}
    return result


def validate_execution_provenance(root, screen):
    """Validate optional parallel execution history without altering old groups."""
    root = Path(root)
    directory = root / "executions"
    entries = sorted(directory.glob("parallel-*.json"))
    legacy = directory / "legacy-before-parallel.groups.jsonl"
    if not entries:
        require(not legacy.exists(), "Parallel transition snapshot has no execution receipt")
        return {"backend": "original_single_gpu", "execution_receipts": {}}
    require(legacy.is_file(), "Missing preserved single-GPU journal snapshot")
    journal = root / "groups.jsonl"
    core_hash = local.sha256(PIPELINE / "filter_perception_v3.py")
    parallel_hash = local.sha256(PIPELINE / "filter_perception_v3_parallel.py")
    require(screen["spec"].get("driver_source_sha256") == core_hash,
            "Parallel execution core differs from frozen screening run")
    records, lengths = [], [legacy.stat().st_size]
    for path in entries:
        item = read_json(path)
        require(item.get("schema") == 1 and item.get("backend") == "independent_gpu_replicas" and
                item.get("devices") == [0, 1] and item.get("tensor_parallel_size_per_worker") == 1,
                "Invalid parallel execution topology")
        require(item.get("run_sha256") == screen["run_sha256"] and item.get("core_driver_sha256") == core_hash,
                "Parallel execution changed the frozen screening core or run")
        archived_hash = item.get("parallel_driver_sha256", "")
        require(len(archived_hash) == 64 and all(char in "0123456789abcdef" for char in archived_hash),
                "Invalid parallel driver hash")
        source = Path("sources") / f"parallel-{archived_hash}.py"
        require(item.get("parallel_driver_source") == str(source) and
                (directory / source).is_file() and local.sha256(directory / source) == archived_hash,
                "Archived parallel execution source changed")
        require(item.get("status") in {"complete", "failed"}, "Parallel execution has not reached a terminal state")
        require(type(item.get("pid")) is int and item["pid"] > 0 and
                type(item.get("start_ticks")) is int and item["start_ticks"] >= 0,
                "Invalid parallel parent process identity")
        require(type(item.get("batch_prompts")) is int and item["batch_prompts"] > 0,
                "Invalid parallel execution batch size")
        try:
            started = datetime.datetime.fromisoformat(item["started_at"])
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeError("Invalid parallel execution start time") from exc
        require(started.tzinfo is not None, "Parallel execution start time requires an explicit timezone")
        for key in ("initial_groups_bytes", "initial_group_count"):
            require(type(item.get(key)) is int and item[key] >= 0, "Invalid initial execution journal boundary")
        lengths.append(item["initial_groups_bytes"])
        final_keys = {"final_groups_bytes", "final_group_count", "final_groups_sha256"}
        has_final = bool(final_keys.intersection(item))
        require(not has_final or final_keys <= item.keys(), "Incomplete final execution journal evidence")
        require(item["status"] != "complete" or has_final, "Completed execution has no final journal evidence")
        if has_final:
            require(type(item["final_groups_bytes"]) is int and
                    item["final_groups_bytes"] >= item["initial_groups_bytes"] and
                    type(item["final_group_count"]) is int and
                    item["final_group_count"] >= item["initial_group_count"],
                    "Execution journal shrank or final boundary is invalid")
            lengths.append(item["final_groups_bytes"])
        workers = item.get("workers")
        require(isinstance(workers, list), "Missing parallel worker process evidence")
        devices = []
        for worker in workers:
            require(type(worker.get("gpu")) is int and worker["gpu"] in (0, 1) and
                    type(worker.get("pid")) is int and worker["pid"] > 0 and
                    type(worker.get("start_ticks")) is int and worker["start_ticks"] >= 0,
                    "Invalid parallel worker process identity")
            devices.append(worker["gpu"])
        require(len(set(devices)) == len(devices), "Duplicate parallel worker GPU")
        if item["status"] == "complete" and item["final_group_count"] > item["initial_group_count"]:
            require(sorted(devices) == [0, 1], "Completed parallel generation lacks both worker identities")
        records.append((path, item, started))
    latest_path, latest, _ = max(records, key=lambda entry: (entry[2], entry[0].name))
    require(latest["parallel_driver_sha256"] == parallel_hash, "Latest parallel execution source differs from current wrapper")
    prefixes = journal_prefixes(journal, lengths)
    legacy_size, legacy_hash = legacy.stat().st_size, local.sha256(legacy)
    require(prefixes[legacy_size]["sha256"] == legacy_hash, "Preserved single-GPU groups changed in the current journal")
    require(any(item["initial_groups_bytes"] == legacy_size and
                item["initial_groups_sha256"] == legacy_hash and
                item["initial_group_count"] == prefixes[legacy_size]["count"] for _, item, _ in records),
            "No execution receipt binds the preserved single-GPU transition")
    for _, item, _ in records:
        for stage in ("initial", "final"):
            key = stage + "_groups_bytes"
            if key not in item:
                continue
            prefix = prefixes[item[key]]
            require(prefix["sha256"] == item[stage + "_groups_sha256"] and
                    prefix["count"] == item[stage + "_group_count"],
                    "Execution journal prefix hash or group count changed")
    journal_size = journal.stat().st_size
    require(any(item["status"] == "complete" and item["final_groups_bytes"] == journal_size
                for _, item, _ in records), "Final screening journal is not covered by a completed parallel execution")
    return {"backend": "independent_gpu_replicas", "devices": [0, 1],
            "execution_receipts": {path.name: local.sha256(path) for path, _, _ in records},
            "legacy_groups_sha256": legacy_hash, "legacy_group_count": prefixes[legacy_size]["count"],
            "legacy_groups_bytes": legacy_size, "current_parallel_driver_sha256": parallel_hash,
            "latest_execution": latest_path.name}


def validate_dataset(dataset_dir, protocol, verify_model=True, verify_images=True):
    """Independently replay all selected raw receipts before accepting a dataset."""
    import pyarrow.parquet as pq
    import prepare_perception_v2 as v2
    from filter_perception_v3 import model_identity, record_identity, record_seed
    from preflight import check_dataset
    dataset_dir = Path(dataset_dir)
    root = dataset_dir.parent
    manifest = read_json(dataset_dir / "manifest.json")
    require(manifest.get("status") == "complete" and manifest.get("records") == 3000,
            "A completed 3000-row screened dataset is required")
    expected = {key: weight * 30 for key, weight in filtering.WEIGHTS.items()}
    require(manifest.get("category_counts") == expected, "Dataset manifest has incorrect category quotas")
    require(manifest.get("training_reward_changed") is False, "Training rewards must remain unchanged")
    paths = {"parquet": dataset_dir / "perception_v3.parquet",
             "records": dataset_dir / "perception_v3.records.jsonl", "groups": root / "groups.jsonl"}
    for kind, path in paths.items():
        require(local.sha256(path) == manifest[kind + "_sha256"], f"Screened {kind} artifact changed")
    screen = read_json(root / "run.json")
    require(digest(screen["spec"]) == screen["run_sha256"] == manifest["run_sha256"],
            "Screening run fingerprint is inconsistent")
    spec = screen["spec"]
    require(spec["group_size"] == 8 and spec["target"] == 3000, "Wrong screening target or group size")
    require(spec["sampling"] == {key: protocol[key] for key in SAMPLING_KEYS},
            "Screening sampling or image settings differ from formal training")
    require(spec["reward_source_sha256"] == local.sha256(EXPERIMENT / "rewards.py") and
            spec["filter_source_sha256"] == local.sha256(PIPELINE / "perception_v3_filtering.py"),
            "Screening reward or correctness implementation changed")
    execution_provenance = validate_execution_provenance(root, screen)
    if verify_model:
        require(model_identity(Path(protocol["model"])) == spec["model_files"],
                "Training base weights or processor differ from frozen screening model")
    rows = pq.read_table(paths["parquet"]).to_pylist()
    records = [json.loads(line) for line in paths["records"].read_text().splitlines() if line.strip()]
    require(len(rows) == len(records) == 3000, "Screened row count differs from manifest")
    # This checks per-record receipt hashes, uniqueness and feasibility without
    # changing the final frozen row order.
    require(filtering.max_feasible_size(records, balance_existence=True) == 3000,
            "Screening receipts or balanced category quotas are invalid")
    by_id = {record["id"]: record for record in records}
    seen_groups, replayed = set(), set()
    with paths["groups"].open("rb") as journal:
        for raw in journal:
            require(raw.endswith(b"\n"), "Screening journal has an uncommitted tail")
            group = json.loads(raw)
            identity = group["id"]
            require(identity not in seen_groups, "Duplicate raw screening group")
            seen_groups.add(identity)
            if identity not in by_id:
                continue
            record = by_id[identity]
            require(group["run_sha256"] == screen["run_sha256"] and
                    group["candidate_sha256"] == record_identity(record) and
                    group["sampling_seed"] == record_seed(identity, spec["seed"]),
                    "Selected raw group has incompatible candidate/model/seed")
            rescored = filtering.accept_group(record, group["completions"])
            require(rescored == group["difficulty_filter"] == record["difficulty_filter"] and rescored["accepted"],
                    "Selected raw group failed independent eight-completion scoring")
            replayed.add(identity)
    require(replayed == set(by_id), "Selected training record has no raw eight-completion evidence")
    category_order = []
    image_hashes = {}
    for index, (row, record) in enumerate(zip(rows, records, strict=True)):
        packaged = v2.package(record, index, protocol)
        for key in ("prompt", "images", "data_source", "reward_model"):
            require(row[key] == packaged[key], f"Training row differs from screened candidate: {record['id']} {key}")
        require(row["ability"] == "perception_v3", "Wrong teacher identity in parquet")
        extra = row["extra_info"]
        for key, value in packaged["extra_info"].items():
            if key != "selection_version":
                require(extra[key] == value, f"Training row metadata changed: {record['id']} {key}")
        require(extra["screening_n_correct"] == record["difficulty_filter"]["n_correct"] and
                extra["screening_n_completions"] == 8 and
                extra["screening_run_sha256"] == screen["run_sha256"] and
                extra["screening_candidate_sha256"] == record_identity(record), "Parquet screening metadata mismatch")
        category_order.append(record["perception_category"])
        for path, pixel_hash in zip(record["images"], record["image_sha256"], strict=True):
            require(path not in image_hashes or image_hashes[path] == pixel_hash, "Contradictory image hashes")
            image_hashes[path] = pixel_hash
    require(dict(Counter(category_order)) == expected, "Wrong total category counts")
    block = {key: value // 5 for key, value in filtering.WEIGHTS.items()}
    for start in range(0, 3000, 20):
        require(dict(Counter(category_order[start:start + 20])) == block, f"Mixture drift at row {start}")
    if verify_images:
        for path, expected_hash in image_hashes.items():
            require(v2.pixel_hash(path) == expected_hash, f"Training image changed after screening: {path}")
    report = check_dataset(paths["parquet"], 3000)
    require(not report["errors"], "Dataset CPU checks failed: " + json.dumps(report["errors"]))
    return {**report, "category_counts": expected, "raw_groups_replayed": len(replayed),
            "screening_run_sha256": screen["run_sha256"],
            "screening_manifest_sha256": local.sha256(dataset_dir / "manifest.json"),
            "data_sha256": manifest["parquet_sha256"], "groups_sha256": manifest["groups_sha256"],
            "execution_provenance": execution_provenance,
            "model_files": spec["model_files"], "verified_image_files": len(image_hashes) if verify_images else 0}


def storage_requirement(checkpoint_bytes, export_bytes, completed=0, owned_bytes=0):
    peak = (checkpoint_bytes + export_bytes if completed >= 100 else
            max(2 * checkpoint_bytes, checkpoint_bytes + export_bytes))
    return max(0, peak - owned_bytes) + 3 * 1024**3


def load_storage_reference():
    """Validate captured sizing evidence without reopening retired HF weights."""
    reference = read_json(STORAGE_REFERENCE)
    require(reference.get("version") == "perception-v3-storage-reference-v1", "Invalid storage reference version")
    payload = {key: value for key, value in reference.items() if key != "payload_sha256"}
    require(digest(payload) == reference.get("payload_sha256"), "Storage reference payload hash mismatch")
    sources = reference["measurement_sources"]
    sizing_source = EXPERIMENT / "runtime/perception_v2_queue.json"
    require(sources["queue_path"] == "runtime/perception_v2_queue.json" and
            sources["queue_measurement_key"] == "storage_capacity_before_formal.measured" and
            local.sha256(sizing_source) == sources["queue_sha256"], "Storage reference queue metadata changed")
    for name in ("protocol", "resolved_config", "resource_usage"):
        suffix = ".yaml" if name == "resolved_config" else ".json"
        require(local.sha256(BASELINE / (name + suffix)) == sources["baseline_" + name + "_sha256"],
                "Storage reference baseline metadata changed: " + name)
    measured = read_json(sizing_source)["storage_capacity_before_formal"]["measured"]
    require(reference["checkpoint_measurement"] == measured and
            reference["checkpoint_bytes"] == measured["checkpoint_bytes"], "Storage checkpoint measurement mismatch")
    exported = reference["historical_export"]
    marker_text = exported["completion_marker_text"]
    require(hashlib.sha256(marker_text.encode()).hexdigest() == exported["completion_marker_sha256"] and
            json.loads(marker_text)["step"] == 100, "Historical export completion evidence changed")
    shards = exported["shards"]
    require(bool(shards) and len({item["name"] for item in shards}) == len(shards) and
            all(Path(item["name"]).name == item["name"] and item["name"].endswith(".safetensors") and
                type(item["size_bytes"]) is int and item["size_bytes"] > 0 for item in shards),
            "Invalid historical export sizing evidence")
    require(type(reference["checkpoint_bytes"]) is int and reference["checkpoint_bytes"] > 0 and
            type(reference["hf_export_bytes"]) is int and reference["hf_export_bytes"] > 0 and
            reference["hf_export_bytes"] == sum(item["size_bytes"] for item in shards),
            "Historical export byte count mismatch")
    return reference


def resource_check(run, completed):
    resource_path = run if run.exists() else run.parent
    filesystem = os.statvfs(resource_path)
    inode_report = {"total": filesystem.f_files, "free": filesystem.f_ffree,
                    "available": filesystem.f_favail, "required_available": 2048,
                    "enforced": filesystem.f_files > 0}
    if inode_report["enforced"]:
        require(filesystem.f_favail >= 2048,
                f"Insufficient filesystem inodes: need 2048 available, have {filesystem.f_favail}")
    reference = load_storage_reference()
    checkpoint_bytes = reference["checkpoint_bytes"]
    export_bytes = reference["hf_export_bytes"]
    owned_bytes = 0
    for directory in (run / "checkpoints", run / "hf"):
        require(not directory.is_symlink(), "Own checkpoint/export roots cannot be symlinks")
        for path in directory.rglob("*"):
            require(not path.is_symlink(), "Unexpected symlink in own checkpoint/export artifacts")
            if path.is_file():
                owned_bytes += path.stat().st_size
    required = storage_requirement(checkpoint_bytes, export_bytes, completed, owned_bytes)
    free = shutil.disk_usage(resource_path).free
    require(free >= required, f"Insufficient checkpoint capacity: need {required} free bytes, have {free}")
    cgroup = Path("/sys/fs/cgroup")
    limit_text = (cgroup / "memory.max").read_text().strip()
    current = int((cgroup / "memory.current").read_text())
    stats = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    working = current - int(stats.get("inactive_file", 0))
    previous_working_peak = read_json(BASELINE / "resource_usage.json")["working_set_peak_bytes"]
    if limit_text != "max":
        require(int(limit_text) - working >= previous_working_peak,
                "Insufficient cgroup memory headroom relative to the measured v2 working-set peak")
    return {"disk_free_bytes": free, "required_free_bytes": required, "checkpoint_bytes": checkpoint_bytes,
            "filesystem_inodes": inode_report,
            "export_bytes": export_bytes, "owned_checkpoint_export_bytes": owned_bytes,
            "checkpoint_sizing_source_sha256": reference["measurement_sources"]["queue_sha256"],
            "storage_reference_sha256": local.sha256(STORAGE_REFERENCE),
            "memory_limit_bytes": limit_text, "current_working_set_bytes": working,
            "historical_working_set_peak_bytes": previous_working_peak,
            "gpu_check": "performed immediately before formal subprocess"}


def validate_export(run):
    actor, export = run / "checkpoints/global_step_100/actor", run / "hf/global_step_100"
    require((actor / "fsdp_config.json").is_file(), "Missing final FSDP checkpoint")
    for kind in ("model", "optim"):
        for rank in (0, 1):
            path = actor / f"{kind}_world_size_2_rank_{rank}.pt"
            require(path.is_file() and path.stat().st_size > 0, "Missing or empty final checkpoint shard")
    require(read_json(export / "sharedexpert_export_complete.json")["step"] == 100, "Wrong final export step")
    require((export / "config.json").is_file(), "Missing final HF config")
    shards = list(export.glob("*.safetensors"))
    require(shards and all(path.stat().st_size > 0 for path in shards), "Missing final HF weights")
    index = export / "model.safetensors.index.json"
    if index.exists():
        require(set(read_json(index)["weight_map"].values()) == {path.name for path in shards},
                "HF weight index differs from exported shards")
    return export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    mode = parser.add_mutually_exclusive_group(required=True)
    for action in ("check", "dry-run", "execute", "merge-only"):
        mode.add_argument("--" + action, action="store_true")
    args = parser.parse_args()
    gpus = args.gpus.split(",")
    protocol, teacher, baseline = load_protocol(gpus, args.dataset_dir, args.model)
    config = compose_and_verify(protocol, teacher, gpus)
    if args.dry_run:
        print(json.dumps({"teacher": teacher, "steps": 100, "batch_size": 30, "group_size": 8,
                          "gpus": gpus, "run_dir": str(RUN_DIR), "config_parity": "passed",
                          "separate_gpu_smoke": False, "overrides": make_overrides(protocol, teacher, gpus)}, indent=2))
        return
    env = dict(os.environ)
    # The frozen 2-GPU recipe used single-threaded CPU ops. On the authorized
    # single-GPU path the actor update runs through FSDP2 CPU offload, so the
    # optimizer/transfer work needs real CPU parallelism; the math is unchanged.
    cpu_threads = "8" if len(gpus) == 1 else "1"
    env.update({"PYTHONPATH": str(WORKSPACE) + ":" + str(WORKSPACE / protocol["repository"]),
                "CUDA_VISIBLE_DEVICES": ",".join(gpus), "PYTHONUNBUFFERED": "1", "VLLM_USE_V1": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": cpu_threads, "MKL_NUM_THREADS": cpu_threads,
                "OPENBLAS_NUM_THREADS": cpu_threads, "WANDB_MODE": "disabled", "TENSORBOARD_DIR": str(RUN_DIR / "tensorboard")})
    os.environ.update(env)
    identity, patch = recipe_identity(protocol, baseline)
    from preflight import check_imports
    report = check_imports()
    require(not report["errors"], "Dependency CPU checks failed: " + json.dumps(report["errors"]))
    from experiments.sharedexpert_c import rewards
    rewards.validate_runtime()
    report["dataset"] = validate_dataset(args.dataset_dir, protocol)
    latest = RUN_DIR / "checkpoints/latest_checkpointed_iteration.txt"
    completed = int(latest.read_text().strip()) if latest.exists() else 0
    require(0 <= completed <= 100, "Checkpoint step exceeds frozen endpoint")
    require(not args.merge_only or completed == 100, "Merge-only requires completed step100")
    report["resources"] = resource_check(RUN_DIR, completed)
    report["gpu_smoke_performed"] = False
    if args.check:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    from omegaconf import OmegaConf
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha256(str(RUN_DIR).encode()).hexdigest()[:24]
    with ExitStack() as stack:
        lock = stack.enter_context(open(f"/tmp/sharedexpert-train-{lock_key}.lock", "a"))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        frozen = {"protocol": protocol, "teacher": teacher, "gpus": gpus, "smoke": False,
                  "actual_steps": 100, "actual_batch_size": 30,
                  "rollout_replica_seeds": {str(rank): protocol["seed"] + rank for rank in range(2)},
                  "data_sha256": report["dataset"]["data_sha256"],
                  "screening_manifest_sha256": report["dataset"]["screening_manifest_sha256"],
                  "screening_groups_sha256": report["dataset"]["groups_sha256"],
                  "screening_execution_provenance": report["dataset"]["execution_provenance"],
                  "model_files": report["dataset"]["model_files"],
                  "dependency_versions": report["versions"], **identity}
        frozen_path = RUN_DIR / "protocol.json"
        require(not frozen_path.exists() or read_json(frozen_path) == frozen,
                "Frozen training inputs changed; refusing silent resume drift")
        write_json(frozen_path, frozen)
        resolved = OmegaConf.to_yaml(config, resolve=True)
        config_path = RUN_DIR / "resolved_config.yaml"
        require(not config_path.exists() or config_path.read_text() == resolved,
                "Resolved training configuration changed on resume")
        config_path.write_text(resolved)
        (RUN_DIR / "repository.patch").write_text(patch)
        report["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_json(RUN_DIR / "preflight.json", report)
        with local.monitor_cgroup(RUN_DIR):
            if completed < 100:
                for gpu in gpus:
                    gpu_lock = stack.enter_context(open(f"/tmp/sharedexpert-perception-v3-gpu-{gpu}.lock", "a"))
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                report["gpu_snapshot"] = local.gpu_check(gpus)
                write_json(RUN_DIR / "preflight.json", report)
                with (RUN_DIR / "training.log").open("a", buffering=1) as log:
                    log.write(f"\nFormal launch/resume at step {completed}; endpoint100; no separate GPU smoke\n")
                    subprocess.run([sys.executable, str(TRAINING / "run.py"), "--config", str(config_path)],
                                   cwd=WORKSPACE, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            local.merge_checkpoint(RUN_DIR, 100)
        exported = validate_export(RUN_DIR)
        write_json(RUN_DIR / "completion.json", {"status": "complete", "step": 100,
                                                  "hf_model": str(exported), "protocol_sha256": local.sha256(frozen_path)})
        print(json.dumps({"teacher": "perception_v3", "step": 100, "hf_model": str(exported)}))


if __name__ == "__main__":
    main()
