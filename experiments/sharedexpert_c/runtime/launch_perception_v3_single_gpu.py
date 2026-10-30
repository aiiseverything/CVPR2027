#!/usr/bin/env python3
"""User-authorized single-GPU memory-optimized launcher for perception_v3.

The frozen ``training/launch_perception_v3.py`` and ``training/launch_local.py``
remain byte-identical. This recovery entry point keeps the scientific recipe
(3,000 screened rows, batch 30, group 8, 100 steps, LR 1e-6, thinking, same
model/rewards) and changes only the execution topology and memory knobs that the
user explicitly authorized on 2026-10-29:

* trainer.n_gpus_per_node: 2 -> 1
* rollout.gpu_memory_utilization: 0.7 -> 0.55
* rollout.max_num_seqs: 120 -> 64
* rollout.enforce_eager: false -> true
* data.dataloader_num_workers: 4 -> 2
* rollout.agent.num_workers: 4 -> 2
* ray_init.num_cpus: 24 -> 12

The v2 119 GiB working-set gate cannot pass in this container's 100 GiB cgroup,
so the preflight substitutes a documented single-GPU incremental budget
(78 GiB) plus a live cgroup/GPU watchdog that terminates the run after three
consecutive over-limit samples.

Evidence written by this entry point:
  <run>/topology_adaptation.json   authorization + overrides + thresholds
  <run>/memory_watchdog.jsonl      samples (on normal completion)
  <run>/memory_watchdog.json       breach receipt (only on abort)
"""
from __future__ import annotations

import contextlib
import datetime as dt
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import threading
import time

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
WORKSPACE = EXPERIMENT.parents[1]
sys.path[:0] = [str(EXPERIMENT / "training"), str(EXPERIMENT / "data_pipeline"),
                str(WORKSPACE), str(WORKSPACE / "pre-lab/repo")]
import launch_perception_v3 as la
import validate_perception_v3_dataset_single_gpu as single

GiB = 1024 ** 3
SINGLE_GPU_OVERRIDES = {
    # Rollout/actor are sequential and free_cache_engine releases vLLM KV before
    # the actor update. Offloading both optimizer states and params to CPU lets
    # the rollout own a larger KV cache (0.82) while the actor update runs with
    # a separate GPU peak.
    "actor_rollout_ref.actor.fsdp_config.param_offload": True,
    # FSDP2 uses offload_policy (CPUOffloadPolicy); optimizer_offload/param_offload are legacy FSDP1 knobs.
    "actor_rollout_ref.actor.fsdp_config.offload_policy": True,
    "actor_rollout_ref.rollout.gpu_memory_utilization": 0.82,
    "actor_rollout_ref.rollout.max_num_seqs": 120,
    "actor_rollout_ref.rollout.enforce_eager": False,
    "data.dataloader_num_workers": 2,
    "actor_rollout_ref.rollout.agent.num_workers": 2,
    "ray_kwargs.ray_init.num_cpus": 12,
}
# v2 measured 119 GiB with two GPU worker stacks. Single-GPU keeps one vLLM
# engine/dataloader/agent pool; 78 GiB is the documented incremental budget.
SINGLE_GPU_ANON_BUDGET = 80 * GiB
GPU_WATCHDOG_THRESHOLD = 76 * GiB
MEMORY_WATCHDOG_RESERVE = 8 * GiB
SAMPLES_BEFORE_ABORT = 3
SAMPLE_SECONDS = 5
ACTIVE_GPUS: list[str] = []


def load_protocol(gpus, dataset_dir=la.DEFAULT_DATASET, model=la.DEFAULT_MODEL):
    baseline = la.read_json(la.BASELINE / "protocol.json")
    common = la.read_json(la.TRAINING / "config/common_local_100.json")
    la.require(common == baseline["protocol"], "Common training protocol changed since perception_v2")
    la.require(gpus in (["0"], ["0", "1"]) and (gpus == ["0"] or gpus == baseline["gpus"]),
               "Formal training requires the original GPUs 0,1 or the authorized single GPU 0")
    la.require((common["steps"], common["batch_size"], common["group_size"]) == (100, 30, 8),
               "Formal budget must remain 100 steps, batch30, eight completions")
    model = Path(model).resolve()
    la.require(model.is_dir() and model.name == Path(common["model"]).name,
               "The relocated original Qwen3.5-4B model is required")
    protocol = {**common, "model": str(model)}
    teacher = la.read_json(la.TRAINING / "config/perception_v3.json")
    la.require(teacher["teacher"] == "perception_v3" and teacher["category_weights"] ==
               {key: weight / 100 for key, weight in la.filtering.WEIGHTS.items()},
               "Wrong perception_v3 mixture")
    teacher["data_file"] = str(Path(dataset_dir).resolve() / "perception_v3.parquet")
    return protocol, teacher, baseline


def make_overrides(protocol, teacher, gpus):
    values = la.local.make_overrides(protocol, teacher, len(gpus), la.RUN_DIR, 100, 30)
    if len(gpus) != 1:
        return values
    mapping = {}
    for item in values:
        key, value = item.split("=", 1)
        mapping[key] = value
    for key, value in SINGLE_GPU_OVERRIDES.items():
        mapping[key] = json.dumps(value, ensure_ascii=False)
    return [f"{key}={value}" for key, value in mapping.items()]


def compose_and_verify(protocol, teacher, gpus):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    with initialize_config_dir(config_dir=str(WORKSPACE / protocol["repository"] / "verl/trainer/config"),
                               version_base=None):
        config = compose(config_name="ppo_trainer", overrides=make_overrides(protocol, teacher, gpus))
    original = la.read_json(la.BASELINE / "protocol.json")["protocol"]
    expected = OmegaConf.create(la.relocate_baseline_paths(
        OmegaConf.to_container(OmegaConf.load(la.BASELINE / "resolved_config.yaml"), resolve=True),
        original["model"], protocol["model"]))
    for key, value in {
        "data.train_files": [teacher["data_file"]], "trainer.experiment_name": la.RUN_DIR.name,
        "trainer.default_local_dir": str(la.RUN_DIR / "checkpoints"),
        "trainer.rollout_data_dir": str(la.RUN_DIR / "rollouts"),
    }.items():
        OmegaConf.update(expected, key, value)
    OmegaConf.update(expected, "trainer.n_gpus_per_node", len(gpus))
    if len(gpus) == 1:
        for key, value in SINGLE_GPU_OVERRIDES.items():
            OmegaConf.update(expected, key, value)
    actual_tree, expected_tree = (OmegaConf.to_container(value, resolve=True) for value in (config, expected))
    la.require(actual_tree == expected_tree,
               "Resolved configuration differs from perception_v2 beyond dataset/run/model/workspace/topology paths")
    return config


def resource_check(run, completed):
    resource_path = run if run.exists() else run.parent
    filesystem = os.statvfs(resource_path)
    inode_report = {"total": filesystem.f_files, "free": filesystem.f_ffree,
                    "available": filesystem.f_favail, "required_available": 2048,
                    "enforced": filesystem.f_files > 0}
    if inode_report["enforced"]:
        la.require(filesystem.f_favail >= 2048,
                   f"Insufficient filesystem inodes: need 2048 available, have {filesystem.f_favail}")
    reference = la.load_storage_reference()
    checkpoint_bytes = reference["checkpoint_bytes"]
    export_bytes = reference["hf_export_bytes"]
    owned_bytes = 0
    for directory in (run / "checkpoints", run / "hf"):
        la.require(not directory.is_symlink(), "Own checkpoint/export roots cannot be symlinks")
        for path in directory.rglob("*"):
            la.require(not path.is_symlink(), "Unexpected symlink in own checkpoint/export artifacts")
            if path.is_file():
                owned_bytes += path.stat().st_size
    required = la.storage_requirement(checkpoint_bytes, export_bytes, completed, owned_bytes)
    free = shutil.disk_usage(resource_path).free
    la.require(free >= required, f"Insufficient checkpoint capacity: need {required} free bytes, have {free}")
    cgroup = Path("/sys/fs/cgroup")
    limit_text = (cgroup / "memory.max").read_text().strip()
    current = int((cgroup / "memory.current").read_text())
    stats = dict(line.split() for line in (cgroup / "memory.stat").read_text().splitlines())
    working = current - int(stats.get("inactive_file", 0))
    common = {"disk_free_bytes": free, "required_free_bytes": required, "checkpoint_bytes": checkpoint_bytes,
              "filesystem_inodes": inode_report, "export_bytes": export_bytes,
              "owned_checkpoint_export_bytes": owned_bytes,
              "checkpoint_sizing_source_sha256": reference["measurement_sources"]["queue_sha256"],
              "storage_reference_sha256": la.local.sha256(la.STORAGE_REFERENCE),
              "memory_limit_bytes": limit_text, "current_working_set_bytes": working,
              "gpu_check": "performed immediately before formal subprocess"}
    if len(ACTIVE_GPUS) == 1:
        la.require(limit_text != "max", "A finite cgroup memory limit is required for single-GPU training")
        limit = int(limit_text)
        la.require(limit >= 95 * GiB,
                   f"Single-GPU training requires at least 95 GiB cgroup limit, have {limit}")
        headroom = limit - working
        anonymous = int(stats.get("anon", 0)) + int(stats.get("shmem", 0))
        page_cache = int(stats.get("file", 0))
        headroom = limit - anonymous
        la.require(headroom >= SINGLE_GPU_ANON_BUDGET,
                   f"Insufficient single-GPU anonymous memory headroom: need {SINGLE_GPU_ANON_BUDGET} bytes, "
                   f"have {headroom} bytes (limit {limit}, anon {anonymous}, working {working}, page_cache {page_cache})")
        threshold = min(limit - MEMORY_WATCHDOG_RESERVE, 92 * GiB)
        return {**common, "single_gpu_anon_budget_bytes": SINGLE_GPU_ANON_BUDGET,
                "anonymous_bytes": anonymous, "page_cache_bytes": page_cache,
                "working_set_bytes": working, "memory_budget_basis": "anonymous+shmem",
                "memory_headroom_bytes": headroom,
                "single_gpu_watchdog": {"memory_threshold_bytes": threshold,
                                        "gpu_threshold_bytes": GPU_WATCHDOG_THRESHOLD,
                                        "sample_seconds": SAMPLE_SECONDS,
                                        "consecutive_samples": SAMPLES_BEFORE_ABORT,
                                        "action": "SIGTERM to the launcher process group after the threshold is exceeded"},
                "historical_two_gpu_working_set_peak_bytes": la.read_json(
                    la.BASELINE / "resource_usage.json")["working_set_peak_bytes"]}
    previous_working_peak = la.read_json(la.BASELINE / "resource_usage.json")["working_set_peak_bytes"]
    if limit_text != "max":
        la.require(int(limit_text) - working >= previous_working_peak,
                   "Insufficient cgroup memory headroom relative to the measured v2 working-set peak")
    return {**common, "historical_working_set_peak_bytes": previous_working_peak}


def validate_export(run):
    frozen = la.read_json(run / "protocol.json") if (run / "protocol.json").is_file() else {"gpus": ["0", "1"]}
    world_size = len(frozen["gpus"])
    actor, export = run / "checkpoints/global_step_100/actor", run / "hf/global_step_100"
    la.require((actor / "fsdp_config.json").is_file(), "Missing final FSDP checkpoint")
    for kind in ("model", "optim"):
        for rank in range(world_size):
            path = actor / f"{kind}_world_size_{world_size}_rank_{rank}.pt"
            la.require(path.is_file() and path.stat().st_size > 0, "Missing or empty final checkpoint shard")
    la.require(la.read_json(export / "sharedexpert_export_complete.json")["step"] == 100, "Wrong final export step")
    la.require((export / "config.json").is_file(), "Missing final HF config")
    shards = list(export.glob("*.safetensors"))
    la.require(shards and all(path.stat().st_size > 0 for path in shards), "Missing final HF weights")
    index = export / "model.safetensors.index.json"
    if index.exists():
        la.require(set(la.read_json(index)["weight_map"].values()) == {path.name for path in shards},
                   "HF weight index differs from exported shards")
    return export


def _gpu_used_bytes():
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        return int(pynvml.nvmlDeviceGetMemoryInfo(handle).used)
    except Exception:
        return None


_ORIGINAL_MONITOR = la.local.monitor_cgroup


@contextlib.contextmanager
def monitored_cgroup(run_dir):
    run_dir = Path(run_dir)
    with _ORIGINAL_MONITOR(run_dir) as report:
        stop = threading.Event()
        samples = []
        limit = int((Path("/sys/fs/cgroup/memory.max")).read_text().strip())
        memory_threshold = min(limit - MEMORY_WATCHDOG_RESERVE, 92 * GiB)

        def watch():
            breaches = 0
            while not stop.wait(SAMPLE_SECONDS):
                try:
                    current = int((Path("/sys/fs/cgroup/memory.current")).read_text())
                    stats = dict(line.split() for line in
                                 (Path("/sys/fs/cgroup/memory.stat")).read_text().splitlines())
                    working = current - int(stats.get("inactive_file", 0))
                    anonymous = int(stats.get("anon", 0)) + int(stats.get("shmem", 0))
                    page_cache = int(stats.get("file", 0))
                    gpu_used = _gpu_used_bytes()
                    sample = {"time": dt.datetime.now(dt.timezone.utc).isoformat(),
                              "anonymous_bytes": anonymous, "working_set_bytes": working,
                              "page_cache_bytes": page_cache, "gpu_used_bytes": gpu_used}
                    samples.append(sample)
                    over = anonymous > memory_threshold or (gpu_used is not None and gpu_used > GPU_WATCHDOG_THRESHOLD)
                    breaches = breaches + 1 if over else 0
                    if breaches >= SAMPLES_BEFORE_ABORT:
                        receipt = {"schema": 1, "status": "aborted_memory_watchdog",
                                   "time": sample["time"], "anonymous_bytes": anonymous,
                                   "working_set_bytes": working, "page_cache_bytes": page_cache,
                                   "memory_threshold_bytes": memory_threshold,
                                   "gpu_used_bytes": gpu_used,
                                   "gpu_threshold_bytes": GPU_WATCHDOG_THRESHOLD,
                                   "consecutive_samples": breaches,
                                   "samples": samples[-10:]}
                        (run_dir / "memory_watchdog.json").write_text(
                            json.dumps(receipt, indent=2) + "\n")
                        os.killpg(os.getpgrp(), signal.SIGTERM)
                        return
                except Exception as exc:
                    samples.append({"time": dt.datetime.now(dt.timezone.utc).isoformat(),
                                    "sampling_error": f"{type(exc).__name__}: {exc}"})

        thread = threading.Thread(target=watch, daemon=True)
        thread.start()
        try:
            yield report
        finally:
            stop.set()
            thread.join(timeout=10)
            if samples:
                with (run_dir / "memory_watchdog.jsonl").open("a") as stream:
                    stream.write("".join(json.dumps(sample, ensure_ascii=False) + "\n" for sample in samples))
            report["single_gpu_watchdog"] = {
                "memory_threshold_bytes": memory_threshold,
                "gpu_threshold_bytes": GPU_WATCHDOG_THRESHOLD,
                "sample_seconds": SAMPLE_SECONDS,
                "consecutive_samples": SAMPLES_BEFORE_ABORT,
                "samples": len(samples),
            }


def write_topology_receipt(gpus):
    if len(gpus) != 1:
        return
    la.RUN_DIR.mkdir(parents=True, exist_ok=True)
    dataset_manifest = la.DEFAULT_DATASET / "manifest.json"
    receipt = {
        "schema": 1,
        "mode": "single_gpu_memory_optimized",
        "devices": gpus,
        "authorized_by": "user",
        "authorized_at": "2026-10-29",
        "screening_topology": "single_gpu_recovery",
        "recipe_invariants": {"dataset_records": 3000, "steps": 100, "batch_size": 30,
                              "group_size": 8, "learning_rate": 0.000001, "thinking": True,
                              "data_manifest_sha256": la.local.sha256(dataset_manifest)},
        "memory_overrides": SINGLE_GPU_OVERRIDES,
        "single_gpu_anon_budget_bytes": SINGLE_GPU_ANON_BUDGET,
        "watchdog": {"memory_threshold_max_bytes": 92 * GiB,
                     "memory_reserve_bytes": 8 * GiB,
                     "gpu_threshold_bytes": GPU_WATCHDOG_THRESHOLD,
                     "sample_seconds": SAMPLE_SECONDS,
                     "consecutive_samples": SAMPLES_BEFORE_ABORT},
        "frozen_launcher_sha256": la.local.sha256(EXPERIMENT / "training/launch_perception_v3.py"),
        "frozen_shared_trainer_sha256": {name: la.local.sha256(EXPERIMENT / "training" / name)
                                         for name in la.SHARED_FILES},
        "recovery_launcher_sha256": la.local.sha256(Path(__file__)),
        "note": "Batch/group/steps/LR/data/model/rewards unchanged; only distributed topology and rollout/worker memory knobs were adapted.",
    }
    path = la.RUN_DIR / "topology_adaptation.json"
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)


def main():
    global ACTIVE_GPUS
    argv = sys.argv[1:]
    gpus = []
    for index, value in enumerate(argv):
        if value == "--gpus" and index + 1 < len(argv):
            gpus = argv[index + 1].split(",")
        elif value.startswith("--gpus="):
            gpus = value.split("=", 1)[1].split(",")
    la.require(gpus in (["0"], ["0", "1"]),
               "This recovery launcher supports GPU 0 or the original GPU 0,1")
    ACTIVE_GPUS = gpus
    if len(gpus) == 1:
        write_topology_receipt(gpus)
    la.load_protocol = load_protocol
    la.make_overrides = make_overrides
    la.compose_and_verify = compose_and_verify
    la.resource_check = resource_check
    la.validate_export = validate_export
    la.validate_execution_provenance = single.validate_execution_provenance_recovery
    la.local.monitor_cgroup = monitored_cgroup
    la.main()


if __name__ == "__main__":
    main()
