#!/usr/bin/env python3
"""Independently replay the published perception_v3 dataset on this one-GPU host.

The frozen training launcher validates execution provenance with a hard-coded
two-GPU topology and requires every historical execution receipt to be
terminal. This recovery run was executed on one H800 and one older receipt was
interrupted mid-generation without a close() finalization. This tool keeps the
launcher's dataset replay intact and substitutes only the execution-topology
gate with an equivalent check that:

* accepts one or two distinct single-GPU replicas;
* still verifies every receipt's frozen core hash, run hash, archived source,
  process identity, journal boundaries and prefix hashes;
* treats a ``running`` receipt as valid only when its parent process is dead
  (an interrupted, unclosed generation), and records it explicitly.

It never edits the frozen training launcher, receipts, journal or dataset.
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(EXPERIMENT / "training"), str(EXPERIMENT / "data_pipeline"),
                str(EXPERIMENT.parents[1]), str(EXPERIMENT.parents[1] / "pre-lab/repo")]
import launch_perception_v3 as la


def process_alive(pid, ticks):
    try:
        fields = (Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] not in {"Z", "X"} and int(fields[19]) == int(ticks)
    except (OSError, ValueError, IndexError):
        return False


def validate_execution_provenance_recovery(root, screen):
    """Topology-aware equivalent of launch_perception_v3.validate_execution_provenance."""
    root = Path(root)
    directory = root / "executions"
    entries = sorted(directory.glob("parallel-*.json"))
    legacy = directory / "legacy-before-parallel.groups.jsonl"
    if not entries:
        la.require(not legacy.exists(), "Parallel transition snapshot has no execution receipt")
        return {"backend": "original_single_gpu", "execution_receipts": {}}
    la.require(legacy.is_file(), "Missing preserved single-GPU journal snapshot")
    journal = root / "groups.jsonl"
    core_hash = la.local.sha256(la.PIPELINE / "filter_perception_v3.py")
    parallel_hash = la.local.sha256(la.PIPELINE / "filter_perception_v3_parallel.py")
    la.require(screen["spec"].get("driver_source_sha256") == core_hash,
               "Parallel execution core differs from frozen screening run")
    records, lengths = [], [legacy.stat().st_size]
    interrupted = []
    allowed_topologies = ([0, 1], [0])
    for path in entries:
        item = la.read_json(path)
        devices_expected = item.get("devices")
        la.require(item.get("schema") == 1 and item.get("backend") == "independent_gpu_replicas" and
                   devices_expected in allowed_topologies and
                   item.get("tensor_parallel_size_per_worker") == 1,
                   f"Invalid parallel execution topology: {path.name}")
        la.require(item.get("run_sha256") == screen["run_sha256"] and
                   item.get("core_driver_sha256") == core_hash,
                   "Parallel execution changed the frozen screening core or run")
        archived_hash = item.get("parallel_driver_sha256", "")
        la.require(len(archived_hash) == 64 and all(char in "0123456789abcdef" for char in archived_hash),
                   "Invalid parallel driver hash")
        source = Path("sources") / f"parallel-{archived_hash}.py"
        la.require(item.get("parallel_driver_source") == str(source) and
                   (directory / source).is_file() and
                   la.local.sha256(directory / source) == archived_hash,
                   "Archived parallel execution source changed")
        status = item.get("status")
        la.require(status in {"complete", "failed", "running"},
                   "Parallel execution has no recognised terminal/interrupted state")
        la.require(type(item.get("pid")) is int and item["pid"] > 0 and
                   type(item.get("start_ticks")) is int and item["start_ticks"] >= 0,
                   "Invalid parallel parent process identity")
        la.require(type(item.get("batch_prompts")) is int and item["batch_prompts"] > 0,
                   "Invalid parallel execution batch size")
        try:
            started = dt.datetime.fromisoformat(item["started_at"])
        except (ValueError, TypeError, KeyError) as exc:
            raise RuntimeError("Invalid parallel execution start time") from exc
        la.require(started.tzinfo is not None,
                   "Parallel execution start time requires an explicit timezone")
        for key in ("initial_groups_bytes", "initial_group_count"):
            la.require(type(item.get(key)) is int and item[key] >= 0,
                       "Invalid initial execution journal boundary")
        lengths.append(item["initial_groups_bytes"])
        final_keys = {"final_groups_bytes", "final_group_count", "final_groups_sha256"}
        has_final = bool(final_keys.intersection(item))
        la.require(not has_final or final_keys <= item.keys(),
                   "Incomplete final execution journal evidence")
        la.require(status != "complete" or has_final,
                   "Completed execution has no final journal evidence")
        if has_final:
            la.require(type(item["final_groups_bytes"]) is int and
                       item["final_groups_bytes"] >= item["initial_groups_bytes"] and
                       type(item["final_group_count"]) is int and
                       item["final_group_count"] >= item["initial_group_count"],
                       "Execution journal shrank or final boundary is invalid")
            lengths.append(item["final_groups_bytes"])
        if status == "running":
            la.require(not process_alive(item["pid"], item["start_ticks"]),
                       f"Receipt {path.name} is still running with a live parent process")
            interrupted.append(path.name)
        workers = item.get("workers")
        la.require(isinstance(workers, list), "Missing parallel worker process evidence")
        devices = []
        for worker in workers:
            la.require(type(worker.get("gpu")) is int and worker["gpu"] in devices_expected and
                       type(worker.get("pid")) is int and worker["pid"] > 0 and
                       type(worker.get("start_ticks")) is int and worker["start_ticks"] >= 0,
                       "Invalid parallel worker process identity")
            devices.append(worker["gpu"])
        la.require(len(set(devices)) == len(devices), "Duplicate parallel worker GPU")
        if status == "complete" and item["final_group_count"] > item["initial_group_count"]:
            la.require(sorted(devices) == sorted(devices_expected),
                       "Completed parallel generation lacks its declared worker identities")
        records.append((path, item, started))
    latest_path, latest, _ = max(records, key=lambda entry: (entry[2], entry[0].name))
    la.require(latest["parallel_driver_sha256"] == parallel_hash,
               "Latest parallel execution source differs from current wrapper")
    prefixes = la.journal_prefixes(journal, lengths)
    legacy_size, legacy_hash = legacy.stat().st_size, la.local.sha256(legacy)
    la.require(prefixes[legacy_size]["sha256"] == legacy_hash,
               "Preserved single-GPU groups changed in the current journal")
    la.require(any(item["initial_groups_bytes"] == legacy_size and
                   item["initial_groups_sha256"] == legacy_hash and
                   item["initial_group_count"] == prefixes[legacy_size]["count"]
                   for _, item, _ in records),
               "No execution receipt binds the preserved single-GPU transition")
    for _, item, _ in records:
        for stage in ("initial", "final"):
            key = stage + "_groups_bytes"
            if key not in item:
                continue
            prefix = prefixes[item[key]]
            la.require(prefix["sha256"] == item[stage + "_groups_sha256"] and
                       prefix["count"] == item[stage + "_group_count"],
                       "Execution journal prefix hash or group count changed")
    journal_size = journal.stat().st_size
    la.require(any(item["status"] == "complete" and item["final_groups_bytes"] == journal_size
                   for _, item, _ in records),
               "Final screening journal is not covered by a completed parallel execution")
    return {
        "backend": "independent_gpu_replicas",
        "devices": latest["devices"],
        "topology_adaptation": "single_gpu_recovery" if latest["devices"] == [0] else "original_two_gpu",
        "interrupted_unclosed_receipts": interrupted,
        "execution_receipts": {path.name: la.local.sha256(path) for path, _, _ in records},
        "legacy_groups_sha256": legacy_hash,
        "legacy_group_count": prefixes[legacy_size]["count"],
        "legacy_groups_bytes": legacy_size,
        "current_parallel_driver_sha256": parallel_hash,
        "latest_execution": latest_path.name,
    }


def main():
    protocol = la.read_json(EXPERIMENT / "training/config/common_local_100.json")
    # The protocol records the old server's model path; the relocated model is
    # the launcher's DEFAULT_MODEL (same basename and frozen hashes).
    protocol["model"] = str(la.DEFAULT_MODEL.resolve())
    original = la.validate_execution_provenance
    la.validate_execution_provenance = validate_execution_provenance_recovery
    started = time.time()
    try:
        report = la.validate_dataset(la.DEFAULT_DATASET, protocol)
    finally:
        la.validate_execution_provenance = original
    report["recovery_validation"] = {
        "validated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "elapsed_seconds": round(time.time() - started, 1),
        "dataset_dir": str(la.DEFAULT_DATASET),
        "frozen_training_launcher_sha256": la.local.sha256(EXPERIMENT / "training/launch_perception_v3.py"),
        "note": "Dataset replay is the frozen launcher's own validate_dataset; only the execution-topology "
                "gate was substituted for this one-GPU recovery host.",
    }
    out = EXPERIMENT / "runtime/perception_v3_dataset_validation.json"
    temporary = out.with_name(out.name + ".writing")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(out)
    print(json.dumps({key: report[key] for key in
                      ("category_counts", "raw_groups_replayed", "verified_image_files", "errors",
                       "screening_run_sha256", "data_sha256", "screening_manifest_sha256")}, indent=2))
    print(json.dumps(report["execution_provenance"], indent=2))
    print(json.dumps(report["recovery_validation"], indent=2))


if __name__ == "__main__":
    main()
