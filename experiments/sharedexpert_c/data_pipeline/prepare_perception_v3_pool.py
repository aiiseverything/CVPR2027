#!/usr/bin/env python3
"""Export original-annotation candidates for the perception v3 difficulty filter.

This CPU-only step preserves the v2 category and leakage rules. Its output is a
candidate pool, NOT a filtered training set: every final training record still
requires eight frozen-model completions with two through six correct answers.
The v2 artifacts and inputs are never rewritten, and no smoke dataset is made.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import sqlite3
import struct
import tempfile

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
SPEC = importlib.util.spec_from_file_location("perception_v3_original_rules", ROOT / "prepare_perception_v2.py")
ORIGINAL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ORIGINAL)
VERSION = "perception-v3-original-candidate-pool-v1"


def ordered_candidates(pools: dict[str, list[dict]], seed: int) -> list[dict]:
    """Visit all distinct candidates, using the target ratio while pools last."""
    if set(pools) != set(ORIGINAL.WEIGHTS):
        raise ValueError("Candidate categories do not match the original six categories")
    decks = {}
    for category in ORIGINAL.WEIGHTS:
        deck = sorted(pools[category], key=lambda row: hashlib.sha256(
            f"{seed}:{category}:{row['id']}".encode()).hexdigest())
        decks[category] = deck
    offsets = Counter()
    records = []
    rng = random.Random(seed)
    while any(offsets[key] < len(decks[key]) for key in decks):
        labels = [key for key, weight in ORIGINAL.WEIGHTS.items() for _ in range(weight // 5)]
        rng.shuffle(labels)
        for key in labels:
            if offsets[key] < len(decks[key]):
                records.append(decks[key][offsets[key]])
                offsets[key] += 1
    ids = [record["id"] for record in records]
    pairs = [(tuple(record["image_sha256"]), record["question_sha256"]) for record in records]
    if len(set(ids)) != len(ids) or len(set(pairs)) != len(pairs):
        raise ValueError("Repeated candidate record or image/question pair")
    return records


def make_pools(blocks: dict, extra_paths: list[Path]) -> tuple[dict, dict]:
    """Apply the original v2 eligibility rules across original and added rows."""
    paths = [ROOT / "normalized" / (source + ".jsonl") for source in ORIGINAL.SOURCES]
    paths.extend(extra_paths)
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("A source file was supplied more than once")
    inputs = []
    records_by_source = defaultdict(list)
    identity_payloads = {}
    for path in paths:
        records = ORIGINAL.read_jsonl(path)
        inputs.append({"path": str(path.resolve()), "sha256": ORIGINAL.sha256(path), "records": len(records)})
        for record in records:
            if record["source"] not in ORIGINAL.SOURCES:
                raise ValueError("Supplemental record is not from an original source: " + record["source"])
            payload = {key: record.get(key) for key in (
                "source", "source_id", "split", "task", "question_sha256", "image_sha256", "answers", "bbox")}
            if record["id"] in identity_payloads and identity_payloads[record["id"]] != payload:
                raise ValueError("Conflicting labels or identity for duplicate source record: " + record["id"])
            identity_payloads[record["id"]] = payload
            records_by_source[record["source"]].append(record)
    pools = {key: [] for key in ORIGINAL.WEIGHTS}
    counts = {}
    seen_pairs, seen_ids = set(), set()
    for source in ORIGINAL.SOURCES:
        records = records_by_source[source]
        stats = Counter(input_records=len(records))
        labels = defaultdict(set)
        if source == "refcoco":
            for record in records:
                labels[(tuple(record["image_sha256"]), record["question_sha256"])].add(
                    tuple(round(value, 6) for value in record["bbox"]))
        ambiguous = {key for key, values in labels.items() if len(values) > 1}
        stats["ambiguous_bbox_pairs"] = len(ambiguous)
        for record in records:
            if record["split"] != "train":
                raise ValueError("Non-train record: " + record["id"])
            if ORIGINAL.question_hash(record["question"]) != record["question_sha256"]:
                raise ValueError("Question hash mismatch: " + record["id"])
            pair = tuple(record["image_sha256"]), record["question_sha256"]
            if blocks["images"].intersection(record["image_sha256"]):
                stats["evaluation_image_removed"] += 1
                continue
            if record["question_sha256"] in blocks["questions"]:
                stats["evaluation_question_removed"] += 1
                continue
            if record["id"] in blocks["ids"] or (source, str(record["source_id"])) in blocks["source_ids"]:
                stats["evaluation_record_id_removed"] += 1
                continue
            if record.get("image_id") and int(record["image_id"]) in blocks["coco_ids"]:
                stats["evaluation_coco_image_id_removed"] += 1
                continue
            if pair in ambiguous:
                stats["ambiguous_bbox_removed"] += 1
                continue
            routed = ORIGINAL.classify(record)
            if routed is None:
                stats["outside_category_rules_removed"] += 1
                continue
            if pair in seen_pairs or record["id"] in seen_ids:
                stats["duplicate_record_or_image_question_removed"] += 1
                continue
            seen_pairs.add(pair)
            seen_ids.add(record["id"])
            category, rule = routed
            pools[category].append({**record, "perception_category": category, "selection_rule": rule})
            stats["eligible"] += 1
        counts[source] = dict(stats)
    return pools, {"input_files": inputs, "counts": counts}


def add_nested_image_blocks(blocks: dict, suites: list[Path], pools: dict, workers: int = 4) -> dict:
    """Decode nested high-res images only where dimensions could match a train image.

    High-res suites store raw-byte hashes in images.sqlite rather than v2 RGB
    hashes. Comparing dimensions is safe for exact RGB identity because width
    and height are included in the original pixel-hash definition.
    """
    nested = []
    for suite in suites:
        rows = ORIGINAL.read_jsonl(suite)
        specialized = [row for row in rows if isinstance(row.get("image"), dict) and "sha256" in row["image"]]
        if specialized:
            nested.append((suite, specialized))
    if not nested:
        return {"suites": [], "decoded_images": 0}
    paths = {path for rows in pools.values() for row in rows for path in row["images"]}
    def dimensions(path):
        with Image.open(path) as image:
            return image.size
    with ThreadPoolExecutor(max_workers=workers) as executor:
        candidate_dimensions = set(executor.map(dimensions, sorted(paths)))
    audits = []
    for suite, rows in nested:
        database = suite.parent / "images.sqlite"
        unique = {}
        for row in rows:
            metadata = row["image"]
            key = metadata["sha256"]
            size = (int(metadata["width"]), int(metadata["height"]))
            if key in unique and unique[key] != size:
                raise ValueError("Conflicting evaluation image dimensions: " + key)
            unique[key] = size
        matching = {key: size for key, size in unique.items() if size in candidate_dimensions}
        if not database.is_file():
            raise FileNotFoundError("Nested evaluation image database is missing: " + str(database))
        with sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True) as connection:
            for key, expected_size in matching.items():
                fetched = connection.execute("SELECT data FROM images WHERE sha256=?", (key,)).fetchone()
                if fetched is None or hashlib.sha256(fetched[0]).hexdigest() != key:
                    raise ValueError("Nested evaluation image payload missing or changed: " + key)
                with Image.open(io.BytesIO(fetched[0])) as image:
                    if image.size != expected_size:
                        raise ValueError("Nested evaluation image dimensions disagree: " + key)
                    image = image.convert("RGB")
                    blocks["images"].add(hashlib.sha256(
                        struct.pack(">II", *image.size) + image.tobytes()).hexdigest())
        audits.append({
            "suite": str(suite), "database": str(database), "rows": len(rows),
            "unique_images": len(unique), "dimension_matched_images_decoded": len(matching),
            "payload_sha256_verified": len(matching),
        })
    return {
        "suites": audits,
        "candidate_dimensions": len(candidate_dimensions),
        "decoded_images": sum(row["dimension_matched_images_decoded"] for row in audits),
        "method": "Compare exact RGB hashes after filtering by frozen suite width/height metadata; each decoded SQLite payload is SHA256-verified against the pinned suite.",
        "limit": "Dimension screening relies on frozen evaluation metadata; nonmatching dimensions are not decoded. This is not a transformed-near-duplicate check.",
    }


def preserve_previous_prefix(records: list[dict], previous: Path | None, protocol_sha: str) -> list[dict]:
    if previous is None:
        return records
    manifest = json.loads((previous / "manifest.json").read_text())
    if manifest["protocol_sha256"] != protocol_sha:
        raise ValueError("Previous pool used a different training protocol")
    previous_records_path = previous / "candidates.records.jsonl"
    if ORIGINAL.sha256(previous_records_path) != manifest["records_sha256"]:
        raise ValueError("Previous pool records changed")
    prior = ORIGINAL.read_jsonl(previous_records_path)
    current = {row["id"]: row for row in records}
    for row in prior:
        if current.get(row["id"]) != row:
            raise ValueError("Previous candidate changed or is no longer eligible: " + row["id"])
    old_ids = {row["id"] for row in prior}
    return prior + [row for row in records if row["id"] not in old_ids]


def package_candidate(record: dict, index: int, protocol: dict) -> dict:
    row = ORIGINAL.package(record, index, protocol)
    row["ability"] = "perception_v3_candidate"
    row["extra_info"].update({
        "selection_version": VERSION,
        "dataset_stage": "candidate_pool",
        "difficulty_filtered": False,
    })
    return row


def verify_images(records: list[dict], workers: int) -> int:
    if workers < 1:
        raise ValueError("Image verification worker count must be positive")
    expected_by_path = {}
    for record in records:
        if not record["images"]:
            raise ValueError("Perception record has no image: " + record["id"])
        for path, expected in zip(record["images"], record["image_sha256"], strict=True):
            if path in expected_by_path and expected_by_path[path] != expected:
                raise ValueError("Contradictory image hash metadata: " + path)
            expected_by_path[path] = expected
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path, actual in zip(expected_by_path, pool.map(ORIGINAL.pixel_hash, expected_by_path), strict=True):
            if actual != expected_by_path[path]:
                raise ValueError("Candidate image pixel hash mismatch: " + path)
    return len(expected_by_path)


def build(protocol_path: Path, output: Path, workers: int = 4, inspect_only: bool = False,
          extra_paths: list[Path] | None = None, previous: Path | None = None) -> dict:
    protocol_path = protocol_path.resolve()
    output = output.resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol["group_size"] != 8:
        raise ValueError("The requested filter requires exactly eight completions per candidate")
    if not inspect_only and output.exists():
        raise FileExistsError("Refusing to overwrite an existing candidate pool: " + str(output))
    extra_paths = [path.resolve() for path in (extra_paths or [])]
    suites = sorted((EXPERIMENT / "evaluation").rglob("suite*.jsonl"))
    if not suites:
        raise ValueError("No evaluation suites found; cannot establish leakage exclusions")
    blocks, leakage = ORIGINAL.evaluation_blocks(suites)
    pools, sources = make_pools(blocks, extra_paths)
    nested_images = add_nested_image_blocks(blocks, suites, pools, workers)
    if nested_images["decoded_images"]:
        pools, sources = make_pools(blocks, extra_paths)
    leakage["nested_image_checks"] = nested_images
    leakage["blocked_counts"] = {key: len(value) for key, value in blocks.items()}
    if any(not rows for rows in pools.values()):
        raise ValueError("At least one required category has no eligible candidates")
    records = ordered_candidates(pools, protocol["seed"])
    records = preserve_previous_prefix(records, previous, ORIGINAL.sha256(protocol_path))
    n_target = protocol["steps"] * protocol["batch_size"]
    audit = {
        "version": VERSION,
        "artifact_kind": "candidate_pool",
        "difficulty_filtered": False,
        "training_ready": False,
        "required_downstream_filter": {"completions_per_candidate": 8, "min_correct": 2, "max_correct": 6},
        "seed": protocol["seed"],
        "target_category_weights": dict(ORIGINAL.WEIGHTS),
        "target_training_count": n_target,
        "target_category_counts": {key: n_target * value // 100 for key, value in ORIGINAL.WEIGHTS.items()},
        "candidate_categories": {key: ORIGINAL.summarize(rows) for key, rows in pools.items()},
        "candidate_all": ORIGINAL.summarize(records),
        "existence_labels": dict(Counter(ORIGINAL.consensus(row)[0] for row in pools["existence_confusion"])),
        "protocol_path": str(protocol_path),
        "protocol_sha256": ORIGINAL.sha256(protocol_path),
        "protocol": protocol,
        "builder_sha256": ORIGINAL.sha256(Path(__file__)),
        "original_rules_sha256": ORIGINAL.sha256(ROOT / "prepare_perception_v2.py"),
        "reward_sha256": ORIGINAL.sha256(EXPERIMENT / "rewards.py"),
        "sources": sources,
        "previous_pool": str(previous.resolve()) if previous else None,
        "leakage": leakage,
        "limitations": [
            "The candidate pool itself need not match the target mixture; the downstream filtered selection must.",
            "No frozen-model completions were generated and no candidate has passed difficulty filtering yet.",
            "Reuses v2 question routing and original labels without generated or modified annotations.",
            "Exact image/question exclusions do not prove the absence of transformed near-duplicates.",
        ],
    }
    if inspect_only:
        return audit
    image_count = verify_images(records, workers)
    audit["verified_image_files"] = image_count
    audit["reference_reward_validation"] = ORIGINAL.validate_references(records)
    # Supplemental downloads may continue independently, but files used by this
    # export must remain frozen throughout the validation pass.
    frozen_inputs = [(Path(item["path"]), item["sha256"]) for item in sources["input_files"]]
    frozen_inputs.extend((Path(item["path"]), item["sha256"]) for item in leakage["eval_manifests"])
    frozen_inputs.extend([
        (protocol_path, audit["protocol_sha256"]),
        (Path(__file__), audit["builder_sha256"]),
        (ROOT / "prepare_perception_v2.py", audit["original_rules_sha256"]),
        (EXPERIMENT / "rewards.py", audit["reward_sha256"]),
    ])
    for path, expected in frozen_inputs:
        if ORIGINAL.sha256(path) != expected:
            raise ValueError("An input changed during candidate export: " + str(path))
    # All expensive validation precedes publication. A failed build leaves no
    # apparently complete candidate dataset, and a completed pool is immutable.
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        records_path = temporary / "candidates.records.jsonl"
        with records_path.open("w") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        parquet_path = temporary / "candidates.parquet"
        pq.write_table(pa.Table.from_pylist([
            package_candidate(record, index, protocol) for index, record in enumerate(records)
        ]), parquet_path, compression="zstd")
        audit_path = temporary / "audit.json"
        audit["artifacts"] = {
            path.name: {"path": str(output / path.name), "sha256": ORIGINAL.sha256(path)}
            for path in (records_path, parquet_path)
        }
        audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
        manifest = {
            "version": VERSION,
            "artifact_kind": "candidate_pool",
            "difficulty_filtered": False,
            "training_ready": False,
            "n": len(records),
            "category_counts": {key: len(rows) for key, rows in pools.items()},
            "target_category_weights": {key: value / 100 for key, value in ORIGINAL.WEIGHTS.items()},
            "required_downstream_filter": audit["required_downstream_filter"],
            "data_file": str(output / parquet_path.name),
            "data_sha256": ORIGINAL.sha256(parquet_path),
            "records_path": str(output / records_path.name),
            "records_sha256": ORIGINAL.sha256(records_path),
            "audit_path": str(output / audit_path.name),
            "audit_sha256": ORIGINAL.sha256(audit_path),
            "protocol_sha256": audit["protocol_sha256"],
            "builder_sha256": audit["builder_sha256"],
            "original_rules_sha256": audit["original_rules_sha256"],
            "reward_sha256": audit["reward_sha256"],
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        if output.exists():
            raise FileExistsError("Output appeared during build; refusing to overwrite it: " + str(output))
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=EXPERIMENT / "training/config/common_local_100.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prepared/perception_v3_pool")
    parser.add_argument("--verify-workers", type=int, default=4)
    parser.add_argument("--inspect-only", action="store_true", help="Report eligible pools without writing artifacts")
    parser.add_argument("--extra-records", type=Path, action="append", default=[],
                        help="Additional normalized original-train JSONL; repeat for multiple files")
    parser.add_argument("--previous-pool", type=Path,
                        help="Keep this immutable prior pool as an unchanged prefix when adding new candidates")
    args = parser.parse_args()
    audit = build(args.protocol, args.output_dir, args.verify_workers, args.inspect_only,
                  args.extra_records, args.previous_pool)
    print(json.dumps({
        "output": None if args.inspect_only else str(args.output_dir.resolve()),
        "candidate_counts": {key: value["records"] for key, value in audit["candidate_categories"].items()},
        "total": audit["candidate_all"]["records"],
        "target_category_counts": audit["target_category_counts"],
        "difficulty_filtered": False,
        "training_ready": False,
    }, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
