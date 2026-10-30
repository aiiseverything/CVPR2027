#!/usr/bin/env python3
"""Build the separately versioned, six-category perception teacher data.

Only existing, original train annotations are used. No label generation, model
judge, cyclic sampling, or mutation of the original teacher data is permitted.
See PERCEPTION_V2.md for the selection and decontamination boundaries.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import re
import struct
import sys
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent
sys.path.insert(0, str(EXPERIMENT))
from rewards import compute_score, normalize_answer  # noqa: E402

WEIGHTS = {
    "object_attribute": 20,
    "grounding": 20,
    "ocr_document": 20,
    "spatial_detail": 15,
    "existence_confusion": 15,
    "general_vqa_replay": 10,
}
VERSION = "perception-v2-original-annotations-v1"
SOURCES = ("vqav2", "refcoco", "docvqa")
BAD_VQA = re.compile(
    r"\b(why|think|would|could|should|might|probably|likely|because|how many|"
    r"how much|more than|less than|most|least|how old|time|purpose|ethnic|"
    r"ethnicity|race|religion|nationality|age|someone|anyone|make sure|used for|"
    r"use for|made for|meant for|appropriate|safe|dangerous|happy|sad|upset|"
    r"angry|feel|feeling|tasty|delicious|professional|expensive|season|reason|"
    r"symbolize|represent|imply|mean|means|famous|popular|healthy|healthier|"
    r"looking at|sauce|cheese|ingredients?|used)\b|for\?$"
)
SPATIAL = re.compile(
    r"\b(left|right|behind|in front of|under|underneath|above|below|on top of|"
    r"foreground|background|between|next to)\b"
)
LOCAL = re.compile(
    r"^(?:what|which|what's)\b.*\b(?:on (?:the|his|her|its)|"
    r"in (?:the|his|her|its) (?:hand|hands|mouth)|wearing on|attached to)\b"
)
TEXT_OR_IDENTITY = re.compile(
    r"\b(numbers?|say|says|written|read|brands?|words?|text|letters?|logos?|"
    r"city|country|state|names?|writing)\b"
)
EXISTENCE = re.compile(r"^(is there|are there|do you see|can you see)\b")
NOT_EXISTENCE = re.compile(
    r"\b(enough|lots?|more|less|few|fewer|through|same|different|two|three|"
    r"four|five|six|seven|eight|nine|ten|many|several|another|other|will|"
    r"able|possible|opportunity|chance|evidence|signs of|room for|caffeine|"
    r"alcohol|source|waiting|trying|about to|ready|clean water)\b"
)
ATTRIBUTE = re.compile(
    r"^(what colou?r|what (?:kind|type|breed|species|material|shape)|"
    r"what is (?:this|that|the) (?:object|animal|vehicle)|which (?:animal|bird|vehicle))\b"
)
REPLAY = re.compile(
    r"^(what (?:is|are) .*(?:doing|playing|holding|eating|wearing)|what sport|"
    r"what room|which room|what kind of room|what type of room)\b"
)
BAD_DOCUMENT = re.compile(
    r"\b(how many|how much|why|calculate|calculated|compute|difference|sum|"
    r"total|average|mean|median|ratio|increase|decrease|greater|less|more|"
    r"highest|lowest|maximum|minimum|most|least|compare|comparison|percent change|"
    r"percentage change|combined|altogether|subtract|multiply|divide|"
    r"denote|denotes|stand for|stands for|purpose|meaning|type of|kind of|"
    r"would|could|should|might|similar|interest|explain|excludes|first|last|before|after|between)\b"
)
DOCUMENT_FIELD = re.compile(
    r"\b(date|dated|name|named|title|heading|headings|header|number|code|address|"
    r"phone|telephone|fax|email|e-mail|page|value|percent|percentage|price|cost|"
    r"amount|rate|year|month|day|author|sender|recipient|subject|company|"
    r"organization|organisation|written|printed|listed|shown|labeled|labelled|"
    r"label|table|column|row|signature|signed|signatory|attendees|attendee|slogan|"
    r"caption|footer|section|stamp|stamped|time|location|city|country)\b"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def question_hash(question: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", question).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pixel_hash(path: str) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(struct.pack(">II", *image.size) + image.tobytes()).hexdigest()


def consensus(record: dict) -> tuple[str, int]:
    return Counter(map(normalize_answer, record["answers"])).most_common(1)[0]


def classify(record: dict) -> tuple[str, str] | None:
    """Conservative, reproducible routing; category membership is exclusive."""
    source = record["source"]
    question = " ".join(record["question"].lower().split())
    if source == "refcoco":
        return "grounding", "original_refcoco_train_bbox"
    if source == "docvqa":
        if not re.match(r"^(what|which|who|when|where)\b", question):
            return None
        if (BAD_DOCUMENT.search(question) or not DOCUMENT_FIELD.search(question)
                or len(question.split()) > 35):
            return None
        return "ocr_document", "document_field_lookup_no_arithmetic_cues"
    if source != "vqav2" or BAD_VQA.search(question) or len(question.split()) > 24:
        return None
    answer, agreement = consensus(record)
    if EXISTENCE.search(question):
        if agreement >= 8 and answer in {"yes", "no"} and not NOT_EXISTENCE.search(question):
            return "existence_confusion", "explicit_existence_8_of_10_" + answer
        # A failed existence candidate must not leak into a different category.
        return None
    if agreement < 7 or answer in {"0", "none", "nothing", "unknown", "not sure", "cant tell",
                                   "cannot tell", "cant see", "not visible", "unanswerable", "unsure"}:
        return None
    boolean_question = bool(re.match(r"^(is|are|do|does|has|have)\b", question))
    if not boolean_question:
        if re.match(r"^(no|not|cant|cannot)\b", answer):
            return None
        # Multiple annotators rejecting the premise make a positive object/attribute
        # answer questionable, even when it is the modal spelling.
        premise_rejections = sum(bool(re.match(r"^(no|none|nothing|not|cant|cannot|unknown)\b", a.lower()))
                                 for a in record["answers"])
        if premise_rejections >= 2:
            return None
    if (re.match(r"^(what|which|where|is|are|does|do)\b", question)
            and SPATIAL.search(question) and not TEXT_OR_IDENTITY.search(question)):
        return "spatial_detail", "explicit_spatial_reference"
    if LOCAL.search(question) and not TEXT_OR_IDENTITY.search(question):
        return "spatial_detail", "localized_visible_detail"
    if REPLAY.search(question):
        return "general_vqa_replay", "visible_activity_object_or_room"
    if ATTRIBUTE.search(question) and not TEXT_OR_IDENTITY.search(question):
        subtype = "color" if re.match(r"what colou?r\b", question) else "object_or_other_attribute"
        return "object_attribute", subtype
    return None


def evaluation_blocks(paths: list[Path]) -> tuple[dict, dict]:
    rows = [row for path in paths for row in read_jsonl(path)]
    blocks = {
        "images": {h for row in rows for h in row.get("image_sha256", [])},
        "questions": {question_hash(row["question"]) for row in rows},
        "ids": {str(row["id"]) for row in rows},
        "source_ids": {(row.get("source"), str(row["source_id"]))
                       for row in rows if row.get("source") and row.get("source_id")},
        "coco_ids": {int(row["image_id"]) for row in rows if row.get("image_id")},
    }
    # RefCOCO original val image IDs are excluded even when outside our 500-row suite.
    selection = ROOT / "raw/refcoco_selection.json"
    blocks["coco_ids"].update(json.loads(selection.read_text())["all_original_val_image_ids"])
    audit = {
        "eval_manifests": [{"path": str(path), "sha256": sha256(path),
                            "records": len(read_jsonl(path))} for path in paths],
        "benchmarks": dict(Counter(row.get("benchmark", "unknown") for row in rows)),
        "blocked_counts": {key: len(value) for key, value in blocks.items()},
        "refcoco_selection_sha256": sha256(selection),
        "question_policy": "Exclude every exact normalized evaluation question, even on a different image.",
        "limits": [
            "Uses recorded RGB pixel hashes in the evaluation manifests; selected train image hashes are recomputed.",
            "Exact pixels do not detect all resized, re-encoded, cropped, or otherwise transformed near-duplicates.",
            "Original COCO image IDs and source record IDs are checked when available; not all benchmarks expose original IDs.",
            "Does not claim absence of prior base-model pretraining exposure or overlap with unavailable external test sets.",
        ],
    }
    return blocks, audit


def make_pools(blocks: dict) -> tuple[dict, dict]:
    pools = {category: [] for category in WEIGHTS}
    audit = {}
    seen_pairs, seen_ids = set(), set()
    for source in SOURCES:
        path = ROOT / "normalized" / (source + ".jsonl")
        records = read_jsonl(path)
        stats = Counter(input_records=len(records))
        labels = defaultdict(set)
        if source == "refcoco":
            for record in records:
                labels[(tuple(record["image_sha256"]), record["question_sha256"])].add(
                    tuple(round(x, 6) for x in record["bbox"]))
        ambiguous = {key for key, targets in labels.items() if len(targets) > 1}
        stats["ambiguous_bbox_pairs"] = len(ambiguous)
        for record in records:
            if record["split"] != "train":
                raise ValueError("Non-train record: " + record["id"])
            if question_hash(record["question"]) != record["question_sha256"]:
                raise ValueError("Question hash mismatch: " + record["id"])
            pair = tuple(record["image_sha256"]), record["question_sha256"]
            if blocks["images"].intersection(record["image_sha256"]):
                stats["evaluation_image_removed"] += 1
                continue
            if record["question_sha256"] in blocks["questions"]:
                stats["evaluation_question_removed"] += 1
                continue
            if (record["id"] in blocks["ids"] or
                    (source, str(record["source_id"])) in blocks["source_ids"]):
                stats["evaluation_record_id_removed"] += 1
                continue
            if record.get("image_id") and int(record["image_id"]) in blocks["coco_ids"]:
                stats["evaluation_coco_image_id_removed"] += 1
                continue
            if pair in ambiguous:
                stats["ambiguous_bbox_removed"] += 1
                continue
            routed = classify(record)
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
        audit[source] = {**dict(stats), "normalized_path": str(path), "normalized_sha256": sha256(path)}
    return pools, audit


def take_unique(pool: list[dict], count: int, rng: random.Random, label: str) -> list[dict]:
    if len(pool) < count:
        raise RuntimeError(f"Insufficient unique {label}: need {count}, have {len(pool)}; no repeats allowed")
    candidates = list(pool)
    rng.shuffle(candidates)
    return candidates[:count]


def select(pools: dict, n: int, seed: int) -> dict:
    selected = {}
    for category, weight in WEIGHTS.items():
        rng = random.Random(f"{seed}:{category}")
        count = n * weight // 100
        pool = pools[category]
        if category == "existence_confusion":
            if count % 2:
                raise ValueError("Existence quota must be even for equal yes/no exposure")
            chosen = [record for answer in ("yes", "no") for record in take_unique(
                [r for r in pool if consensus(r)[0] == answer], count // 2, rng, category + ":" + answer)]
        elif category == "spatial_detail":
            local_pool = [r for r in pool if r["selection_rule"] == "localized_visible_detail"]
            local_count = min(count // 2, len(local_pool))
            chosen = take_unique([r for r in pool if r["selection_rule"] == "explicit_spatial_reference"],
                                 count - local_count, rng, category + ":spatial")
            chosen += take_unique(local_pool, local_count, rng, category + ":local_detail")
        elif category == "object_attribute":
            # Keep the category from degenerating into an almost entirely color dataset.
            objects = count // 4
            chosen = take_unique([r for r in pool if r["selection_rule"] == "object_or_other_attribute"],
                                 objects, rng, category + ":object_or_other_attribute")
            chosen += take_unique([r for r in pool if r["selection_rule"] == "color"],
                                  count - objects, rng, category + ":color")
        else:
            chosen = take_unique(pool, count, rng, category)
        rng.shuffle(chosen)
        selected[category] = chosen
    return selected


def interleave(selected: dict, n: int, seed: int) -> list[dict]:
    if n % 20:
        raise ValueError("The exposure count must be divisible by the exact 20-row mixture block")
    result = []
    offsets = Counter()
    rng = random.Random(seed)
    for _ in range(n // 20):
        labels = [category for category, weight in WEIGHTS.items() for _ in range(weight // 5)]
        rng.shuffle(labels)
        for category in labels:
            result.append(selected[category][offsets[category]])
            offsets[category] += 1
    return result


def reward_payload(record: dict) -> dict:
    payload = {"task": record["task"], "answers": record["answers"], "format": "boxed"}
    if record["task"] == "bbox":
        payload["bbox"] = record["bbox"]
    return payload


def reference_answer(record: dict) -> str:
    if record["task"] == "bbox":
        return json.dumps(record["bbox"])
    if record["task"] == "vqa":
        wanted = consensus(record)[0]
        return next(answer for answer in record["answers"] if normalize_answer(answer) == wanted)
    return record["answers"][0]


def package(record: dict, index: int, protocol: dict) -> dict:
    cap = min(protocol["max_image_pixels"], protocol["max_total_image_pixels"] // len(record["images"]))
    return {
        "data_source": "sharedexpert_" + record["source"],
        "prompt": [{"role": "user", "content": "<image>\n" * len(record["images"]) +
                    record["question"] + "\nGive the final answer inside \\boxed{...}."}],
        "images": [{"path": path, "min_pixels": protocol["min_image_pixels"], "max_pixels": cap}
                   for path in record["images"]],
        "ability": "perception_v2",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(reward_payload(record), ensure_ascii=False)},
        "extra_info": {
            "source": record["source"], "source_id": record["source_id"], "split": "train",
            "task": record["task"], "question_sha256": record["question_sha256"],
            "image_sha256": json.dumps(record["image_sha256"]), "record_id": record["id"], "index": index,
            "perception_category": record["perception_category"], "selection_rule": record["selection_rule"],
            "selection_version": VERSION,
        },
    }


def summarize(records: list[dict]) -> dict:
    return {
        "records": len(records), "unique_record_ids": len({r["id"] for r in records}),
        "unique_images": len({h for r in records for h in r["image_sha256"]}),
        "unique_image_questions": len({(tuple(r["image_sha256"]), r["question_sha256"]) for r in records}),
        "sources": dict(Counter(r["source"] for r in records)),
        "reward_tasks": dict(Counter(r["task"] for r in records)),
        "selection_rules": dict(Counter(r["selection_rule"] for r in records)),
    }


def validate_references(records: list[dict]) -> dict:
    scores = Counter()
    below_full = []
    for record in records:
        answer = reference_answer(record)
        solution = "</think>\\boxed{" + answer + "}"
        result = compute_score("sharedexpert_" + record["source"], solution, reward_payload(record),
                               {"require_think_close": True})
        if result["format_valid"] != 1 or result["score"] <= 0:
            raise RuntimeError("Unparseable or unrewarded reference: " + record["id"])
        scores[str(result["score"])] += 1
        if result["score"] < 1:
            below_full.append({"id": record["id"], "score": result["score"], "answer": answer})
    return {"checked": len(records), "score_histogram": dict(scores), "below_full": below_full,
            "note": "Original annotator multiplicities are preserved. VQA samples require >=7 matching annotations (>=8 for presence/absence); the selected majority answer therefore receives full consensus reward."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=EXPERIMENT / "training/config/common_local_100.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "prepared")
    parser.add_argument("--verify-workers", type=int, default=4)
    parser.add_argument("--skip-image-verification", action="store_true", help="Inspection only; refuses to emit train parquet")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    generalist_protocol_path = EXPERIMENT / "runs/generalist/protocol.json"
    generalist_protocol = json.loads(generalist_protocol_path.read_text())["protocol"]
    if protocol != generalist_protocol:
        raise RuntimeError("Configuration differs from the currently recorded generalist protocol")
    n = protocol["steps"] * protocol["batch_size"]
    if n % 60:
        raise ValueError("This data version requires complete 60-prompt / two-step exposure cycles")
    suites = sorted((EXPERIMENT / "evaluation").glob("suite*.jsonl"))
    if not suites:
        raise RuntimeError("No current evaluation suites found")
    blocks, leakage = evaluation_blocks(suites)
    pools, source_audit = make_pools(blocks)
    pool_audit = {category: summarize(records) for category, records in pools.items()}
    print(json.dumps({"eligible_pools": pool_audit}, indent=2), flush=True)
    selected = select(pools, n, protocol["seed"])
    records = interleave(selected, n, protocol["seed"])
    if len({r["id"] for r in records}) != n:
        raise RuntimeError("Repeated record exposure")
    if len({(tuple(r["image_sha256"]), r["question_sha256"]) for r in records}) != n:
        raise RuntimeError("Repeated image/question exposure")
    image_expected = {}
    for record in records:
        for path, expected in zip(record["images"], record["image_sha256"], strict=True):
            if path in image_expected and image_expected[path] != expected:
                raise RuntimeError("Contradictory image hash metadata")
            image_expected[path] = expected
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "perception_v2.pool_audit.json").write_text(json.dumps(pool_audit, indent=2))
    samples = {category: [{"id": r["id"], "question": r["question"], "answers": r["answers"],
                           "bbox": r.get("bbox"), "reference_answer": reference_answer(r),
                           "images": r["images"], "selection_rule": r["selection_rule"]}
                          for r in rows[:12]] for category, rows in selected.items()}
    (output / "perception_v2.samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    if args.skip_image_verification:
        print("Inspection samples written; no train parquet emitted", flush=True)
        return
    with ThreadPoolExecutor(max_workers=args.verify_workers) as pool:
        for path, actual in zip(image_expected, pool.map(pixel_hash, image_expected), strict=True):
            if actual != image_expected[path]:
                raise RuntimeError("Selected image pixel hash mismatch: " + path)
    rewards = validate_references(records)
    rows = [package(record, i, protocol) for i, record in enumerate(records)]
    parquet_path = output / "perception_v2.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path, compression="zstd")
    # Smoke rows cover every category but do not predict rollout reward variance.
    smoke_records = [selected[category][0] for category in WEIGHTS]
    smoke_path = output / "perception_v2_smoke.parquet"
    pq.write_table(pa.Table.from_pylist([package(r, i, protocol) for i, r in enumerate(smoke_records)]),
                   smoke_path, compression="zstd")
    records_path = output / "perception_v2.records.jsonl"
    records_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    audit = {
        "version": VERSION, "seed": protocol["seed"], "n": n, "steps": protocol["steps"],
        "batch_size": protocol["batch_size"], "group_size": protocol["group_size"],
        "category_weights": WEIGHTS, "block_size": 20,
        "block_counts": {k: v // 5 for k, v in WEIGHTS.items()},
        "protocol_path": str(args.protocol), "protocol_sha256": sha256(args.protocol),
        "generalist_protocol_path": str(generalist_protocol_path),
        "generalist_protocol_sha256": sha256(generalist_protocol_path),
        "builder_sha256": sha256(Path(__file__)), "reward_sha256": sha256(EXPERIMENT / "rewards.py"),
        "sources": source_audit, "available_pools": pool_audit,
        "selected_categories": {category: summarize(rows_) for category, rows_ in selected.items()},
        "selected_all": summarize(records), "repeat_exposures": 0,
        "existence_labels": dict(Counter(consensus(r)[0] for r in selected["existence_confusion"])),
        "selected_image_files_verified": len(image_expected), "leakage": leakage,
        "reference_reward_validation": rewards,
        "smoke": {"path": str(smoke_path), "records": 6,
                  "note": "One record per category; no claim of measured difficulty or non-uniform rollout rewards."},
        "artifacts": {path.name: {"path": str(path), "sha256": sha256(path)}
                      for path in (parquet_path, smoke_path, records_path, output / "perception_v2.samples.json")},
        "selection_limitations": [
            "Question rules and annotator agreement select candidate perception tasks; they are not a model-based difficulty measurement.",
            "Category 5 contains actual annotated presence/absence positives and negatives; it is not advertised as adversarial hard-negative mining.",
            "Category 4 covers explicit spatial references and localized visible details; small-object size and occlusion are not newly annotated.",
            "DocVQA original ANLS reward is unchanged, including partial credit for near-matching numeric strings.",
        ],
    }
    audit_path = output / "perception_v2.audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2))
    manifest = {
        "teacher": "perception_v2", "version": VERSION,
        "data_file": str(parquet_path), "data_sha256": sha256(parquet_path),
        "smoke_file": str(smoke_path), "smoke_sha256": sha256(smoke_path),
        "n": n, "category_weights": {k: v / 100 for k, v in WEIGHTS.items()},
        "protocol_sha256": sha256(args.protocol), "builder_sha256": sha256(Path(__file__)),
        "audit_path": str(audit_path), "audit_sha256": sha256(audit_path),
        "records_path": str(records_path), "records_sha256": sha256(records_path),
    }
    (output / "perception_v2.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(json.dumps({"written": str(parquet_path), "selected": audit["selected_all"],
                      "existence_labels": audit["existence_labels"], "rewards": rewards}, indent=2), flush=True)


if __name__ == "__main__":
    main()
