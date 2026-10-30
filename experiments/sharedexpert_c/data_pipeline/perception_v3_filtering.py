"""CPU-only correctness and quota utilities for frozen-model eight-rollout screening.

These rules select data; they never change the existing training rewards. A
screening receipt describes one observed group, not future online success rates.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics


_REWARD_PATH = Path(__file__).resolve().parents[1] / "rewards.py"
_SPEC = importlib.util.spec_from_file_location("perception_v3_original_rewards", _REWARD_PATH)
_REWARDS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_REWARDS)
REWARD_SOURCE_SHA256 = hashlib.sha256(_REWARD_PATH.read_bytes()).hexdigest()

WEIGHTS = {
    "object_attribute": 20,
    "grounding": 20,
    "ocr_document": 20,
    "spatial_detail": 15,
    "existence_confusion": 15,
    "general_vqa_replay": 10,
}
POLICY = {
    "version": "perception-v3-eight-rollouts-two-through-six-v1",
    "group_size": 8,
    "min_correct": 2,
    "max_correct": 6,
    "bbox_correct_iou_minimum": 0.5,
    "vqa_correct": "full_consensus_score",
    "anls_correct": "full_anls_score",
    "full_score_absolute_tolerance": 1e-9,
    "require_think_close": True,
    "zero_on_length_termination": True,
    "transport_errors_are_not_incorrect_answers": True,
}
POLICY_SHA256 = hashlib.sha256(json.dumps(
    POLICY, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()


def _identity(record: dict) -> str:
    value = record.get("id") or record.get("extra_info", {}).get("record_id")
    if not isinstance(value, str) or not value:
        raise ValueError("A stable record id is required")
    return value


def record_sha256(record: dict) -> str:
    """Bind an observed group to the exact candidate, excluding its new receipt."""
    original = {key: value for key, value in record.items() if key != "difficulty_filter"}
    return hashlib.sha256(json.dumps(original, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _gold(record: dict) -> dict:
    if "reward_model" in record:
        gold = record["reward_model"]["ground_truth"]
        gold = json.loads(gold) if isinstance(gold, str) else gold
    else:
        gold = {"task": record["task"], "answers": record.get("answers", []),
                "format": "boxed"}
        if gold["task"] == "bbox":
            gold["bbox"] = record["bbox"]
    if gold.get("task") not in {"vqa", "anls", "bbox"}:
        raise ValueError(f"Unsupported perception screening task: {gold.get('task')!r}")
    return gold


def score_completion(record: dict, text: str, finish_reason: str) -> dict:
    """Score raw generated text with the unmodified training parser and reward.

Only normal EOS completion (``stop``) and token-budget exhaustion (``length``)
are model outcomes. Transport errors or missing finish reasons are invalid data.
    """
    if not isinstance(text, str):
        raise ValueError("Completion text must be a string")
    if finish_reason not in {"stop", "length"}:
        raise ValueError(f"Invalid completion finish_reason: {finish_reason!r}")
    gold = _gold(record)
    scored = _REWARDS.compute_score(
        data_source=record.get("data_source", "sharedexpert_" + record.get("source", "unknown")),
        solution_str=text,
        ground_truth=gold,
        extra_info={"require_think_close": True, "truncated": finish_reason == "length"},
    )
    reward = float(scored["score"])
    correct = (reward >= POLICY["bbox_correct_iou_minimum"] if gold["task"] == "bbox"
               else math.isclose(reward, 1.0, rel_tol=0.0,
                                 abs_tol=POLICY["full_score_absolute_tolerance"]))
    return {**scored, "correct": bool(correct and scored["format_valid"] and
                                       not scored["truncated"]),
            "finish_reason": finish_reason, "task": gold["task"]}


def accept_group(record: dict, completions: list[dict]) -> dict:
    """Rescore exactly eight unique raw outputs and return a screening receipt.

Each completion has ``index`` (0..7), ``text``, ``finish_reason`` and optional
``error``. Caller-provided scores/correctness flags are intentionally ignored.
Malformed or incomplete groups return accepted=False and n_correct=None, so
infrastructure failure is never represented as a wrong model answer.
    """
    receipt = {
        "record_id": _identity(record), "policy_sha256": POLICY_SHA256,
        "record_sha256": record_sha256(record), "reward_source_sha256": REWARD_SOURCE_SHA256,
        "policy_version": POLICY["version"], "accepted": False,
        "n_completions": len(completions), "n_correct": None,
        "rejection_reason": None,
    }
    if len(completions) != 8:
        return {**receipt, "rejection_reason": "expected_exactly_eight_completions"}
    if any(not isinstance(item, dict) for item in completions):
        return {**receipt, "rejection_reason": "invalid_completion_payload"}
    indices = [item.get("index") for item in completions]
    if (any(type(index) is not int for index in indices) or
            sorted(indices) != list(range(8))):
        return {**receipt, "rejection_reason": "invalid_or_duplicate_completion_indices"}
    if any(item.get("error") for item in completions):
        return {**receipt, "rejection_reason": "completion_error_requires_retry"}
    if any(not isinstance(item.get("text"), str) or
           item.get("finish_reason") not in {"stop", "length"} for item in completions):
        return {**receipt, "rejection_reason": "invalid_completion_payload"}
    scored = [score_completion(record, item["text"], item["finish_reason"])
              for item in sorted(completions, key=lambda item: item["index"])]
    rewards = [item["score"] for item in scored]
    correct = [item["correct"] for item in scored]
    n_correct = sum(correct)
    accepted = 2 <= n_correct <= 6
    return {
        **receipt, "accepted": accepted, "n_correct": n_correct,
        "rejection_reason": None if accepted else "correct_count_outside_two_through_six",
        "rewards": rewards, "correct_flags": correct,
        "reward_mean": statistics.mean(rewards),
        "reward_std": statistics.pstdev(rewards),
        "reward_range": max(rewards) - min(rewards),
        "n_truncated": sum(bool(item["truncated"]) for item in scored),
        "n_invalid_format": sum(not item["format_valid"] for item in scored),
    }


def _category(record: dict) -> str:
    return record.get("perception_category", record.get("extra_info", {}).get("perception_category"))


def _presence_label(record: dict) -> str:
    answers = Counter(_REWARDS.normalize_answer(answer) for answer in _gold(record)["answers"])
    if not answers:
        raise ValueError("Presence record has no annotated answer")
    most_common = answers.most_common()
    if (most_common[0][0] not in {"yes", "no"} or
            (len(most_common) > 1 and most_common[0][1] == most_common[1][1])):
        raise ValueError("Presence record must have an unambiguous yes/no majority")
    return most_common[0][0]


def _accepted_pools(accepted_records: list[dict]) -> dict[str, list[dict]]:
    pools = {category: [] for category in WEIGHTS}
    ids, pairs = set(), set()
    for record in accepted_records:
        identity = _identity(record)
        receipt = record.get("difficulty_filter", {})
        if (receipt.get("accepted") is not True or
                receipt.get("policy_sha256") != POLICY_SHA256 or
                receipt.get("record_id") != identity or
                receipt.get("record_sha256") != record_sha256(record) or
                receipt.get("reward_source_sha256") != REWARD_SOURCE_SHA256 or
                receipt.get("n_completions") != 8 or
                type(receipt.get("n_correct")) is not int or
                not 2 <= receipt["n_correct"] <= 6):
            raise ValueError(f"Missing or invalid accepted screening receipt: {identity}")
        category = _category(record)
        if category not in pools:
            raise ValueError(f"Unknown perception category: {category!r}")
        if identity in ids:
            raise ValueError(f"Duplicate accepted record: {identity}")
        ids.add(identity)
        images = record.get("image_sha256", record.get("extra_info", {}).get("image_sha256"))
        question = record.get("question_sha256", record.get("extra_info", {}).get("question_sha256"))
        if isinstance(images, str):
            images = json.loads(images)
        if images is not None and question is not None:
            pair = (tuple(images), question)
            if pair in pairs:
                raise ValueError(f"Duplicate accepted image/question pair: {identity}")
            pairs.add(pair)
        pools[category].append(record)
    return pools


def max_feasible_size(accepted_records: list[dict], balance_existence: bool = True) -> int:
    """Return largest distinct mixture; balanced presence requires multiples of 40."""
    pools = _accepted_pools(accepted_records)
    blocks = min(len(pools[category]) // (weight // 5) for category, weight in WEIGHTS.items())
    if balance_existence:
        labels = Counter(_presence_label(record) for record in pools["existence_confusion"])
        blocks = min(blocks, 2 * min(labels["yes"], labels["no"]) // 3)
        blocks -= blocks % 2
    return 20 * blocks


def select_records(accepted_records: list[dict], n: int, seed: int,
                   balance_existence: bool = True) -> list[dict]:
    """Select unique screened rows with exact original ratios in each 20-row block.

Presence yes/no balance is preserved by default. Attribute and spatial subtype
counts are left to the measured screened pools; only the six user-selected
category ratios are fixed. Input records and receipts are never mutated.
    """
    if type(n) is not int or n <= 0 or n % 20:
        raise ValueError("n must be a positive multiple of 20 for exact category ratios")
    if balance_existence and (n * WEIGHTS["existence_confusion"] // 100) % 2:
        raise ValueError("Balanced presence requires n divisible by 40")
    pools = _accepted_pools(accepted_records)
    selected = {}
    for category, weight in WEIGHTS.items():
        needed = n * weight // 100
        pool = sorted(pools[category], key=_identity)
        rng = random.Random(f"{seed}:{category}")
        if category == "existence_confusion" and balance_existence:
            chosen = []
            for answer in ("yes", "no"):
                candidates = [record for record in pool if _presence_label(record) == answer]
                if len(candidates) < needed // 2:
                    raise ValueError(f"Quota shortage for {category}:{answer}: need {needed // 2}, "
                                     f"have {len(candidates)}; no repeats or relaxed screening allowed")
                rng.shuffle(candidates)
                chosen.extend(candidates[:needed // 2])
            rng.shuffle(chosen)
        else:
            if len(pool) < needed:
                raise ValueError(f"Quota shortage for {category}: need {needed}, have {len(pool)}; "
                                 "no repeats or relaxed screening allowed")
            rng.shuffle(pool)
            chosen = pool[:needed]
        selected[category] = chosen
    # Same 20-row interleaving protocol as prepare_perception_v2.interleave.
    result, offsets, rng = [], Counter(), random.Random(seed)
    for _ in range(n // 20):
        labels = [category for category, weight in WEIGHTS.items() for _ in range(weight // 5)]
        rng.shuffle(labels)
        for category in labels:
            result.append(selected[category][offsets[category]])
            offsets[category] += 1
    return result
