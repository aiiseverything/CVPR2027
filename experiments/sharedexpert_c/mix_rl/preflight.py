"""Read-only CPU validation for the fixed-data Mix-RL/OPD comparison."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from experiments.sharedexpert_c.distillation.config import (
    WORKSPACE, _check_weights, _image_signature, _read_object,
    _tokenizer_signature, load_config as load_opd_config,
)

MATCHED_FIELDS = ("steps", "batch_size", "prompt_length", "response_length",
                  "learning_rate", "temperature", "top_p", "top_k")
DOMAINS = ("perception", "reasoning", "generalist")
TASKS = {"bbox", "vqa", "math", "anls", "relaxed_numeric", "exact", "json"}


def _path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else WORKSPACE / path).resolve()


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_student(path):
    """Inspect metadata and safetensor extents; never load tensors or teachers."""
    directory = _path(path)
    model = _read_object(directory / "config.json")
    if model.get("model_type") != "qwen3_5":
        raise ValueError("Only the project's dense multimodal Qwen3.5 student is supported")
    tokenizer_config = _read_object(directory / "tokenizer_config.json")
    template = directory / "chat_template.jinja"
    if not tokenizer_config.get("chat_template") and not (template.is_file() and template.stat().st_size):
        raise ValueError("Missing student chat template")
    _check_weights(directory)
    signature = _tokenizer_signature(directory)
    image = _image_signature(directory)
    return {"path": str(directory), "model_type": model["model_type"],
            "vocab_size": signature["vocab_size"], "image_patch_size": image["patch_size"],
            "weights": "safetensor headers and complete payload extents validated"}


def check_comparison(config):
    errors = []
    opd = load_opd_config(_path(config["opd_config_path"]))
    matches = {}
    for key in ("student_path", "data_path"):
        matches[key] = str(_path(config[key])) == opd[key]
    for key in MATCHED_FIELDS:
        matches[key] = config.get(key) == opd.get(key)
    for key, matched in matches.items():
        if not matched:
            errors.append(f"Mix-RL {key} differs from OPD; the requested comparison requires a match")
    if config.get("rollout_n") != 8 or opd.get("rollout_n") != 1:
        errors.append("Comparison requires Mix-RL rollout_n=8 and OPD rollout_n=1")
    return {"errors": errors, "opd_config_path": str(_path(config["opd_config_path"])),
            "matched_fields": matches, "teacher_checkpoints_required": False,
            "opd_responses": opd["steps"] * opd["batch_size"] * opd["rollout_n"],
            "mix_rl_responses": config["steps"] * config["batch_size"] * config["rollout_n"],
            "scope": "Same ordered prompts and optimizer steps; eight times the sampled responses, not equal compute."}


def _reference_score(row):
    """Verify gold can earn reward; this is a schema audit, not model accuracy."""
    from experiments.sharedexpert_c import rewards
    reward = row["reward_model"]
    if reward.get("style") != "rule":
        raise ValueError("reward_model.style must be rule")
    gold = json.loads(reward["ground_truth"])
    task = gold.get("task")
    if task not in TASKS:
        raise ValueError(f"Unsupported task: {task!r}")
    if gold.get("format") not in ("boxed", "json"):
        raise ValueError("Gold must specify boxed or json format")
    if task == "json" and gold.get("content_task", "exact") not in TASKS - {"json", "bbox"}:
        raise ValueError("Unsupported JSON content_task")
    if task == "bbox":
        box = gold.get("bbox")
        if (not isinstance(box, list) or len(box) != 4 or
                any(isinstance(x, bool) or not isinstance(x, (float, int)) or
                    not math.isfinite(x) or not 0 <= x <= 1000 for x in box) or
                not (box[0] < box[2] and box[1] < box[3])):
            raise ValueError("Invalid bbox reference")
        candidates = [json.dumps(box)]
    else:
        answers = gold.get("answers")
        if not isinstance(answers, list) or not answers:
            raise ValueError("Gold answers must be a nonempty list")
        if any(not isinstance(answer, str) or not answer.strip() for answer in answers):
            raise ValueError("Gold answers must contain nonempty strings")
        if (task == "vqa" or (task == "json" and gold.get("content_task") == "vqa")) and len(answers) != 10:
            raise ValueError("VQA consensus requires ten annotator answers")
        candidates = list(dict.fromkeys(answers))
    best = 0.0
    for candidate in candidates:
        if task == "math":
            candidate = rewards.last_boxed(candidate) or candidate
        final = (json.dumps({gold.get("json_key", "answer"): candidate}, ensure_ascii=False)
                 if gold["format"] == "json" or task == "json"
                 else "\\boxed{" + candidate + "}")
        result = rewards.compute_score(row["data_source"], "</think>\n" + final, gold,
                                       {"require_think_close": True, "truncated": False})
        score = result["score"]
        if not math.isfinite(score) or not 0 <= score <= 1 or result["format_valid"] != 1:
            raise ValueError("Reference produced invalid reward or final-answer format")
        best = max(best, score)
    if best <= 0:
        raise ValueError("No gold reference receives positive reward")
    return task, best


def check_inputs(config):
    """Return an audit report without mutating data or allocating GPU memory."""
    from experiments.sharedexpert_c import rewards
    from experiments.sharedexpert_c.training.preflight import check_dataset
    import pyarrow.parquet as pq

    report = {"errors": [], "dataset": {}, "reward_validation": {}, "comparison": {}}
    errors = report["errors"]
    try:
        report["student"] = check_student(config["student_path"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"student: {exc}")
    try:
        report["comparison"] = check_comparison(config)
        errors.extend(report["comparison"]["errors"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"comparison: {exc}")
    try:
        report["reward_validation"]["runtime"] = rewards.validate_runtime()
    except Exception as exc:
        errors.append(f"reward runtime: {type(exc).__name__}: {exc}")
        return report
    try:
        path = _path(config["data_path"])
        expected = config.get("expected_data_sha256")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            errors.append("expected_data_sha256 must pin the original 6,400-prompt OPD parquet")
        digest = _sha256(path)
        if digest != expected:
            errors.append("Dataset SHA-256 differs from the pinned OPD data")
        dataset = check_dataset(path, config["steps"] * config["batch_size"])
        report["dataset"] = {**dataset, "sha256": digest, "expected_sha256": expected}
        errors.extend(f"dataset: {error}" for error in dataset["errors"])
        rows = pq.read_table(path).to_pylist()
        domains, task_counts, source_counts, seen = Counter(), Counter(), Counter(), set()
        failures = []
        audited = 0
        minimum_reward = 1.0
        for index, row in enumerate(rows):
            try:
                domain = row.get("domain")
                if domain != DOMAINS[index % 3]:
                    raise ValueError("Expected cyclic perception/reasoning/generalist domain order")
                if row.get("opd_index") != index:
                    raise ValueError("opd_index does not preserve original ordered prompt identity")
                info = row.get("extra_info") or {}
                record_id = info.get("record_id")
                if not isinstance(record_id, str) or not record_id or record_id in seen:
                    raise ValueError("Missing or duplicate record_id")
                seen.add(record_id)
                if info.get("split") != "train":
                    raise ValueError("Expected training split metadata")
                if not isinstance(row.get("data_source"), str) or not row["data_source"]:
                    raise ValueError("Missing data_source for reward manager")
                task, score = _reference_score(row)
                domains[domain] += 1
                task_counts[task] += 1
                source_counts[row["data_source"]] += 1
                minimum_reward = min(minimum_reward, score)
                audited += 1
            except Exception as exc:
                if len(failures) < 20:
                    failures.append(f"row {index}: {type(exc).__name__}: {exc}")
        if failures:
            errors.extend(f"reward/data: {error}" for error in failures)
        report["dataset"].update(domain_counts=dict(domains), ordered_unique_records=len(seen),
                                  order="perception,reasoning,generalist; shuffle=False")
        report["reward_validation"].update(
            audited_rows=audited, task_counts=dict(task_counts), source_counts=dict(source_counts),
            minimum_best_reference_reward=minimum_reward if audited else None, errors=failures,
            scope="Gold reference self-consistency only; VQA consensus may score below one; not sampled-model evaluation.")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        errors.append(f"dataset: {type(exc).__name__}: {exc}")
    return report
