"""Shared, network-free evaluation primitives. No VLMEvalKit or .env import."""
from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
# Frozen four-model comparison plus named intermediate checkpoints evaluated
# under the same protocol (e.g. the step-100 perception/reasoning exports).
MODEL_NAMES = ["baseline", "generalist", "perception", "reasoning", "perception-step100", "reasoning-step100"]
PROTOCOL = {
    "version": "sharedexpert-teacher-eval-v3-total-image-budget",
    "enable_thinking": True,
    "temperature": 0.0,
    "top_p": 1.0,
    "seed": 20261019,
    "max_tokens": 8192,
    "max_prompt_tokens": 2048,
    "max_model_len": 10240,
    "max_images": 8,
    "image_size": {"longest_edge": 786432, "shortest_edge": 4096},
    "max_total_image_pixels": 1572864,
    "multi_image_policy": "per-image max_pixels=min(786432,1572864//image_count)",
    "force_answering": False,
    "judge": "local-strict-extraction-and-exact-or-numeric-match-v1",
    "max_token_policy": "no extra answer pass; final answer must be present in original response",
}
# Scoring revisions never change the frozen prompts or generation-cache identity.
SCORING_PROTOCOL = {
    "version": "strict-local-v2-length-zero",
    "zero_on_length_termination": True,
    "length_finish_reason": "length",
    "preserve_extraction_and_pre_penalty_score_for_audit": True,
    "recompute_from_raw_response": True,
}


def image_size_for_count(image_count):
    if image_count < 1 or image_count > PROTOCOL["max_images"]:
        raise ValueError(f"Unsupported image count: {image_count}")
    return {**PROTOCOL["image_size"], "longest_edge": min(
        PROTOCOL["image_size"]["longest_edge"],
        PROTOCOL["max_total_image_pixels"] // image_count,
    )}


def local_lock_path(resource, purpose):
    # EPC's distributed flock lease can outlive a process killed by cgroup OOM.
    # Inference endpoints and GPUs are host-local, so use a host-local flock too.
    directory = Path('/tmp') / f'sharedexpert-evaluation-locks-{os.getuid()}'
    directory.mkdir(mode=0o700, exist_ok=True)
    identity = hashlib.sha256(f'{Path(resource).resolve()}:{purpose}'.encode()).hexdigest()
    return directory / (identity + '.lock')


def json_bytes(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def digest(obj):
    return hashlib.sha256(json_bytes(obj)).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalize_question(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def question_sha(text):
    return hashlib.sha256(normalize_question(text).encode()).hexdigest()


def pixel_sha(image):
    rgb = image.convert("RGB")
    return hashlib.sha256(struct.pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()


def read_jsonl(path):
    if not Path(path).exists():
        return []
    text = Path(path).read_text()
    lines = text.splitlines()
    # Live inference appends records. Ignore an incomplete last write only;
    # malformed committed records still fail loudly.
    if lines and not text.endswith("\n"):
        try:
            json.loads(lines[-1])
        except json.JSONDecodeError:
            lines.pop()
    return [json.loads(line) for line in lines if line.strip()]


def load_suite(path):
    path = Path(path)
    meta = json.loads(path.with_suffix(".meta.json").read_text())
    if file_sha(path) != meta["suite_sha256"]:
        raise ValueError("Suite changed after preparation; refusing inconsistent evaluation")
    if meta["protocol"] != PROTOCOL:
        raise ValueError("Protocol changed; prepare a separate evaluation suite")
    rows = read_jsonl(path)
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate evaluation IDs")
    return rows, meta


def message_content(prompt, image_items):
    content = []
    for i, item in enumerate(image_items):
        if len(image_items) > 1:
            content.append({"type": "text", "text": f"Image {i + 1}:"})
        content.append(item)
    content.append({"type": "text", "text": prompt})
    return content


def final_answer(text):
    """Never award a number or option mentioned only in an unfinished thought."""
    text = text or ""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return "", "unfinished_thinking"
    tagged = re.findall(r"<answer>(.*?)</answer>", text, flags=re.S | re.I)
    if tagged:
        return tagged[-1].strip(), "answer_tag"
    # Balanced braces, including nested fractions; avoid regex stopping at first }.
    starts = list(re.finditer(r"\\boxed\s*\{", text))
    for start in reversed(starts):
        depth, pos = 1, start.end()
        for end in range(pos, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    return text[pos:end].strip(), "boxed"
    lines = [x.strip().strip("* ") for x in text.splitlines() if x.strip()]
    if not lines:
        return "", "empty"
    last = re.sub(r"^(?:final\s+)?answer\s*[:：]\s*", "", lines[-1], flags=re.I)
    return last, "last_line"


def canonical(text):
    text = unicodedata.normalize("NFKC", str(text)).strip().lower()
    text = text.replace("\\left", "").replace("\\right", "").replace("\\,", "")
    text = re.sub(r"\\(?:mathrm|text|mathbf)\{([^{}]*)\}", r"\1", text)
    return re.sub(r"\s+", "", text).strip("$ .。")


def number(text):
    from fractions import Fraction
    text = canonical(text).replace(",", "")
    text = re.sub(r"\\(?:dfrac|tfrac|frac)\{([-+\d.]+)\}\{([-+\d.]+)\}", r"\1/\2", text)
    if not re.fullmatch(r"[-+]?\d*\.?\d+(?:e[-+]?\d+)?(?:/[-+]?\d*\.?\d+)?", text):
        return None
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def score(row, response):
    answer, extraction = final_answer(response)
    result = {"extracted": answer, "extraction": extraction, "correct": False, "parse_success": bool(answer)}
    if not answer:
        return result
    gold = str(row["answer"])
    if row["task"] == "mcq":
        # Accept one unambiguous leading letter, or the complete option text.
        match = re.fullmatch(r"\s*\(?([A-J])\)?[.。]?\s*", answer, flags=re.I)
        if match is None:
            match = re.match(r"^\(?([A-J])\)?\s*[:.)：]\s+", answer, flags=re.I)
        predicted = match.group(1).upper() if match else None
        if predicted is None:
            matches = [k for k, value in row["choices"].items() if canonical(value) == canonical(answer)]
            predicted = matches[0] if len(matches) == 1 else None
        result.update(correct=predicted == gold.upper(), parse_success=predicted is not None, predicted=predicted)
    elif row["task"] == "yesno":
        match = re.match(r"^(yes|no)\b", answer, flags=re.I)
        result.update(correct=bool(match and match.group(1).lower() == gold.lower()), parse_success=bool(match))
    elif row["task"] == "grounding":
        try:
            bbox = json.loads(answer)
            target = row["bbox_xyxy_1000"]
            if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in bbox):
                raise ValueError("Expected a list of four coordinates")
            if not all(0 <= x <= 1000 for x in bbox) or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                raise ValueError("Invalid normalized bounding box")
            inter = max(0, min(bbox[2], target[2]) - max(bbox[0], target[0])) * max(0, min(bbox[3], target[3]) - max(bbox[1], target[1]))
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + (target[2] - target[0]) * (target[3] - target[1]) - inter
            iou = inter / area if area > 0 else 0.0
            result.update(correct=iou >= 0.5, iou=iou)
        except (ValueError, TypeError, KeyError):
            result.update(parse_success=False, iou=0.0)
    else:
        a, b = number(answer), number(gold)
        equal = canonical(answer) == canonical(gold)
        if a is not None and b is not None:
            equal = abs(a - b) <= max(1e-8, 1e-8 * abs(b))
        result["correct"] = equal
    return result


def score_prediction(row, record):
    """Recompute a saved generation under the current scoring-only protocol."""
    if record.get("error"):
        return {**record, "correct": False, "parse_success": False,
                "length_terminated": record.get("finish_reason") == "length",
                "scoring_protocol_sha256": digest(SCORING_PROTOCOL)}
    content = record.get("raw_response") or ""
    if PROTOCOL["enable_thinking"] and record.get("reasoning") is None and "</think>" not in content:
        content = "<think>" + content
    assessed = score(row, content)
    assessed["correct_before_length_penalty"] = assessed["correct"]
    assessed["length_terminated"] = record.get("finish_reason") == "length"
    if assessed["length_terminated"]:
        assessed["correct"] = False
        if "iou" in assessed:
            assessed["iou_before_length_penalty"] = assessed["iou"]
            assessed["iou"] = 0.0
    assessed["scoring_protocol_sha256"] = digest(SCORING_PROTOCOL)
    return {**record, **assessed}
