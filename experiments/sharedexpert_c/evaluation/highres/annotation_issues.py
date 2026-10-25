"""Exact exceptions for eleven inspected, pinned upstream annotation rows.

This module does not correct labels, answers, questions, or source option lines.
Matching requires the full text and source identity; metadata flags alone never
authorize an exception. Existing prepared rows need no mutation to be detected.
"""
from __future__ import annotations


def _source(repo, revision, path, sha256, row_index):
    return dict(repo=repo, revision=revision, path=path, sha256=sha256, row_index=row_index)


def _record(benchmark, ordinal, original_id, question, answer, lines, source, issue, missing, options):
    return {"id": f"{benchmark}:{ordinal:06d}", "benchmark": benchmark,
            "original_id": original_id, "question": question, "answer": answer,
            "option_lines": lines, "source": source, "issue_type": issue,
            "missing_gold": missing, "options": options}


_ZOOM_LINES = ["A. no", "B. yes\n\nAnswer with the option's letter from the given choices."]
_records = [_record(
    "zoom", 477, "2e0304061d079575c689b86916217248",
    "Is there a person visible on the sailboat?\n\nA. no\nB. yes\n\nAnswer with the option's letter from the given choices.",
    "D", _ZOOM_LINES,
    _source("inclusionAI/ZoomBench", "b788097e57d30510c6877824833234a73bf80d25",
            "data/test.parquet", "d44ebda2eda485cba055181f4e6dc50c42f81b5d0f7e936bf427fa01502a391a", 477),
    "missing_gold_label", True,
    {"A": "no", "B": "yes\n\nAnswer with the option's letter from the given choices."})]

for ordinal, original_id, noun, gold, shard, row_index, sha in [
    (19452, 12906, "truck", "D", 57, 224, "17083eb2f0dbcb1cf825ef7bf49db89fab8aaeb9eea7a838e2a56cccfefd1f2c"),
    (19625, 13080, "bicycle", "A", 58, 60, "49a6d4d786d01ba057a0dc861e13e5f79ed3a358f9dd4709f83af60c022f9765"),
]:
    lines = ["(C) The right of the image", "(B) The bottom of the image",
             "(C) The left of the image", "(D) The top of the image",
             f"(E) The image does not feature the {noun}"]
    _records.append(_record(
        "mme_rw_en", ordinal, original_id, f"What is the orientation of the {noun} in the image?",
        gold, lines,
        _source("yifanzhang114/MME-RealWorld-Lmms-eval", "f157390129e8d2b60757eb68d340aade9446b52b",
                f"data/train-{shard:05d}-of-00070.parquet", sha, row_index),
        "duplicate_option_label" if gold == "D" else "missing_gold_duplicate_label", gold == "A",
        {"C": "The right of the image", "B": "The bottom of the image",
         "D": "The top of the image", "E": f"The image does not feature the {noun}"}))

for ordinal, original_id, row_index, noun, location in [
    (5119, 5121, 112, "三轮车", "右中心"),
    (5126, 5126, 119, "遮阳篷三轮车", "左中心"),
    (5133, 5134, 126, "摩托车", "右中心"),
    (5149, 5148, 142, "摩托车", "右中心"),
    (5206, 5207, 199, "公交车", "右中心"),
    (5207, 5209, 200, "汽车", "左中心"),
    (5211, 5211, 204, "汽车", "左中心"),
    (5217, 5215, 210, "卡车", "左中心"),
]:
    _records.append(_record(
        "mme_rw_cn", ordinal, original_id, f"图像中的{noun}在哪里？", "A",
        [f"（一）{location}", "(B)左上角", "(C)左下角", "(D)右上角", f"(E)图像中没有{noun}"],
        _source("yifanzhang114/MME-RealWorld-CN-Lmms-eval", "37ac6f4e57ef211b37d1ee04a8f1268f1411b3e0",
                "data/train-00011-of-00013.parquet", "dcd19ed6a94c00089ea6ea10781fd8a7fcdff78626f768f96700d19cd06d9184", row_index),
        "non_latin_label_missing_gold", True,
        {"一": location, "B": "左上角", "C": "左下角", "D": "右上角", "E": f"图像中没有{noun}"}))

KNOWN_ANNOTATION_ISSUES = tuple(_records)
_BY_ID = {record["id"]: record for record in KNOWN_ANNOTATION_ISSUES}


def _matched_record(row, *, check_options=True):
    if not isinstance(row, dict):
        return None
    record = _BY_ID.get(row.get("id"))
    metadata = row.get("metadata", {})
    if record is None or not isinstance(metadata, dict):
        return None
    for key in ("id", "benchmark", "question", "answer"):
        if row.get(key) != record[key]:
            return None
    original_id = metadata.get("original_id")
    if type(original_id) is not type(record["original_id"]) or original_id != record["original_id"]:
        return None
    if metadata.get("option_lines") != record["option_lines"]:
        return None
    source = metadata.get("source")
    if not isinstance(source, dict) or any(source.get(k) != v for k, v in record["source"].items()):
        return None
    if record["benchmark"] == "zoom" and metadata.get("question_type") != "mcq":
        return None
    if check_options and row.get("options") != record["options"]:
        return None
    return record


def known_annotation_issue(row) -> str | None:
    """Identify only an exact inspected row, regardless of advisory flags."""
    record = _matched_record(row)
    return record["issue_type"] if record else None


def known_missing_gold(row) -> bool:
    """Whether this exact inspected row has a gold label absent from options."""
    record = _matched_record(row)
    return bool(record and record["missing_gold"])


def known_non_latin_option_source(row) -> bool:
    """Pre-parser identity gate; all original text/source fields must match."""
    record = _matched_record(row, check_options=False)
    return bool(record and record["issue_type"] == "non_latin_label_missing_gold")
