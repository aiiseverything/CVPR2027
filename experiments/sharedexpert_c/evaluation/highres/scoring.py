#!/usr/bin/env python3
"""Strict local high-resolution scoring; independent of the old evaluation.

This is deterministic option/numeric matching, NOT the official ZoomBench LLM
judge. Full answers are extracted only after a thinking boundary, from the last
explicit answer/box, or from an entire short unambiguous answer. HRBench averages
accuracy across option cycles; it does not require all cycles to be correct.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import unicodedata

try:
    from .protocol import DATA, RESULTS, MODEL_PATHS, PROTOCOL, atomic_json, digest, file_sha, load_suite
    from .annotation_issues import known_annotation_issue, known_missing_gold
except ImportError:
    from protocol import DATA, RESULTS, MODEL_PATHS, PROTOCOL, atomic_json, digest, file_sha, load_suite
    from annotation_issues import known_annotation_issue, known_missing_gold

SCORING_PROTOCOL = {
    "version": "sharedexpert-highres-strict-local-v1",
    "official_llm_judge": False,
    "empty_or_api_error_score": 0,
    "length_termination_score": 0,
    "accepted_finish_reason": "stop",
    "extraction": "final content after last </think>; last complete answer tag, else last balanced boxed, else whole short answer",
    "mcq": "one option letter only; never scan prose or thought for letters; preserve duplicate source labels",
    "numeric": "whole numeric expression only, exact rational equality; no substring, tolerance, or external judge",
    "hrbench": "mean of per-cycle row accuracy; category breakdown averaged across its cycles",
    "other_benchmarks": "row micro accuracy; original metadata groups retained",
    "completeness": "full dataset accuracy only with all expected rows, zero API errors, compatible run, and no incomplete JSONL tail",
    "duplicate_predictions": "last complete JSONL record for each ID wins, including a later error",
    "known_invalid_annotation": "Eleven exactly identified upstream label anomalies (one Zoom, two MME EN, eight MME CN) retain original questions, option labels and gold in primary accuracy; a separately labelled sensitivity excludes them. No label correction or row removal.",
}
SCORING_PROTOCOL_SHA256 = digest(SCORING_PROTOCOL)
ANSWER_OPEN = re.compile(r"<answer\s*>", re.I)
ANSWER_TAG = re.compile(r"<answer\s*>(.*?)</answer\s*>", re.I | re.S)
BOX_OPEN = re.compile(r"\\boxed\s*\{")
PREFIX = re.compile(r"^(?:(?:final\s+)?answer\s*[:：]|the\s+answer\s+is\s*[:：]?|(?:最终)?答案\s*[:：])\s*", re.I)
DECIMAL = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
KNOWN_ZOOM_INVALID_ID = "2e0304061d079575c689b86916217248"


def known_invalid_annotation(row):
    metadata = row.get("metadata", {})
    return (metadata.get("answer_label_missing_from_options") is True
            and known_missing_gold(row))


def source_annotation_issue(row):
    return known_annotation_issue(row)


def _unwrap(text):
    text = unicodedata.normalize("NFKC", text).strip()
    text = PREFIX.sub("", text, count=1).strip()
    for opening, closing in [("**", "**"), ("$", "$"), (r"\(", r"\)")]:
        if text.startswith(opening) and text.endswith(closing) and len(text) > len(opening) + len(closing):
            text = text[len(opening):-len(closing)].strip()
    wrapped = re.fullmatch(r"\\(?:text|mathrm|mathbf)\{([^{}]*)\}", text)
    if wrapped:
        text = wrapped.group(1).strip()
    return text


def extract_final(text):
    if not isinstance(text, str) or not text.strip():
        return "", "empty_final_content"
    boundaries = list(re.finditer(r"</think\s*>", text, re.I))
    if boundaries:
        text = text[boundaries[-1].end():]
    if re.search(r"<think\s*>", text, re.I):
        return "", "unfinished_thinking"
    if not text.strip():
        return "", "empty_final_content"
    starts = list(ANSWER_OPEN.finditer(text))
    if starts:
        matches = list(ANSWER_TAG.finditer(text))
        if not matches or matches[-1].start() != starts[-1].start():
            return "", "unfinished_answer_tag"
        return matches[-1].group(1).strip(), "answer_tag"
    boxes = list(BOX_OPEN.finditer(text))
    if boxes:
        start = boxes[-1].end()
        depth = 1
        for end in range(start, len(text)):
            depth += (text[end] == "{") - (text[end] == "}")
            if depth == 0:
                return text[start:end].strip(), "boxed"
        return "", "unfinished_boxed_answer"
    # No last-line fallback: a trailing option in an unmarked long thought is
    # not evidence of a final answer. The complete string must parse below.
    return text.strip(), "whole_answer"


def numeric_value(text):
    text = _unwrap(text).replace("−", "-")
    text = text.replace(r"\left", "").replace(r"\right", "").replace(r"\,", "")
    fraction = re.fullmatch(r"\\(?:frac|dfrac|tfrac)\{(" + DECIMAL + r")\}\{(" + DECIMAL + r")\}", text)
    if fraction:
        text = fraction.group(1) + "/" + fraction.group(2)
    # Permit only properly grouped thousands, never discard arbitrary commas.
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?", text):
        text = text.replace(",", "")
    if text.endswith("。"):
        text = text[:-1].rstrip()
    if len(text) > 128 or not re.fullmatch(DECIMAL + r"(?:\s*/\s*" + DECIMAL + r")?", text):
        return None
    # Avoid unbounded integer allocation from an adversarial scientific exponent.
    if any(abs(int(e)) > 1000 for e in re.findall(r"[eE]([+-]?\d+)", text)):
        return None
    try:
        pieces = text.split("/")
        value = Fraction(pieces[0].strip())
        return value if len(pieces) == 1 else value / Fraction(pieces[1].strip())
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def validate_row(row):
    options = row.get("options")
    if not isinstance(options, dict):
        raise ValueError(f"Missing options dictionary: {row.get('id')}")
    if row.get("metadata", {}).get("answer_label_missing_from_options") and not known_invalid_annotation(row):
        raise ValueError(f"Unrecognized source annotation exception: {row.get('id')}")
    if options:
        known_cn_labels = (known_missing_gold(row) and row.get("benchmark") == "mme_rw_cn"
                           and set(options) == {"一", "B", "C", "D", "E"})
        if any(not re.fullmatch(r"[A-Z]", key) for key in options) and not known_cn_labels:
            raise ValueError(f"Invalid original option label: {row.get('id')}")
        if row.get("answer") not in options and not known_invalid_annotation(row):
            raise ValueError(f"Gold label is absent from source options: {row.get('id')}")
    else:
        if row.get("benchmark") != "zoom" or row.get("metadata", {}).get("question_type") == "mcq":
            raise ValueError(f"Multiple-choice source has no parsed options: {row.get('id')}")
        if numeric_value(str(row.get("answer", ""))) is None:
            raise ValueError(f"Zoom numeric gold answer is not numeric: {row.get('id')}")
    metadata = row.get("metadata", {})
    required = ["cycle_category", "category"] if row["benchmark"].startswith("hrbench_") else ["category", "l2-category"] if row["benchmark"].startswith("mme_rw_") else []
    for key in required:
        if key not in metadata or metadata[key] is None:
            raise ValueError(f"Missing original {key}: {row.get('id')}")


def score_prediction(row, pred):
    """Return a copy with strict scoring fields; never alter either input."""
    validate_row(row)
    result = {**pred, "correct": False, "parse_success": False, "prediction": None,
              "reason": "api_error" if pred.get("error") else "unparsed",
              "length_terminated": pred.get("finish_reason") == "length",
              "correct_before_length_penalty": False,
              "scoring_protocol_sha256": SCORING_PROTOCOL_SHA256}
    if pred.get("error"):
        return result
    answer, extraction = extract_final(pred.get("raw_response") or "")
    result.update(extracted=answer, extraction=extraction)
    if answer:
        if row["options"]:
            candidate = _unwrap(answer)
            match = re.fullmatch(r"(?:([A-Z])|\(([A-Z])\)|\[([A-Z])\])[.。]?", candidate, re.I)
            value = next((x for x in match.groups() if x), "").upper() if match else None
            allowed = set(row["options"])
            if known_invalid_annotation(row):
                allowed.add(row["answer"])
            if value not in allowed:
                value = None
            result.update(prediction=value, parse_success=value is not None,
                          correct=value == row["answer"])
        else:
            value, gold = numeric_value(answer), numeric_value(row["answer"])
            result.update(prediction=str(value) if value is not None else None,
                          parse_success=value is not None, correct=value is not None and value == gold)
        result["reason"] = "correct" if result["correct"] else "incorrect" if result["parse_success"] else "ambiguous_or_invalid_answer"
    else:
        result["reason"] = extraction
    result["correct_before_length_penalty"] = result["correct"]
    if result["length_terminated"]:
        result.update(correct=False, reason="length_termination")
    elif pred.get("finish_reason") != "stop":
        result.update(correct=False, reason="non_stop_finish_reason")
    return result


def _read_predictions(path, identities, model):
    """Read a stable byte-prefix; last retry wins; never repair or mutate raw data."""
    data, count, duplicate, tail = {}, 0, 0, False
    if not path.exists():
        return data, {"records": 0, "duplicates": 0, "incomplete_tail": False, "snapshot_bytes": 0, "sha256": None}
    h = __import__("hashlib").sha256()
    with path.open("rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        while stream.tell() < size:
            line = stream.readline(size - stream.tell())
            h.update(line)
            if not line.strip():
                continue
            try:
                p = json.loads(line)
            except (ValueError, UnicodeError):
                if stream.tell() == size and not line.endswith(b"\n"):
                    tail = True
                    break
                raise ValueError(f"Corrupt prediction JSONL: {path}, record {count + 1}") from None
            if not isinstance(p, dict) or p.get("id") not in identities:
                raise ValueError(f"Unknown prediction ID in {path}")
            if p.get("model") != model or p.get("row_sha256") != identities[p["id"]]:
                raise ValueError(f"Prediction provenance mismatch: {path}, {p['id']}")
            if not p.get("error") and not isinstance(p.get("raw_response"), str):
                raise ValueError(f"Successful prediction lacks raw_response: {path}, {p['id']}")
            duplicate += p["id"] in data
            data[p["id"]] = p
            count += 1
    return data, {"records": count, "duplicates": duplicate, "incomplete_tail": tail,
                  "snapshot_bytes": size, "sha256": h.hexdigest()}


def _stats(rows, scored, input_ok=True):
    ps = [scored[r["id"]] for r in rows if r["id"] in scored]
    n, have = len(rows), len(ps)
    errors = sum(bool(p.get("error")) for p in ps)
    complete = have == n and errors == 0 and input_ok
    correct = sum(bool(p["correct"]) for p in ps)
    return {"status": "complete" if complete else "partial" if have else "not_started",
            "n_expected": n, "n_predicted": have, "missing": n - have, "api_errors": errors,
            "correct": correct, "accuracy": correct / n if complete else None,
            "observed_accuracy": correct / have if have else None,
            "length_terminated": sum(p.get("finish_reason") == "length" for p in ps),
            "parse_failures": sum(not p["parse_success"] for p in ps if not p.get("error")),
            "finish_reasons": dict(Counter(str(p.get("finish_reason")) for p in ps)),
            "mean_completion_tokens": sum((p.get("usage") or {}).get("completion_tokens", 0) or 0 for p in ps) / have if have else None}


def _groups(rows, scored, key, input_ok):
    groups = defaultdict(list)
    for row in rows:
        if key in row.get("metadata", {}):
            groups[str(row["metadata"][key])].append(row)
    return {name: _stats(rs, scored, input_ok) for name, rs in sorted(groups.items())}


def summarize(root=DATA, results=RESULTS):
    """Write independent highres summary.json/summary.csv; return the report."""
    root, results = Path(root), Path(results)
    rows, meta = load_suite(root)
    by_bench = defaultdict(list)
    for row in rows:
        validate_row(row)
        by_bench[row["benchmark"]].append(row)
    identities = {r["id"]: digest(r) for r in rows}
    report = {"created_at": datetime.now(timezone.utc).isoformat(),
              "suite_sha256": meta["suite_sha256"], "protocol_sha256": digest(PROTOCOL),
              "selection_sha256": meta["selection_sha256"],
              "selection": meta.get("selection"), "selected_counts": meta["counts"],
              "inference_protocol": PROTOCOL, "scoring_protocol": SCORING_PROTOCOL,
              "scoring_protocol_sha256": SCORING_PROTOCOL_SHA256,
              "scorer_source_sha256": file_sha(Path(__file__)),
              "score_scale": "fractions in [0,1]", "models": {},
              "limitations": ["Strict local deterministic scores, not official LLM-judge scores.",
                  "Perception subsets only: all selected rows retained, including repeated MME-RealWorld question IDs and duplicate option labels. These are not full-benchmark scores.",
                  "HRBench cycle-average accuracy, not all-cycles-correct accuracy.",
                  "Exactly identified upstream label anomalies remain unchanged in primary scores; the separately named sensitivity excludes these eleven source rows.",
                  "A partial observed_accuracy describes available predictions only and is not full-dataset accuracy.",
                  "Full image only, including ZoomBench; no privileged crop-view result or zooming gap."]}
    csvrows = []
    for model in MODEL_PATHS:
        directory = results / "full" / model
        runpath = directory / "run.json"
        run_ok = runpath.exists()
        if run_ok:
            run = json.loads(runpath.read_text())
            expected = {"name": model, "suite_sha256": meta["suite_sha256"],
                        "selection_sha256": meta["selection_sha256"],
                        "protocol_sha256": digest(PROTOCOL), "protocol": PROTOCOL,
                        "n_requested": len(rows), "smoke": False}
            if any(run.get(k) != v for k, v in expected.items()):
                raise ValueError(f"Incompatible highres run metadata: {runpath}")
            if run.get("source_manifest_sha256") != meta.get("source_manifest_sha256"):
                raise ValueError(f"Source manifest mismatch: {runpath}")
        pred, snapshot = _read_predictions(directory / "predictions.jsonl", identities, model)
        scored = {r["id"]: score_prediction(r, pred[r["id"]]) for r in rows if r["id"] in pred}
        input_ok = run_ok and not snapshot["incomplete_tail"]
        result = {**_stats(rows, scored, input_ok), "input_snapshot": snapshot,
                  "run_metadata_present": run_ok, "datasets": {}}
        # Overall is a completion status only: different datasets have different
        # official denominators, so do not advertise a cross-dataset accuracy.
        result.pop("accuracy")
        result.pop("observed_accuracy")
        for bench, br in sorted(by_bench.items()):
            item = _stats(br, scored, input_ok)
            item["metric"] = "strict_local_micro_accuracy"
            item["groups"] = {key: _groups(br, scored, key, input_ok)
                              for key in ["category", "l2-category", "question_type", "cycle_category"]
                              if any(key in r.get("metadata", {}) for r in br)}
            invalid = [r for r in br if source_annotation_issue(r)]
            item["source_anomalies"] = [{"id": r["id"], "source_id": r["metadata"]["original_id"],
                "kind": source_annotation_issue(r), "original_gold": r["answer"],
                "original_option_labels": list(r["options"]), "primary_score_policy": "retain original gold and denominator"}
                for r in invalid]
            if invalid:
                keep = [r for r in br if not source_annotation_issue(r)]
                item["exclude_known_invalid_annotations"] = {
                    **_stats(keep, scored, input_ok), "metric": "strict_local_micro_accuracy",
                    "excluded_ids": [r["id"] for r in invalid],
                    "interpretation": "Auxiliary sensitivity excluding exactly identified upstream label anomalies; primary score remains unchanged."}
            if bench.startswith("hrbench_"):
                item["metric"] = "strict_local_cycle_macro_accuracy"
                item["micro_accuracy"] = item["accuracy"]
                cycles = item["groups"]["cycle_category"]
                item["accuracy"] = sum(x["accuracy"] for x in cycles.values()) / len(cycles) if item["status"] == "complete" else None
                nested = defaultdict(list)
                for r in br:
                    nested[str(r["metadata"]["category"])].append(r)
                item["category_cycle_macro"] = {}
                for category, cr in sorted(nested.items()):
                    categories = _stats(cr, scored, input_ok)
                    cc = _groups(cr, scored, "cycle_category", input_ok)
                    categories["micro_accuracy"] = categories["accuracy"]
                    categories["accuracy"] = sum(x["accuracy"] for x in cc.values()) / len(cc) if categories["status"] == "complete" else None
                    categories["cycles"] = cc
                    item["category_cycle_macro"][category] = categories
            if bench.startswith("mme_rw_"):
                nested = defaultdict(lambda: defaultdict(list))
                for r in br:
                    md = r["metadata"]
                    nested[str(md["category"])][str(md["l2-category"])].append(r)
                item["category_l2"] = {c: {l2: _stats(rs, scored, input_ok) for l2, rs in sorted(v.items())}
                                       for c, v in sorted(nested.items())}
                item["rows_with_duplicate_option_labels"] = sum(bool(r["metadata"].get("option_labels_duplicated")) for r in br)
            result["datasets"][bench] = item
            csvrows.append({"model": model, "dataset": bench, "status": item["status"],
                "metric": item["metric"], **{k: item[k] for k in ["n_expected", "n_predicted", "missing", "api_errors", "correct", "accuracy", "observed_accuracy", "length_terminated", "parse_failures", "mean_completion_tokens"]}})
        report["models"][model] = result
    results.mkdir(parents=True, exist_ok=True)
    csvpath = results / "summary.csv"
    temporary = csvpath.with_name(csvpath.name + f".{os.getpid()}.tmp")
    with temporary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csvrows[0]))
        w.writeheader()
        w.writerows(csvrows)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(csvpath)
    report["summary_csv_sha256"] = file_sha(csvpath)
    atomic_json(results / "summary.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=DATA)
    ap.add_argument("--results", "--output-root", dest="results", type=Path, default=RESULTS)
    args = ap.parse_args()
    report = summarize(args.root, args.results)
    print(json.dumps({"summary": str(args.results / "summary.json"),
                      "status": {m: x["status"] for m, x in report["models"].items()}}))


if __name__ == "__main__":
    main()
