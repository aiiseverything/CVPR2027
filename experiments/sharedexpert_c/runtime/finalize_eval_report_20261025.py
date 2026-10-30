#!/usr/bin/env python3
"""Verify stable scores and export all completed models, with IFEval sensitivity."""
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

RUNTIME = Path(__file__).resolve().parent
EVAL = RUNTIME.parent / "evaluation"
OUT = EVAL / "results_expanded"
AUDIT = RUNTIME / "ifeval_scoring_stabilization_20261025"
MODELS = ["baseline", "baseline-9b", "perception-step100", "reasoning-step100", "generalist", "perception-v2-step100"]
LABELS = ["基座4B", "基座9B（历史）", "感知教师v1", "推理教师", "通用教师", "感知教师v2"]
AFFECTED = {"IFEval:1122", "IFEval:1129"}


def records(path):
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def main():
    report = json.loads((OUT / "summary.json").read_text())
    general = json.loads((EVAL / "results_general/summary.json").read_text())
    original = json.loads((AUDIT / "audit.json").read_text())
    if report["scoring_protocol"]["version"] == "strict-local-v2-length-zero":
        raise SystemExit("Run revised CPU summaries before finalizing")
    assert hashlib.sha256((EVAL / "suite_general.jsonl").read_bytes()).hexdigest() == original["frozen_suite_sha256"]
    sensitivity = {"affected_ids": sorted(AFFECTED), "full_n": 541, "sensitivity_n": 539,
                   "official_checker_limitation": "LetterFrequencyChecker replaces # and ! with random ASCII letters. The wrapper freezes that fallback, not the intended punctuation semantics.",
                   "max_effect_percentage_points": 200 / 541, "models": {}}
    for model in original["models"]:
        raw_path = EVAL / "results_general/full" / model / "predictions.jsonl"
        assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == original["raw_predictions_sha256"][model]
        assert report["status"][model]["combined"]["n_predicted"] == 7162
        assert report["status"][model]["combined"]["missing"] == report["status"][model]["combined"]["api_errors"] == 0
        assert general["general_details"][model] == report["general_details"][model], "Repeated CPU runs must agree on all IF metrics"
        for benchmark in ["OCRBench_heldout", "IFEval_strict"]:
            assert general["benchmarks"][benchmark][model]["score"] == report["benchmarks"][benchmark][model]["score"]
        pred = records(raw_path.with_name("predictions.scored.jsonl"))
        valid = [v for k, v in pred.items() if k.startswith("IFEval:") and k not in AFFECTED]
        assert len(valid) == 539
        correct = sum(bool(v["correct"]) for v in valid)
        before = original["scores_before"][model]
        assert all(before[k]["correct"] == v["correct"] for k, v in pred.items() if k.startswith("IFEval:") and k not in AFFECTED)
        sensitivity["models"][model] = {
            "full_541_accuracy": report["benchmarks"]["IFEval_strict"][model]["score"],
            "without_2_defective_rows_accuracy": correct / 539,
            "full_541_bounds_from_defective_rows": [correct / 541, (correct + 2) / 541],
            "affected_rows": {k: {"before": before[k], "after": {key: pred[k].get(key) for key in ["correct", "loose_correct", "strict_instructions", "loose_instructions", "ifeval_random_seed", "finish_reason"]}} for k in sorted(AFFECTED)},
        }
    (OUT / "ifeval_sensitivity.json").write_text(json.dumps(sensitivity, ensure_ascii=False, indent=2) + "\n")
    rows = []

    def add(kind, metric, count, scores):
        rows.append({"范围": kind, "指标": metric, "题数": count,
                     **{LABELS[i]: f"{100*scores[model]:.2f}" if model in scores else "" for i, model in enumerate(MODELS)}})

    for benchmark, values in report["benchmarks"].items():
        add("基准主指标（%）", benchmark, next(iter(values.values()))["n_rows"], {m: v["score"] for m, v in values.items()})
    for dimension, values in report["dimension_macro"].items():
        add("同基准集合等权宏平均（%）", dimension, "", {m: v["score"] for m, v in values["models"].items() if set(v["benchmarks"]) == set(values["benchmarks"])})
    details = report["general_details"]
    for category in details["baseline"]["ocr_categories"]:
        add("OCRBench内部类别（%）", category, details["baseline"]["ocr_categories"][category]["n"],
            {m: v["ocr_categories"][category]["accuracy"] for m, v in details.items()})
    for metric in ["strict_instruction_accuracy", "loose_prompt_accuracy", "loose_instruction_accuracy"]:
        add("IFEval次级指标（%）", metric, "", {m: v["ifeval"][metric] for m, v in details.items()})
    add("IFEval敏感性分析（%）", "排除两条官方checker异常题", 539, {m: v["without_2_defective_rows_accuracy"] for m, v in sensitivity["models"].items()})
    blink_rows = list(records(EVAL / "suite_blink_mmvp.jsonl").values())
    blink_predictions = {m: records(EVAL / "results_blink_mmvp/full" / m / "predictions.scored.jsonl") for m in original["models"]}
    for category in sorted({r["category"] for r in blink_rows if r["benchmark"] == "BLINK_val"}):
        ids = [r["id"] for r in blink_rows if r["benchmark"] == "BLINK_val" and r["category"] == category]
        add("BLINK内部类别（%）", category, len(ids), {m: sum(p[k]["correct"] for k in ids) / len(ids) for m, p in blink_predictions.items()})
    with (OUT / "all_model_scores.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["范围", "指标", "题数"] + LABELS)
        writer.writeheader()
        writer.writerows(rows)
    receipt = {"stage": "complete", "finished_at": datetime.now(timezone.utc).isoformat(),
               "scoring_protocol": report["scoring_protocol"], "scoring_protocol_sha256": report["scoring_protocol_sha256"],
               "general_scorer_source_sha256": report["general_scorer_source_sha256"],
               "raw_predictions_unchanged": True, "frozen_suite_unchanged": True,
               "independent_general_and_merged_scores_equal": True, "nondefective_if_strict_scores_unchanged": True,
               "complete_4b_models": original["models"], "predictions_per_model": 7162,
               "full_report": str(OUT / "summary.md"), "csv": str(OUT / "all_model_scores.csv"),
               "sensitivity": str(OUT / "ifeval_sensitivity.json"), "exported_metric_rows": len(rows)}
    (AUDIT / "receipt.json").write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    state_path = RUNTIME / "eval_extensions_20261025.json"
    state = json.loads(state_path.read_text())
    state.update(scoring_stabilization=str(AUDIT / "receipt.json"), updated_at=receipt["finished_at"])
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
