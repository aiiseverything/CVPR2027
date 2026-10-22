#!/usr/bin/env python3
"""Merge the main 3423-row suite and the BLINK/MMVP suite into one report.

Both suites share the frozen protocol and scorer, so per-benchmark results can
live in one table. A model is only scored on a benchmark when every row of that
benchmark has a successful prediction. The original main-suite summary is
overwritten; per-suite run.json/scoring.json files remain the provenance.
"""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import HERE, MODEL_NAMES, SCORING_PROTOCOL, digest, file_sha, load_suite, read_jsonl, score_prediction
from summarize import estimate, units

TARGET_MODELS = ["baseline", "perception-step100", "reasoning-step100"]
SUITES = [
    ("main", HERE / "suite.jsonl", HERE / "results"),
    ("blink_mmvp", HERE / "suite_blink_mmvp.jsonl", HERE / "results_blink_mmvp"),
]


def write_scored(directory, pred):
    scored_path = directory / "predictions.scored.jsonl"
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, prefix=".scored-", suffix=".tmp", delete=False) as output:
        output.write("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in pred.values()))
        temporary = Path(output.name)
    temporary.replace(scored_path)
    (directory / "scoring.json").write_text(json.dumps({
        "scoring_protocol": SCORING_PROTOCOL, "scoring_protocol_sha256": digest(SCORING_PROTOCOL),
        "scorer_source_sha256": file_sha(HERE / "common.py"), "n_scored": len(pred),
        "raw_source": str(directory / "predictions.jsonl"), "scored_file": str(scored_path),
    }, indent=2) + "\n")


def load_suite_predictions(suite_path, root, model):
    directory = root / "full" / model
    run_path = directory / "run.json"
    if not run_path.is_file():
        return None
    rows, meta = load_suite(suite_path)
    run = json.loads(run_path.read_text())
    expected = {"suite_sha256": meta["suite_sha256"], "protocol_sha256": meta["protocol_sha256"], "name": model}
    if any(run.get(key) != value for key, value in expected.items()) or run.get("smoke"):
        raise SystemExit(f"Incompatible run for {suite_path.name}/{model}")
    pred = {record["id"]: record for record in read_jsonl(directory / "predictions.jsonl")}
    for row in rows:
        record = pred.get(row["id"])
        if record is None:
            continue
        if record.get("row_sha256") != digest(row):
            raise SystemExit(f"Prediction row hash mismatch: {model}, {row['id']}")
        pred[row["id"]] = score_prediction(row, record)
    write_scored(directory, pred)
    return {"rows": rows, "meta": meta, "pred": pred, "directory": directory}


def suite_status(rows, pred):
    missing = sum(row["id"] not in pred for row in rows)
    errors = sum(bool(pred.get(row["id"], {}).get("error")) for row in rows)
    return {
        "status": "not_started" if not pred else "complete" if not missing and not errors else "incomplete",
        "n_expected": len(rows), "n_predicted": len(pred), "missing": missing, "api_errors": errors,
        "extraction_failures": sum(not p.get("parse_success", False) for p in pred.values() if not p.get("error")),
        "length_terminated": sum(p.get("finish_reason") == "length" for p in pred.values()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=HERE / "results")
    args = ap.parse_args()

    suites = {}
    for name, suite_path, root in SUITES:
        rows, meta = load_suite(suite_path)
        suites[name] = {"rows": rows, "meta": meta, "root": root, "preds": {}}
        for model in MODEL_NAMES:
            loaded = load_suite_predictions(suite_path, root, model)
            suites[name]["preds"][model] = loaded["pred"] if loaded else None

    ids = [row["id"] for suite in suites.values() for row in suite["rows"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate IDs across suites")
    bench_suite, bench_rows = {}, defaultdict(list)
    for name, suite in suites.items():
        for row in suite["rows"]:
            if row["benchmark"] in bench_suite and bench_suite[row["benchmark"]] != name:
                raise SystemExit(f"Benchmark split across suites: {row['benchmark']}")
            bench_suite[row["benchmark"]] = name
            bench_rows[row["benchmark"]].append(row)

    status, predictions = {}, {}
    for model in MODEL_NAMES:
        predictions[model] = {}
        status[model] = {}
        for name, suite in suites.items():
            pred = suite["preds"][model]
            status[model][name] = (suite_status(suite["rows"], pred) if pred is not None
                                   else {"status": "not_started", "n_expected": len(suite["rows"]),
                                         "n_predicted": 0, "missing": len(suite["rows"])})
            if pred:
                predictions[model].update(pred)
        status[model]["combined"] = {
            "n_expected": sum(s["n_expected"] for s in status[model].values() if isinstance(s, dict) and "n_predicted" in s),
            "n_predicted": sum(s.get("n_predicted", 0) for s in status[model].values() if isinstance(s, dict)),
            "missing": sum(s.get("missing", 0) for s in status[model].values() if isinstance(s, dict)),
            "api_errors": sum(s.get("api_errors", 0) for s in status[model].values() if isinstance(s, dict)),
            "extraction_failures": sum(s.get("extraction_failures", 0) for s in status[model].values() if isinstance(s, dict)),
            "length_terminated": sum(s.get("length_terminated", 0) for s in status[model].values() if isinstance(s, dict)),
        }

    results, deltas = {}, {}
    for benchmark, rows in sorted(bench_rows.items()):
        suite_name = bench_suite[benchmark]
        results[benchmark], deltas[benchmark] = {}, {}
        for model in MODEL_NAMES:
            pred = suites[suite_name]["preds"][model]
            if pred is None or any(row["id"] not in pred or pred[row["id"]].get("error") for row in rows):
                continue
            accuracy, ci = estimate(units(rows, pred))
            results[benchmark][model] = {
                "score": accuracy, "ci95": ci, "n_rows": len(rows),
                "metric": "circular_group_accuracy" if benchmark == "MMBench_DEV_EN_V11" else "strict_accuracy",
            }
        for model in TARGET_MODELS:
            if "baseline" in results[benchmark] and model != "baseline" and model in results[benchmark]:
                delta, ci = estimate(units(rows, suites[suite_name]["preds"][model]),
                                     units(rows, suites[suite_name]["preds"]["baseline"]))
                deltas[benchmark][model] = {"delta_vs_baseline": delta, "ci95": ci}
        for model in ["perception-step100", "reasoning-step100"]:
            other = "reasoning-step100" if model == "perception-step100" else "perception-step100"
            if model in results[benchmark] and other in results[benchmark]:
                delta, ci = estimate(units(rows, suites[suite_name]["preds"][model]),
                                     units(rows, suites[suite_name]["preds"][other]))
                deltas[benchmark][f"{model}_vs_{other}"] = {"delta": delta, "ci95": ci}

    dimensions = {}
    for dimension in ["reasoning", "perception", "general"]:
        benches = sorted({row["benchmark"] for rows in bench_rows.values() for row in rows if row["dimension"] == dimension})
        dimensions[dimension] = {"benchmarks": benches, "models": {}, "common_benchmarks": [], "common_models": {}}
        for model in MODEL_NAMES:
            available = [b for b in benches if model in results[b]]
            if available:
                dimensions[dimension]["models"][model] = {
                    "score": float(np.mean([results[b][model]["score"] for b in available])),
                    "benchmarks": available,
                }
        common = [b for b in benches if all(m in results[b] for m in TARGET_MODELS)]
        dimensions[dimension]["common_benchmarks"] = common
        for model in TARGET_MODELS:
            if common and all(model in results[b] for b in common):
                dimensions[dimension]["common_models"][model] = float(np.mean([results[b][model]["score"] for b in common]))

    report = {
        "suites": {name: {"suite_sha256": suite["meta"]["suite_sha256"], "counts": suite["meta"]["counts"],
                          "protocol_sha256": suite["meta"]["protocol_sha256"]} for name, suite in suites.items()},
        "status": status, "benchmarks": results, "dimension_macro": dimensions,
        "paired_delta_vs_baseline": deltas, "scoring_protocol": SCORING_PROTOCOL,
        "scoring_protocol_sha256": digest(SCORING_PROTOCOL),
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    lines = [
        "# SharedExpert 三教师统一评测（主套件 + BLINK/MMVP）", "",
        "两个套件使用完全相同的冻结协议：thinking=True，temperature=0，response≤8192；多模态 prompt≤2048；finish_reason=length 一律记零。"
        f"主套件 3423 行，BLINK/MMVP 增量 2201 行（BLINK val 1901、MMVP 300），合计 5624 行。", "",
        "| 模型 | 主套件 3423 | BLINK/MMVP 2201 | 缺失 | API 错误 | 抽取失败 | 长度截断 |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_NAMES:
        s = status[model]
        def label(name):
            part = s[name]
            return part["status"] if part["status"] != "complete" else f"complete ({part['n_predicted']}/{part['n_expected']})"
        lines.append(f"| {model} | {label('main')} | {label('blink_mmvp')} | {s['combined']['missing']} | "
                     f"{s['combined']['api_errors']} | {s['combined']['extraction_failures']} | {s['combined']['length_terminated']} |")
    lines += ["", "只报告该基准所有行都成功预测的模型；baseline 主套件结果为远端合并的完整 3423 行结果。", "",
              "| 评测集 | 行数 | " + " | ".join(MODEL_NAMES) + " |", "|---|---:|" + "---:|" * len(MODEL_NAMES)]
    for benchmark, rows in sorted(bench_rows.items()):
        cells = []
        for model in MODEL_NAMES:
            r = results[benchmark].get(model)
            cells.append(f"{100*r['score']:.2f} ({100*r['ci95'][0]:.2f}–{100*r['ci95'][1]:.2f})" if r else "—")
        lines.append(f"| {benchmark} | {len(rows)} | " + " | ".join(cells) + " |")
    lines += ["", "MMBench 使用循环选项组全对；BLINK/MMVP 为严格 MCQ；RefCOCO 为 IoU≥0.5；其余为严格答案准确率。", "",
              "| 维度宏平均 | 基准集 | " + " | ".join(MODEL_NAMES) + " |", "|---|---|" + "---:|" * len(MODEL_NAMES)]
    for dimension, values in dimensions.items():
        cells = []
        for model in MODEL_NAMES:
            entry = values["models"].get(model)
            cells.append(f"{100*entry['score']:.2f}" if entry else "—")
        lines.append(f"| {dimension} | {', '.join(values['benchmarks'])} | " + " | ".join(cells) + " |")
    common = dimensions["perception"]["common_benchmarks"]
    if common and set(common) != set(dimensions["perception"]["benchmarks"]):
        lines.append(f"| perception（共同基准） | {', '.join(common)} | " +
                     " | ".join(f"{100*dimensions['perception']['common_models'][m]:.2f}"
                                if m in dimensions["perception"]["common_models"] else "—" for m in MODEL_NAMES) + " |")
    lines += ["", "维度宏平均是该模型已完整覆盖基准的等权均值；三模型当前均覆盖全部所列基准，可直接比较。", "",
              "对 baseline 的配对差值与 step100 之间的配对差值见 summary.json。", "",
              f"固定样本总数：{sum(len(s['rows']) for s in suites.values())}；协议与评分器 sha256 见 summary.json。",
              "逐题结果详见 per_sample.csv；未完整覆盖的基准不产生准确率。"]
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n")

    with (args.output_root / "per_sample.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "suite", "benchmark", "dimension", "category", "group"] + MODEL_NAMES)
        writer.writeheader()
        for name, suite in suites.items():
            for row in suite["rows"]:
                writer.writerow({
                    "id": row["id"], "suite": name, "benchmark": row["benchmark"], "dimension": row["dimension"],
                    "category": row["category"], "group": row["group"],
                    **{model: predictions[model].get(row["id"], {}).get("correct", "") for model in MODEL_NAMES},
                })
    print(json.dumps({"status": {m: status[m]["combined"] for m in MODEL_NAMES},
                      "report": str(args.output_root / "summary.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
