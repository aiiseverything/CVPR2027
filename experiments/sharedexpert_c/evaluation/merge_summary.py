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
from general_metrics import metric_name

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
        "scorer_source_sha256": file_sha(HERE / "common.py"),
        "general_scorer_source_sha256": file_sha(HERE / "general_metrics.py"), "n_scored": len(pred),
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


def render_markdown(suites, status, results, dimensions, bench_rows, model_names):
    """Render selected suites and compare only complete, identical benchmark sets."""
    suite_labels = {"main": "主套件", "blink_mmvp": "BLINK/MMVP", "general": "OCR/IF"}
    names = list(suites)
    labels = {name: suite_labels.get(name, name) for name in names}
    total = sum(len(suite["rows"]) for suite in suites.values())
    counts = "；".join(f"{labels[name]} {len(suites[name]['rows'])} 行" for name in names)
    lines = [
        "# SharedExpert 统一评测（" + " + ".join(labels[name] for name in names) + "）", "",
        f"{len(names)} 个套件沿用冻结生成协议：thinking=True，temperature=0，response≤8192；"
        "多模态 prompt≤2048；finish_reason=length 一律记零。",
        f"{counts}；合计 {total} 行。", "",
        "| 模型 | " + " | ".join(f"{labels[name]} {len(suites[name]['rows'])}" for name in names)
        + " | 缺失 | API 错误 | 抽取失败 | 长度截断 |",
        "|---|" + "---|" * len(names) + "---:|---:|---:|---:|",
    ]
    for model in model_names:
        selected = status[model]
        cells = []
        for name in names:
            part = selected[name]
            cells.append(part["status"] if part["status"] != "complete" else
                         f"complete ({part['n_predicted']}/{part['n_expected']})")
        combined = selected["combined"]
        lines.append(f"| {model} | " + " | ".join(cells) +
                     f" | {combined['missing']} | {combined['api_errors']} | "
                     f"{combined['extraction_failures']} | {combined['length_terminated']} |")
    lines += ["", "只报告该基准所有行都成功预测的模型；baseline 主套件结果为远端合并的完整 3423 行结果。", "",
              "| 评测集 | 行数 | " + " | ".join(model_names) + " |",
              "|---|---:|" + "---:|" * len(model_names)]
    for benchmark, rows in sorted(bench_rows.items()):
        cells = []
        for model in model_names:
            result = results[benchmark].get(model)
            cells.append(f"{100*result['score']:.2f} ({100*result['ci95'][0]:.2f}–{100*result['ci95'][1]:.2f})"
                         if result else "—")
        lines.append(f"| {benchmark} | {len(rows)} | " + " | ".join(cells) + " |")
    metric_note = "MMBench 使用循环选项组全对；BLINK/MMVP 为严格 MCQ；RefCOCO 为 IoU≥0.5。"
    if "general" in suites:
        metric_note += "OCRBench 使用归一化答案包含匹配；IFEval 使用官方 checker 的 strict prompt-level accuracy。"
    metric_note += "其他任务按各自冻结的本地评分规则计分。"
    lines += ["", metric_note, "",
              "| 维度宏平均 | 基准集 | " + " | ".join(model_names) + " |",
              "|---|---|" + "---:|" * len(model_names)]
    for dimension, values in dimensions.items():
        benchmarks = values["benchmarks"]
        cells = []
        for model in model_names:
            entry = values["models"].get(model)
            covered = set(entry["benchmarks"]) & set(benchmarks) if entry else set()
            missing = [benchmark for benchmark in benchmarks if benchmark not in covered]
            if entry and benchmarks and not missing:
                cells.append(f"{100*entry['score']:.2f}")
            elif missing:
                absent = "全部" if not covered else "、".join(missing)
                cells.append(f"缺项（{len(covered)}/{len(benchmarks)}；缺{absent}）")
            else:
                cells.append("—")
        lines.append(f"| {dimension} | {', '.join(benchmarks)} | " + " | ".join(cells) + " |")
    lines += ["", "维度宏平均仅在模型完整覆盖该行列出的全部基准时显示；缺项单元格注明已覆盖／总基准数。",
              "不同基准集合的均值不可直接比较。JSON 为兼容既有数据，保留已覆盖子集的均值及其 benchmarks 列表，"
              "也保留预设比较模型的共同基准统计。", "",
              "对 baseline 的配对差值与 step100 之间的配对差值见 summary.json。", "",
              f"固定样本总数：{total}；协议与评分器 sha256 见 summary.json。",
              "逐题结果详见 per_sample.csv；未完整覆盖的基准不产生准确率。"]
    if "general" in suites:
        lines += ["", "IFEval 评分可重复性修订：按冻结题目逐题固定 Python 随机种子，strict/loose 使用相同种子。",
                  "官方检查器对 IFEval:1122 的 # 和 IFEval:1129 的 ! 会随机替换为英文字母；本修订固定其回退，未修正这两题的标点语义。",
                  "完整 541 题分数仍包含这两题；两题最多影响 0.37 个百分点，另见 ifeval_sensitivity.json 的 539 题敏感性分析。"]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path)
    ap.add_argument("--include-general", action="store_true", help="Include OCR/IF extension; defaults to a separate results_expanded report")
    args = ap.parse_args()
    args.output_root = args.output_root or HERE / ("results_expanded" if args.include_general else "results")

    suites = {}
    selected_suites = SUITES + ([("general", HERE / "suite_general.jsonl", HERE / "results_general")] if args.include_general else [])
    for name, suite_path, root in selected_suites:
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
                "metric": metric_name(rows[0]),
            }
        for model in MODEL_NAMES:
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
    from general_metrics import aggregate_general
    report['general_details'] = aggregate_general([r for s in suites.values() for r in s['rows']], predictions)
    report['general_scorer_source_sha256'] = file_sha(HERE / "general_metrics.py")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    (args.output_root / "summary.md").write_text(
        render_markdown(suites, status, results, dimensions, bench_rows, MODEL_NAMES))

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
