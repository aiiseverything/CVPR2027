#!/usr/bin/env python3
"""Audit completeness and compare four models on identical frozen evaluation rows."""
from __future__ import annotations

import argparse
import csv
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from common import HERE, SCORING_PROTOCOL, digest, file_sha, load_suite, read_jsonl, score_prediction

MODELS = ["baseline", "generalist", "perception", "reasoning"]


def units(rows, predictions):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(float(predictions.get(row["id"], {}).get("correct", False)))
    if rows[0]["benchmark"] == "MMBench_DEV_EN_V11":
        return np.array([[float(all(values)), 1] for values in groups.values()])
    return np.array([[sum(values), len(values)] for values in groups.values()])


def estimate(a, b=None):
    """Cluster bootstrap; shared draws give a paired CI when b is provided."""
    rng = np.random.default_rng(20261019)
    values = []
    for _ in range(20):
        indices = rng.integers(0, len(a), size=(100, len(a)))
        sums = a[indices].sum(axis=1)
        value = sums[:, 0] / sums[:, 1]
        if b is not None:
            other = b[indices].sum(axis=1)
            value -= other[:, 0] / other[:, 1]
        values.extend(value)
    point = a[:, 0].sum() / a[:, 1].sum()
    if b is not None:
        point -= b[:, 0].sum() / b[:, 1].sum()
    return float(point), [float(v) for v in np.quantile(values, [0.025, 0.975])]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", type=Path, default=HERE / "suite.jsonl")
    ap.add_argument("--output-root", type=Path, default=HERE / "results")
    args = ap.parse_args()
    rows, meta = load_suite(args.suite)
    by_bench = defaultdict(list)
    for row in rows:
        by_bench[row["benchmark"]].append(row)
    predictions, status = {}, {}
    for model in MODELS:
        directory = args.output_root / "full" / model
        core_progress = False
        if model == "baseline" and not (directory / "run.json").exists() and args.output_root.resolve() == (HERE / "results").resolve():
            core = HERE / "core_results/full/baseline"
            if (core / "run.json").exists():
                directory = core
                core_progress = True
        if not (directory / "run.json").exists():
            status[model] = {"status": "not_started", "n_expected": len(rows), "missing": len(rows)}
            continue
        run = json.loads((directory / "run.json").read_text())
        if (not core_progress and run["suite_sha256"] != meta["suite_sha256"]) or run["protocol_sha256"] != meta["protocol_sha256"] or run["smoke"]:
            raise SystemExit(f"Incompatible evaluation run for {model}")
        pred = {r["id"]: r for r in read_jsonl(directory / "predictions.jsonl")}
        for row in rows:
            if row["id"] in pred and pred[row["id"]]["row_sha256"] != digest(row):
                raise SystemExit(f"Prediction row hash mismatch: {model}, {row['id']}")
            if row["id"] in pred:
                pred[row["id"]] = score_prediction(row, pred[row["id"]])
        # A separate scored snapshot preserves append-only raw generation caches,
        # including caches still being written by a previously started client.
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
        n_missing = sum(r["id"] not in pred for r in rows)
        n_errors = sum(bool(pred.get(r["id"], {}).get("error")) for r in rows)
        predictions[model] = pred
        status[model] = {
            "status": "core_in_progress" if core_progress else "complete" if not n_missing and not n_errors else "incomplete",
            "n_expected": len(rows), "missing": n_missing, "api_errors": n_errors,
            "extraction_failures": sum(not p.get("parse_success", False) for p in pred.values() if not p.get("error")),
            "length_terminated": sum(p.get("finish_reason") == "length" for p in pred.values()),
            "mean_completion_tokens": float(np.mean([p.get("usage", {}).get("completion_tokens", 0) for p in pred.values()])) if pred else None,
        }
    results, deltas = {}, {}
    for bench, samples in sorted(by_bench.items()):
        results[bench], deltas[bench] = {}, {}
        for model, pred in predictions.items():
            if status[model]["status"] != "complete":
                continue
            array = units(samples, pred)
            accuracy, ci = estimate(array)
            results[bench][model] = {"score": accuracy, "ci95": ci, "n_rows": len(samples), "n_clusters": len(array),
                                     "metric": "circular_group_accuracy" if bench == "MMBench_DEV_EN_V11" else "accuracy_at_iou_0.5" if bench == "RefCOCO_heldout" else "strict_accuracy"}
            if model != "baseline" and status["baseline"]["status"] == "complete":
                delta, delta_ci = estimate(array, units(samples, predictions["baseline"]))
                deltas[bench][model] = {"delta": delta, "ci95": delta_ci}
    dimensions = {}
    for dimension in ["reasoning", "perception", "general"]:
        benches = sorted({r["benchmark"] for r in rows if r["dimension"] == dimension})
        dimensions[dimension] = {"benchmarks": benches, "models": {}}
        for model in MODELS:
            if all(model in results[b] for b in benches):
                dimensions[dimension]["models"][model] = float(np.mean([results[b][model]["score"] for b in benches]))
    complementarity = {}
    if all(status[m]["status"] == "complete" for m in ["generalist", "perception", "reasoning"]):
        for dimension in ["reasoning", "perception", "general"]:
            subset = [r for r in rows if r["dimension"] == dimension]
            for specialist in ["perception", "reasoning"]:
                pairs = [(bool(predictions["generalist"][r["id"]]["correct"]), bool(predictions[specialist][r["id"]]["correct"])) for r in subset]
                complementarity[f"{dimension}/{specialist}"] = {"n": len(pairs), "specialist_only_correct": sum(s and not g for g, s in pairs), "generalist_only_correct": sum(g and not s for g, s in pairs), "both_wrong": sum(not g and not s for g, s in pairs)}
        complementarity["warning"] = "Descriptive final-answer oracle only; not an estimate of C-router prefix continuation gains. Evaluation answers must not train the router."
    report = {"suite": meta, "scoring_protocol": SCORING_PROTOCOL, "scoring_protocol_sha256": digest(SCORING_PROTOCOL),
              "status": status, "benchmarks": results, "paired_delta_vs_baseline": deltas, "dimension_macro": dimensions, "complementarity": complementarity}
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# SharedExpert 三教师统一评测", "", "固定协议：thinking=True，temperature=0，response≤8192；实际多模态 prompt≤2048。超过 prompt 预算的样本对四模型统一排除。finish_reason=length 一律记零，即使截断前已有完整正确 final；四模型统一。", "", "本表为本地严格抽取及匹配协议，与既有 32768-token / DeepSeek judge 的历史成绩不能直接比较。", "", "| 模型 | 状态 | 缺失 | API 错误 | 抽取失败 | 长度截断 |", "|---|---|---:|---:|---:|---:|"]
    for model in MODELS:
        s = status[model]
        lines.append(f"| {model} | {s['status']} | {s.get('missing', 0)} | {s.get('api_errors', '—')} | {s.get('extraction_failures', '—')} | {s.get('length_terminated', '—')} |")
    lines += ["", "只报告完整模型的成绩；准确率单位为百分比，括号为按图像／循环选项组 bootstrap 的 95% 区间。", "", "| 评测集 | 实际行数 | baseline | generalist | perception | reasoning |", "|---|---:|---:|---:|---:|---:|"]
    for bench, samples in sorted(by_bench.items()):
        cells = []
        for model in MODELS:
            r = results[bench].get(model)
            cells.append(f"{100*r['score']:.2f} ({100*r['ci95'][0]:.2f}–{100*r['ci95'][1]:.2f})" if r else "—")
        lines.append(f"| {bench} | {len(samples)} | " + " | ".join(cells) + " |")
    lines += ["", "MMBench 使用所选 base question 的全部循环选项版本，主指标为一组全对；其余是严格答案准确率，RefCOCO 为 IoU≥0.5。MME 仅 count/position/color/existence 子集，非完整 MME 总分。", "", "| 维度（各评测集等权） | baseline | generalist | perception | reasoning |", "|---|---:|---:|---:|---:|"]
    for dim, values in dimensions.items():
        lines.append(f"| {dim} | " + " | ".join(f"{100*values['models'][m]:.2f}" if m in values["models"] else "—" for m in MODELS) + " |")
    lines += ["", f"固定样本总数：{len(rows)}；预先排除条目：{len(meta['excluded'])}；grounding 包含状态：{meta['grounding_included']}。", "", "逐题结果与配对差值详见 summary.json / per_sample.csv；未完成的任务不产生最终准确率。"]
    (args.output_root / "summary.md").write_text("\n".join(lines) + "\n")
    with (args.output_root / "per_sample.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "benchmark", "dimension", "category", "group"] + MODELS)
        writer.writeheader()
        for row in rows:
            writer.writerow({**{k: row[k] for k in ["id", "benchmark", "dimension", "category", "group"]}, **{m: predictions.get(m, {}).get(row["id"], {}).get("correct", "") for m in MODELS}})
    print(json.dumps({"status": status, "report": str(args.output_root / "summary.md")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
