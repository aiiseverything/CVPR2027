#!/usr/bin/env python3
"""Reproduce teacher overlap from immutable remote outputs and the local suite.

Run with Python containing numpy and matplotlib. No inference or network calls.
MMBench is collapsed to circular question groups before primary calculations.
Other benchmark point estimates remain row-weighted, with group bootstrap CIs.
"""
from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


P, R = "perception-step100", "reasoning-step100"
BENCHES = ["MMBench_DEV_EN_V11", "MME_perception", "MMMU_DEV_VAL", "MMStar",
           "MathVision_MINI", "MathVista_MINI", "RefCOCO_heldout"]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--local-root", type=Path, default=Path("/home/ml-user/workdir/CVPR2027"))
    ap.add_argument("--remote-root", type=Path, default=Path("/home/ml-user/workdir/CVPR2027_eval_23eed8b"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument("--bootstrap", type=int, default=20000)
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    evaluation = args.local_root / "experiments/sharedexpert_c/evaluation"
    remote = args.remote_root / "experiments/sharedexpert_c/evaluation/results"
    spec = importlib.util.spec_from_file_location("overlap_common", evaluation / "common.py")
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    rows, meta = common.load_suite(evaluation / "suite.jsonl")
    ids = {x["id"] for x in rows}
    summary = json.loads((remote / "summary.json").read_text())
    for key in ["suite_sha256", "protocol_sha256"]:
        assert summary["suite"][key] == meta[key], key
    scoring_hash = common.digest(common.SCORING_PROTOCOL)
    assert summary["scoring_protocol_sha256"] == scoring_hash
    predictions, files = {}, [evaluation / "suite.jsonl", remote / "summary.json", remote / "per_sample.csv"]
    audit = {}
    for model in [P, R, "baseline"]:
        directory = (remote if model != "baseline" else evaluation / "results") / "full" / model
        run = json.loads((directory / "run.json").read_text())
        assert not run["smoke"]
        assert run["suite_sha256"] == meta["suite_sha256"]
        assert run["protocol_sha256"] == meta["protocol_sha256"]
        assert run["scoring_protocol_sha256"] == scoring_hash
        source = directory / "predictions.scored.jsonl"
        original = common.read_jsonl(source)
        assert len(original) == len(ids) == len({x["id"] for x in original})
        original = {x["id"]: x for x in original}
        assert set(original) == ids
        rescored = {}
        for row in rows:
            record = original[row["id"]]
            assert not record.get("error"), (model, row["id"])
            assert record["row_sha256"] == common.digest(row), (model, row["id"])
            scored = common.score_prediction(row, record)
            assert scored["correct"] == record["correct"], (model, row["id"], "rescore mismatch")
            assert scored["parse_success"] == record["parse_success"]
            rescored[row["id"]] = scored
        predictions[model] = rescored
        files += [source, directory / "run.json", directory / "scoring.json"]
        audit[model] = {"rows": len(rescored), "correct": sum(x["correct"] for x in rescored.values()),
                        "length_terminated": sum(x["length_terminated"] for x in rescored.values()),
                        "mean_completion_tokens": float(np.mean([x["usage"]["completion_tokens"] for x in rescored.values()])),
                        "row_hash_and_rescoring_mismatches": 0}
    csv_rows = list(csv.DictReader((remote / "per_sample.csv").open()))
    assert len(csv_rows) == len(ids) == len({x["id"] for x in csv_rows})
    for x in csv_rows:
        for model in [P, R]:
            assert x[model] in {"True", "False"}
            assert (x[model] == "True") == predictions[model][x["id"]]["correct"]

    def failure(record):
        if record["correct"]:
            return "correct"
        if record["length_terminated"]:
            return "length"
        if not record["parse_success"]:
            return "parse"
        return "wrong_answer"

    def units(subset):
        grouped = defaultdict(list)
        for row in subset:
            key = row["group"] if row["benchmark"] == BENCHES[0] else row["id"]
            grouped[key].append(row)
        result = []
        for key, group in grouped.items():
            first = group[0]
            assert len({x["category"] for x in group}) == 1
            result.append({"id": key, "benchmark": first["benchmark"], "dimension": first["dimension"],
                           "category": first["category"], "cluster": first["group"],
                           "p": all(predictions[P][x["id"]]["correct"] for x in group),
                           "r": all(predictions[R][x["id"]]["correct"] for x in group),
                           "baseline": all(predictions["baseline"][x["id"]]["correct"] for x in group),
                           "both_finished": all(not predictions[m][x["id"]]["length_terminated"] for x in group for m in [P, R])})
        return result

    def counts(uu):
        c = Counter((u["p"], u["r"]) for u in uu)
        a, b, c_, d = c[True, True], c[True, False], c[False, True], c[False, False]
        n = len(uu)
        return {"n": n, "both_correct": a, "p_only": b, "r_only": c_, "both_wrong": d,
                "p_accuracy": (a+b)/n, "r_accuracy": (a+c_)/n, "oracle_accuracy": (a+b+c_)/n,
                "r_minus_p": (c_-b)/n, "oracle_minus_r": b/n,
                "correct_set_jaccard": a/(a+b+c_) if a+b+c_ else None,
                "correctness_agreement": (a+d)/n,
                "p_correct_also_r_correct": a/(a+b) if a+b else None,
                "r_correct_also_p_correct": a/(a+c_) if a+c_ else None,
                "r_errors_rescued_by_p": b/(b+d) if b+d else None,
                "baseline_accuracy": sum(u["baseline"] for u in uu)/n,
                "p_only_and_baseline_wrong": sum(u["p"] and not u["r"] and not u["baseline"] for u in uu),
                "r_only_and_baseline_wrong": sum(u["r"] and not u["p"] and not u["baseline"] for u in uu)}

    rng = np.random.default_rng(20261022)
    benchmark_results, bootstrap_samples, category_results = {}, {}, []
    all_units = []
    for bench in BENCHES:
        rr = [x for x in rows if x["benchmark"] == bench]
        uu = units(rr)
        all_units.extend(uu)
        stat = counts(uu)
        for model, k in [(P, "p_accuracy"), (R, "r_accuracy")]:
            assert abs(stat[k] - summary["benchmarks"][bench][model]["score"]) < 1e-12
        clusters = defaultdict(lambda: np.zeros(5, dtype=np.int64))
        for u in uu:
            clusters[u["cluster"]] += [1, u["p"], u["r"], u["p"] or u["r"], u["baseline"]]
        arr = np.array(list(clusters.values()))
        samples = []
        for start in range(0, args.bootstrap, 200):
            ix = rng.integers(0, len(arr), size=(min(200, args.bootstrap-start), len(arr)))
            sums = arr[ix].sum(axis=1)
            samples.append(sums[:, 1:] / sums[:, :1])
        boot = np.concatenate(samples)
        bootstrap_samples[bench] = boot
        stat.update(benchmark=bench, dimension=rr[0]["dimension"], raw_rows=len(rr), clusters=len(arr),
                    metric="circular_group_all_correct" if bench == BENCHES[0] else "row_accuracy",
                    r_minus_p_ci95=np.quantile(boot[:, 1]-boot[:, 0], [.025,.975]).tolist(),
                    oracle_minus_r_ci95=np.quantile(boot[:, 2]-boot[:, 1], [.025,.975]).tolist(),
                    both_finished_only=counts([u for u in uu if u["both_finished"]]))
        benchmark_results[bench] = stat
        for category in sorted({u["category"] for u in uu}):
            category_results.append({"benchmark": bench, "category": category,
                                     **counts([u for u in uu if u["category"] == category])})
    dimension_results = {}
    for dimension in ["perception", "reasoning", "general"]:
        benches = [b for b in BENCHES if benchmark_results[b]["dimension"] == dimension]
        boot = np.mean([bootstrap_samples[b] for b in benches], axis=0)
        stat = {k: float(np.mean([benchmark_results[b][k] for b in benches]))
                for k in ["p_accuracy", "r_accuracy", "oracle_accuracy", "r_minus_p", "oracle_minus_r", "baseline_accuracy"]}
        stat.update(benchmarks=benches,
                    r_minus_p_ci95=np.quantile(boot[:, 1]-boot[:, 0], [.025,.975]).tolist(),
                    oracle_minus_r_ci95=np.quantile(boot[:, 2]-boot[:, 1], [.025,.975]).tolist())
        dimension_results[dimension] = stat

    row_records, failure_stats, raw_units = [], {}, []
    for row in rows:
        p, r, b = (predictions[m][row["id"]] for m in [P, R, "baseline"])
        raw_units.append({"p": p["correct"], "r": r["correct"], "baseline": b["correct"]})
        if p["correct"] == r["correct"]:
            continue
        row_records.append({"id": row["id"], "benchmark": row["benchmark"], "dimension": row["dimension"],
                            "category": row["category"], "group": row["group"],
                            "winner": "P" if p["correct"] else "R", "baseline_correct": b["correct"],
                            "p_failure": failure(p), "r_failure": failure(r),
                            "question": row["question"], "gold": str(row["answer"]),
                            "choices": json.dumps(row.get("choices", {}), ensure_ascii=False),
                            "p_extracted": p.get("extracted", ""), "r_extracted": r.get("extracted", ""),
                            "p_tokens": p["usage"]["completion_tokens"], "r_tokens": r["usage"]["completion_tokens"]})
    for scope in ["all", "perception", "reasoning", "general"]:
        rr = [x for x in row_records if scope == "all" or x["dimension"] == scope]
        failure_stats[scope] = {winner: dict(Counter(x["r_failure" if winner == "P" else "p_failure"]
                                                   for x in rr if x["winner"] == winner)) for winner in ["P", "R"]}
    baseline_transitions = []
    for dimension in ["perception", "reasoning", "general"]:
        uu = [u for u in all_units if u["dimension"] == dimension]
        for teacher in ["p", "r"]:
            recovered = sum(u[teacher] and not u["baseline"] for u in uu)
            regressed = sum(not u[teacher] and u["baseline"] for u in uu)
            baseline_transitions.append({"dimension": dimension, "teacher": teacher.upper(),
                                         "evaluation_units": len(uu), "baseline_wrong_now_correct": recovered,
                                         "baseline_correct_now_wrong": regressed, "net_correct_gain": recovered-regressed})
    notes = [
        "P=perception-step100; R=reasoning-step100. All predictions rescored locally; IDs, row hashes, suite and protocols match.",
        "Primary MMBench units are 250 circular question groups (all variants correct), not its 930 variant rows.",
        "Oracle selects one teacher per evaluation unit using ground truth. For MMBench the same teacher must pass the entire circular group.",
        "Raw-row totals include repeated MMBench variants and are descriptive only, not the reported dimension macro metric.",
        f"{args.bootstrap:,} paired cluster bootstrap draws within each benchmark; dimension scores equally weight benchmarks. CIs are pointwise, not multiplicity adjusted.",
        "Both-finished subset conditions on model behavior and is a diagnostic only; it does not estimate a causal truncation-free ability.",
        "Final-answer overlap is not prefix continuation advantage, actual router performance, or a distillation upper bound.",
        "Category statistics are exploratory and often small; test labels must not train a router.",
    ]
    commit = subprocess.check_output(["git", "-C", str(args.remote_root), "rev-parse", "HEAD"], text=True).strip()
    result = {"source_commit": commit, "bootstrap_draws": args.bootstrap, "seed": 20261022,
              "suite_sha256": meta["suite_sha256"], "protocol_sha256": meta["protocol_sha256"],
              "scoring_protocol_sha256": scoring_hash, "audit": audit,
              "input_sha256": {str(p): common.file_sha(p) for p in files},
              "benchmarks": benchmark_results, "dimension_macro": dimension_results,
              "raw_row_overlap": counts(raw_units), "primary_unit_overlap_descriptive": counts(all_units),
              "baseline_transitions_descriptive_counts": baseline_transitions,
              "discordant_row_loser_failure": failure_stats, "notes": notes}
    (out / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2)+"\n")
    flat = [{k: v for k, v in x.items() if k != "both_finished_only"} for x in benchmark_results.values()]
    write_csv(out / "benchmark_overlap.csv", flat)
    write_csv(out / "category_overlap.csv", category_results)
    write_csv(out / "discordant_cases.csv", row_records)
    write_csv(out / "evaluation_units.csv", all_units)
    write_csv(out / "baseline_transitions.csv", baseline_transitions)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 5.2), layout="constrained")
    labels = ["MMBench (250 groups)", "MME perception", "MMMU", "MMStar", "MathVision", "MathVista", "RefCOCO"]
    left = np.zeros(len(BENCHES))
    for key, label, color in [("both_correct", "Both correct", "#557f9a"), ("p_only", "P only", "#efab48"),
                              ("r_only", "R only", "#569c80"), ("both_wrong", "Both wrong", "#d8dce1")]:
        values = np.array([benchmark_results[b][key]/benchmark_results[b]["n"]*100 for b in BENCHES])
        ax.barh(labels, values, left=left, color=color, label=label, height=.66)
        for i, (v, x) in enumerate(zip(values, left)):
            if v >= 3.5:
                ax.text(x+v/2, i, f"{v:.1f}", va="center", ha="center", fontsize=9,
                        color="white" if key == "both_correct" else "#142c3a")
        left += values
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of evaluation units (%)")
    ax.set_title("Teacher correct-answer overlap | frozen step-100 evaluation", loc="left", pad=16)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(.5, -.12), frameon=False)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="x", alpha=.15)
    fig.savefig(out / "overlap.png", dpi=180)
    plt.close(fig)

    headers = ["评测集", "计分单位", "共同答对", "仅 P 对", "仅 R 对", "共同答错", "P / R / 理想选择 (%)", "R−P，95% CI (pp)"]
    table_rows = []
    for b, x in benchmark_results.items():
        ci = x["r_minus_p_ci95"]
        table_rows.append([b, x["n"], x["both_correct"], x["p_only"], x["r_only"], x["both_wrong"],
                           " / ".join(f"{100*x[k]:.2f}" for k in ["p_accuracy", "r_accuracy", "oracle_accuracy"]),
                           f"{100*x['r_minus_p']:.2f} [{100*ci[0]:.2f}, {100*ci[1]:.2f}]"])
    def table(head, data):
        return "<table><thead><tr>"+"".join(f"<th>{html.escape(str(v))}</th>" for v in head)+"</tr></thead><tbody>"+"".join("<tr>"+"".join(f"<td>{html.escape(str(v))}</td>" for v in row)+"</tr>" for row in data)+"</tbody></table>"
    dim_rows = [[d]+[f"{100*x[k]:.2f}" for k in ["p_accuracy", "r_accuracy", "oracle_accuracy", "oracle_minus_r"]]
                for d,x in dimension_results.items()]
    report = """<!doctype html><html lang="zh"><meta charset="utf-8"><title>两教师答案互补分析</title>
    <style>body{max-width:1250px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#243442;background:#fafbfd}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{padding:9px;border-bottom:1px solid #dce2e6;text-align:left}th{background:#eaf0f4}img{width:100%}code{background:#eaf0f4;padding:3px}a{color:#176688}li{margin:6px 0}</style>
    <h1>两教师答案正确性重合与互补分析</h1>"""
    report += f"<p>远端提交 <code>{commit}</code>。P 为感知教师，R 为推理教师，均为 step 100。全部 3423 行 ID、逐行 hash、协议已核对，并从原始回答重新评分，与保存结果完全一致。</p>"
    report += "<img src='overlap.png' alt='各基准的两教师正确性重合比例'>" + table(headers, table_rows)
    report += "<p>MMBench 的 930 个选项版本先合并为 250 个问题组，要求同一教师整组全对；其余基准保留原计分口径。理想选择使用真实标签，仅表示当前教师最终答案的潜在覆盖。</p>"
    report += table(["维度（基准等权）", "P (%)", "R (%)", "理想选择 (%)", "相对 R 增量 (pp)"], dim_rows)
    report += "<h2>相对 baseline 的逐题变化</h2><p>以下为计数，general 中 MMBench 按循环组计分；不是各基准等权均分。</p>"
    report += table(["维度", "教师", "计分单位", "原本错、现在对", "原本对、现在错", "净增正确"],
                    [[x[k] for k in ["dimension", "teacher", "evaluation_units", "baseline_wrong_now_correct", "baseline_correct_now_wrong", "net_correct_gain"]] for x in baseline_transitions])
    fs = failure_stats["reasoning"]
    reasoning_n = sum(x["dimension"] == "reasoning" for x in rows)
    report += (f"<h2>推理题的截断相关分歧</h2><p>{reasoning_n} 道推理题中，"
               f"P 独有答对 {sum(fs['P'].values())} 道（对方失败类型：{html.escape(str(fs['P']))}）；"
               f"R 独有答对 {sum(fs['R'].values())} 道（对方失败类型：{html.escape(str(fs['R']))}）。"
               f"R 净多答对 {sum(fs['R'].values())-sum(fs['P'].values())} 道，截断相关胜负的净差为 "
               f"{fs['R'].get('length',0)-fs['P'].get('length',0)} 道。"
               "这是当前预算下的错误类型统计，不是放宽预算后性能的因果预测。</p>")
    report += f"<h2>判读边界</h2><ul><li>重合指正确题集合重合，不代表生成文本或推理过程相同。</li><li>正确性互补不是已验证的路由收益，也不能保证被学生蒸馏吸收。</li><li>截断仍按原协议判零；双方均完成的子集仅用于诊断，存在选择偏差。</li><li>95% 区间来自基准内配对分组 bootstrap，{args.bootstrap} 次；子类统计是探索性结果。</li></ul>"
    report += "<p>完整文件：<a href='analysis.json'>结构化结果及输入哈希</a> · <a href='benchmark_overlap.csv'>基准汇总</a> · <a href='category_overlap.csv'>子类统计</a> · <a href='discordant_cases.csv'>全部分歧样本</a> · <a href='baseline_transitions.csv'>相对 baseline 变化</a> · <a href='analyze.py'>复现脚本</a></p></html>"
    (out / "report.html").write_text(report, encoding="utf-8")
    print(json.dumps({"source_commit": commit, "audit": audit, "benchmarks": flat,
                      "dimension_macro": dimension_results, "raw_overlap": result["raw_row_overlap"],
                      "failure_stats": failure_stats, "report": str(out/"report.html")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
