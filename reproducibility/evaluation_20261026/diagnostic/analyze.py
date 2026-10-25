#!/usr/bin/env python3
"""Read frozen teacher outputs and diagnose perception scores; never rescore them.

Writes only analysis.json and comparison.csv beside this script. Bootstrap uses
the existing summarize.py convention: paired cluster draws, 2,000 replicates,
NumPy default_rng(20261019). Conditional subsets are descriptive, not causal.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import statistics

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2] / "experiments" / "sharedexpert_c"
EVAL = PROJECT / "evaluation"
MODELS = ["baseline", "generalist", "perception-step100", "reasoning-step100", "perception-v2-step100"]
REASON = "reasoning-step100"
SPECS = [("suite.jsonl", "results", "summary_main.json"),
         ("suite_blink_mmvp.jsonl", "results_blink_mmvp", "summary.json"),
         ("suite_general.jsonl", "results_general", "summary.json")]
SEED = 20261019


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def units(rows, pred):
    groups = defaultdict(list)
    for row in rows:
        groups[row["group"]].append(float(pred[row["id"]]["correct"]))
    if rows[0]["benchmark"] == "MMBench_DEV_EN_V11":
        return np.array([[float(all(v)), 1] for v in groups.values()])
    return np.array([[sum(v), len(v)] for v in groups.values()])


def estimate(a, b):
    rng = np.random.default_rng(SEED)
    values = []
    for _ in range(20):
        ix = rng.integers(0, len(a), size=(100, len(a)))
        x, y = a[ix].sum(axis=1), b[ix].sum(axis=1)
        values.extend(x[:, 0] / x[:, 1] - y[:, 0] / y[:, 1])
    point = a[:, 0].sum() / a[:, 1].sum() - b[:, 0].sum() / b[:, 1].sum()
    return {"delta_pp": float(point * 100),
            "ci95_pp": [float(v * 100) for v in np.quantile(values, [.025, .975])]}


def rowstats(rows, pred):
    ps = [pred[r["id"]] for r in rows]
    n = len(ps)
    if not n:
        return {"n": 0, "correct": 0, "accuracy_pct": None}
    completion = [p["usage"]["completion_tokens"] for p in ps]
    return {"n": n, "correct": sum(bool(p["correct"]) for p in ps),
            "accuracy_pct": 100 * sum(bool(p["correct"]) for p in ps) / n,
            "finish_reason": dict(Counter(p["finish_reason"] for p in ps)),
            "completion_tokens_mean": statistics.mean(completion),
            "completion_tokens_median": statistics.median(completion),
            "parse_failures": sum(not p.get("parse_success", False) for p in ps),
            "parse_failures_nonlength": sum(not p.get("parse_success", False) and p["finish_reason"] != "length" for p in ps),
            "length_with_parsed_answer": sum(p.get("parse_success", False) and p["finish_reason"] == "length" for p in ps),
            "length_with_correct_before_penalty": sum(p.get("correct_before_length_penalty", False) and p["finish_reason"] == "length" for p in ps),
            "api_errors": sum(bool(p.get("error")) for p in ps)}


def conditional(rows, rp, op):
    n = len(rows)
    rc = sum(bool(rp[r["id"]]["correct"]) for r in rows)
    oc = sum(bool(op[r["id"]]["correct"]) for r in rows)
    return {"n": n, "reason_correct": rc, "other_correct": oc,
            "reason_accuracy_pct": 100 * rc / n if n else None,
            "other_accuracy_pct": 100 * oc / n if n else None,
            "delta_pp": 100 * (rc - oc) / n if n else None}


def compare(rows, rp, op, other, bootstrap=True):
    n = len(rows)
    win = [r for r in rows if rp[r["id"]]["correct"] and not op[r["id"]]["correct"]]
    loss = [r for r in rows if not rp[r["id"]]["correct"] and op[r["id"]]["correct"]]
    strata = defaultdict(list)
    for r in rows:
        rl, ol = rp[r["id"]]["finish_reason"] == "length", op[r["id"]]["finish_reason"] == "length"
        key = "both_length" if rl and ol else "reason_only_length" if rl else "other_only_length" if ol else "neither_length"
        strata[key].append(r)
    decomposition = {}
    for key in ["neither_length", "other_only_length", "reason_only_length", "both_length"]:
        s = conditional(strata[key], rp, op)
        s["net_correct"] = s["reason_correct"] - s["other_correct"]
        s["contribution_to_full_row_delta_pp"] = 100 * s["net_correct"] / n
        decomposition[key] = s
    assert sum(v["net_correct"] for v in decomposition.values()) == len(win) - len(loss)
    bothstop = [r for r in rows if rp[r["id"]]["finish_reason"] == op[r["id"]]["finish_reason"] == "stop"]
    bothparse = [r for r in bothstop if rp[r["id"]].get("parse_success") and op[r["id"]].get("parse_success")]
    answer = {"reason_model": REASON, "other_model": other, "n": n,
              "win": len(win), "loss": len(loss), "ties": n - len(win) - len(loss),
              "row_delta_pp": 100 * (len(win) - len(loss)) / n,
              "length_decomposition": decomposition,
              "both_nonlength": conditional(strata["neither_length"], rp, op),
              "both_finish_stop": conditional(bothstop, rp, op),
              "both_finish_stop_and_parse": conditional(bothparse, rp, op)}
    if bootstrap:
        answer["benchmark_metric_paired_bootstrap"] = estimate(units(rows, rp), units(rows, op))
    return answer


def main():
    inputs, rows, predictions, summaries = {}, [], {m: {} for m in MODELS}, {}
    completeness = {}
    def record(path):
        inputs[str(path)] = {"sha256": sha(path), "size_bytes": path.stat().st_size}
    for suite, result, summaryname in SPECS:
        path = EVAL / suite
        record(path)
        record(path.with_suffix(".meta.json"))
        sr = [json.loads(l) for l in path.open()]
        rows.extend(sr)
        expected = {r["id"] for r in sr}
        summarypath = EVAL / result / summaryname
        record(summarypath)
        summaries[result] = json.load(summarypath.open())
        completeness[result] = {}
        for model in MODELS:
            pp = EVAL / result / "full" / model / "predictions.scored.jsonl"
            record(pp)
            record(pp.with_name("scoring.json"))
            record(pp.with_name("run.json"))
            loaded = {}
            for lineno, line in enumerate(pp.open(), 1):
                p = json.loads(line)
                assert p["id"] not in loaded, (pp, "duplicate", p["id"])
                p["source_file"], p["source_line"] = str(pp), lineno
                p.pop("raw_response", None)
                p.pop("reasoning", None)
                loaded[p["id"]] = p
            assert set(loaded) == expected, (pp, "incomplete or extra IDs")
            assert not set(loaded) & set(predictions[model]), (pp, "cross-suite ID collision")
            assert not any(p.get("error") for p in loaded.values()), (pp, "API errors")
            predictions[model].update(loaded)
            completeness[result][model] = {"n": len(loaded), "missing": 0, "api_errors": 0,
                "scoring_protocol_sha256": sorted({p["scoring_protocol_sha256"] for p in loaded.values()})}
    # Existing audits are evidence of exact-hash exclusions, not near-duplicate proof.
    auditpaths = [PROJECT / "data_pipeline/prepared/validation.json",
                  PROJECT / "data_pipeline/prepared/audit.json",
                  PROJECT / "data_pipeline/prepared/perception_v2.audit.json",
                  EVAL / "prepare_general.py", EVAL / "summarize.py", EVAL / "common.py"]
    for p in auditpaths:
        record(p)
    report = {
        "purpose": "Descriptive reasoning-teacher perception diagnosis from existing frozen scores",
        "models": MODELS, "input_sha256_and_sizes": inputs, "completeness": completeness,
        "bootstrap": {"seed": SEED, "replicates": 2000, "unit": "suite group; paired shared draws", "multiple_comparison_correction": False},
        "limitations": [
            "No rescoring, GPU run, generation, or input mutation; existing correct fields are used.",
            "Both-stop/nonlength/parse subsets are post-treatment selections, not causal ablations; they do not estimate equal-budget uncensored potential accuracy.",
            "Length-stratum contributions are an exact arithmetic decomposition of observed row-score differences, not a causal attribution.",
            "All reported CIs are exploratory and unadjusted for multiple comparisons.",
            "MMBench benchmark accuracy is all-correct circular groups; win/loss and conditional analyses remain explicitly row-level.",
            "MMVP existing score is row accuracy, not the official paired-question metric.",
            "MME is only 240 count/color/existence/position rows, not full official MME.",
            "OCRBench is the filtered 997-row suite scored by normalized substring match, not official 1000-point total.",
            "Exact RGB/question-hash exclusions do not exclude all crops/re-encodings or base-model pretraining exposure.",
            "High-resolution benchmarks not yet evaluated are excluded; no mechanism is identified by this observational comparison."
        ], "benchmarks": {}, "categories": {},
        "overlap_evidence": {
            "prepared_validation": json.load((PROJECT / "data_pipeline/prepared/validation.json").open()),
            "ocr_suite_exclusions": summaries["results_general"]["suite"]["excluded"],
            "ocr_suite_training_sha256": summaries["results_general"]["suite"]["training_sha256"],
            "perception_v2_leakage_policy": json.load((PROJECT / "data_pipeline/prepared/perception_v2.audit.json").open())["leakage"],
        }
    }
    grouped = defaultdict(list)
    for r in rows:
        if r["benchmark"] != "IFEval_strict":
            grouped[r["benchmark"]].append(r)
    csvrows = []
    def analyze_group(rs, bench, category=None):
        stats = {m: rowstats(rs, predictions[m]) for m in MODELS}
        for m in MODELS:
            a = units(rs, predictions[m])
            stats[m]["benchmark_accuracy_pct"] = float(100 * a[:, 0].sum() / a[:, 1].sum())
            stats[m]["n_clusters"] = len(a)
        pairs = {}
        for m in MODELS:
            if m == REASON:
                continue
            p = compare(rs, predictions[REASON], predictions[m], m, bootstrap=category is None)
            pairs[m] = p
            ci = p.get("benchmark_metric_paired_bootstrap", {})
            csvrows.append({"benchmark": bench, "category": category or "ALL", "other_model": m,
                "n": len(rs), "reason_accuracy_pct": stats[REASON]["benchmark_accuracy_pct"],
                "other_accuracy_pct": stats[m]["benchmark_accuracy_pct"],
                "win_rows": p["win"], "loss_rows": p["loss"], "row_delta_pp": p["row_delta_pp"],
                "metric_delta_pp": ci.get("delta_pp", ""), "metric_ci95_low_pp": ci.get("ci95_pp", ["", ""])[0],
                "metric_ci95_high_pp": ci.get("ci95_pp", ["", ""])[1],
                "reason_length_n": stats[REASON]["finish_reason"].get("length", 0),
                "other_length_n": stats[m]["finish_reason"].get("length", 0),
                "neither_length_net_correct": p["length_decomposition"]["neither_length"]["net_correct"],
                "other_only_length_net_correct": p["length_decomposition"]["other_only_length"]["net_correct"],
                "reason_only_length_net_correct": p["length_decomposition"]["reason_only_length"]["net_correct"],
                "both_length_net_correct": p["length_decomposition"]["both_length"]["net_correct"],
                "both_stop_n": p["both_finish_stop"]["n"], "both_stop_delta_pp": p["both_finish_stop"]["delta_pp"]})
        return {"n": len(rs), "models": stats, "reason_vs_others": pairs}
    for bench, rs in sorted(grouped.items()):
        report["benchmarks"][bench] = analyze_group(rs, bench)
        if bench in ["BLINK_val", "OCRBench_heldout", "MME_perception", "MMStar"]:
            cats = defaultdict(list)
            for r in rs:
                cats[r["category"]].append(r)
            report["categories"][bench] = {cat: analyze_group(crs, bench, cat) for cat, crs in sorted(cats.items())}
    report["script_sha256"] = sha(Path(__file__))
    (HERE / "analysis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    with (HERE / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csvrows[0]))
        w.writeheader()
        w.writerows(csvrows)
    print(json.dumps({"output": str(HERE), "benchmarks": len(report["benchmarks"]),
                      "comparison_rows": len(csvrows), "input_files": len(inputs)}))


if __name__ == "__main__":
    main()
