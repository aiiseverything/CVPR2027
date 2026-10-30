#!/usr/bin/env python3
"""汇总 6 组推理结果，生成逐样本表与准确率汇总。

产出:
  probe/results/per_sample.csv   每行一个样本，6 组的预测与对错
  probe/results/summary.csv      各 model × condition 准确率
"""

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    os.makedirs(common.RESULT_DIR, exist_ok=True)
    samples = common.load_samples()
    by_idx = {s["idx"]: s for s in samples}

    # 载入 6 组结果
    runs = {}
    for mk in common.MODELS:
        for cond in common.CONDITIONS:
            recs = common.read_jsonl(common.out_path(mk, cond))
            runs[(mk, cond)] = {r["idx"]: r for r in recs}
            print(f"{mk:>3} / {cond:<9} {len(recs):>5} 条")

    # 完整性检查：6 组的 idx 集合必须完全一致
    idx_sets = {k: set(v) for k, v in runs.items() if v}
    if idx_sets:
        ref_key = next(iter(idx_sets))
        ref = idx_sets[ref_key]
        for k, s in idx_sets.items():
            if s != ref:
                msg = (f"✗ {k} 的样本集与 {ref_key} 不一致: "
                       f"缺 {sorted(ref - s)[:5]} 多 {sorted(s - ref)[:5]}")
                print(msg)
                sys.exit(1)
        print(f"✓ 6 组样本集一致 ({len(ref)} 条)")
    n_common = len(next(iter(idx_sets.values()))) if idx_sets else 0

    # 逐样本表
    per_path = os.path.join(common.RESULT_DIR, "per_sample.csv")
    header = ["idx", "gold_answer", "bbox_area"]
    for mk in common.MODELS:
        for cond in common.CONDITIONS:
            header += [f"{mk}_{cond}_pred", f"{mk}_{cond}_correct",
                       f"{mk}_{cond}_trunc", f"{mk}_{cond}_method"]

    rows = []
    for idx in sorted(idx_sets[ref_key]):
        s = by_idx.get(idx, {})
        row = {"idx": idx, "gold_answer": s.get("gold_answer"), "bbox_area": s.get("bbox_area")}
        for mk in common.MODELS:
            for cond in common.CONDITIONS:
                r = runs[(mk, cond)].get(idx)
                pred = r.get("pred_answer") if r else None
                trunc = 1 if (r and r.get("finish_reason") == "length") else 0
                row[f"{mk}_{cond}_pred"] = pred or ""
                row[f"{mk}_{cond}_trunc"] = trunc if r else ""
                row[f"{mk}_{cond}_method"] = (r.get("extract_method") or "") if r else ""
                # 口径：截断不额外惩罚 —— 在 4096 预算内给出的答案算数。
                # 否则比的就成了「谁的思维链更短」。抽取可信度另用 _method 列区分。
                if not r:
                    row[f"{mk}_{cond}_correct"] = ""
                else:
                    row[f"{mk}_{cond}_correct"] = int(pred == row["gold_answer"]) if pred else 0
        rows.append(row)

    with open(per_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"✓ {per_path}")

    # 汇总。
    #   主指标 accuracy      = 预测 == 金标，**不因截断额外惩罚**（4096 预算内给出就算数）
    #   acc_answered         = 只看自然收尾的样本（模型自己决定停了）
    #   trunc_correct        = 撞顶但答对的条数；其中可靠抽取(tag/explicit)的有多少
    #     —— 可靠抽取的才是「模型真给出了答案后继续啰嗦被截断」，
    #        tail/fallback 是「从半截推理里捡字母」，准确率约等于 1/4 随机
    RELIABLE = {"tag", "explicit"}
    sum_path = os.path.join(common.RESULT_DIR, "summary.csv")
    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "n", "n_correct", "accuracy",
                    "n_trunc", "trunc_rate", "n_answered", "acc_answered",
                    "trunc_correct", "trunc_correct_reliable", "n_extract_fail"])
        for mk in common.MODELS:
            for cond in common.CONDITIONS:
                key = f"{mk}_{cond}_correct"
                vals = [r[key] for r in rows if r[key] != ""]
                n = len(vals)
                nc = sum(vals)
                recs = runs[(mk, cond)]
                ids = [i for i in idx_sets[ref_key] if i in recs]
                trunc = [i for i in ids if recs[i].get("finish_reason") == "length"]
                ans = [i for i in ids if recs[i].get("finish_reason") != "length"]
                nc_ans = sum(1 for i in ans
                             if recs[i].get("pred_answer") == by_idx[i]["gold_answer"])
                tc = [i for i in trunc
                      if recs[i].get("pred_answer") == by_idx[i]["gold_answer"]]
                tc_rel = [i for i in tc if recs[i].get("extract_method") in RELIABLE]
                fails = sum(1 for r in rows if r[f"{mk}_{cond}_pred"] == "")
                w.writerow([mk, cond, n, nc, round(nc / n, 4) if n else "",
                            len(trunc), round(len(trunc) / n, 4) if n else "",
                            len(ans), round(nc_ans / len(ans), 4) if ans else "",
                            len(tc), len(tc_rel), fails])
    print(f"✓ {sum_path}")

    def row_of(mk, cond):
        vals = [x[f"{mk}_{cond}_correct"] for x in rows if x[f"{mk}_{cond}_correct"] != ""]
        recs = runs[(mk, cond)]
        ids = [i for i in idx_sets[ref_key] if i in recs]
        trunc = [i for i in ids if recs[i].get("finish_reason") == "length"]
        tc = [i for i in trunc if recs[i].get("pred_answer") == by_idx[i]["gold_answer"]]
        tc_rel = [i for i in tc if recs[i].get("extract_method") in RELIABLE]
        return sum(vals) / len(vals) if vals else 0, len(trunc), len(ids), len(tc), len(tc_rel)

    print("\n=== 准确率（撞顶但答对算对） ===")
    print(f"{'model':>5} | " + " | ".join(f"{c:>10}" for c in common.CONDITIONS))
    for mk in common.MODELS:
        cells = [f"{row_of(mk,c)[0]*100:9.1f}%" for c in common.CONDITIONS]
        print(f"{mk:>5} | " + " | ".join(cells))

    print("\n=== 撞顶率 / 其中答对（可靠抽取） ===")
    print(f"{'model':>5} | " + " | ".join(f"{c:>14}" for c in common.CONDITIONS))
    for mk in common.MODELS:
        cells = []
        for cond in common.CONDITIONS:
            _, nt, n, tc, tcr = row_of(mk, cond)
            cells.append(f"{nt/n*100:5.1f}% ({tc:>2}对/{tcr}可靠)".rjust(14))
        print(f"{mk:>5} | " + " | ".join(cells))


if __name__ == "__main__":
    main()
