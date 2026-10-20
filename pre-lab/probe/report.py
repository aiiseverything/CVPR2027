#!/usr/bin/env python3
"""由 per_sample.csv 生成对比报告：准确率、增益、配对显著性检验、难度分层。

产出:
  probe/results/crosstab.csv
  probe/results/report.md
"""

import csv
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def read_per_sample():
    path = os.path.join(common.RESULT_DIR, "per_sample.csv")
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def acc(rows, mk, cond):
    vals = [int(r[f"{mk}_{cond}_correct"]) for r in rows if r[f"{mk}_{cond}_correct"] != ""]
    return (sum(vals) / len(vals), len(vals), sum(vals)) if vals else (0.0, 0, 0)


def mcnemar(rows, a, b):
    """配对 McNemar 检验。a/b 为 (model, condition) 标识。
    返回 (b, c, chi2, p)，b = a 对 b 错，c = a 错 b 对。"""
    bb = cc = 0
    for r in rows:
        ka = f"{a[0]}_{a[1]}_correct"
        kb = f"{b[0]}_{b[1]}_correct"
        if r[ka] == "" or r[kb] == "":
            continue
        x, y = int(r[ka]), int(r[kb])
        if x == 1 and y == 0:
            bb += 1
        elif x == 0 and y == 1:
            cc += 1
    if bb + cc == 0:
        return bb, cc, 0.0, 1.0
    chi2 = (abs(bb - cc) - 1) ** 2 / (bb + cc)
    p = math.erfc(math.sqrt(chi2 / 2))  # 1 自由度卡方
    return bb, cc, chi2, p


def sig(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def main():
    rows = read_per_sample()
    n = len(rows)

    # ---- 交叉表：每个模型内部的条件组合 ----
    ct_path = os.path.join(common.RESULT_DIR, "crosstab.csv")
    with open(ct_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "no_image", "original", "bbox", "n", "pattern"])
        for mk in common.MODELS:
            patterns = {}
            for r in rows:
                key = tuple(
                    1 if r[f"{mk}_{c}_correct"] == "1" else 0 for c in common.CONDITIONS
                )
                patterns[key] = patterns.get(key, 0) + 1
            for key, cnt in sorted(patterns.items(), key=lambda kv: -kv[1]):
                w.writerow([mk, key[0], key[1], key[2], cnt, "".join(map(str, key))])
    print(f"✓ {ct_path}")

    # ---- 报告 ----
    lines = []
    A = lines.append
    A("# Vision-OPD-6K 细粒度感知探针：有图 / 无图 × Qwen3.5-4B/9B\n")
    A(f"样本数: **{n}**（两模型共用同一批，固定种子）｜thinking 开启｜temperature=0\n")

    def trunc_rate(mk, cond):
        vals = [int(r[f"{mk}_{cond}_trunc"]) for r in rows if r[f"{mk}_{cond}_trunc"] != ""]
        return (sum(vals) / len(vals), len(vals), sum(vals)) if vals else (0.0, 0, 0)

    def acc_answered(mk, cond):
        """只看自然收尾（finish_reason != length）的样本。"""
        sub = [r for r in rows if r[f"{mk}_{cond}_trunc"] == "0" and r[f"{mk}_{cond}_pred"] != ""]
        if not sub:
            return 0.0, 0, 0
        nc = sum(1 for r in sub if r[f"{mk}_{cond}_pred"] == r["gold_answer"])
        return nc / len(sub), len(sub), nc

    def trunc_reliable(mk, cond):
        """撞顶但答对的条数，以及其中来自可靠抽取(tag/explicit)的条数。
        可靠抽取 = 模型确实写出了 "Answer: X" 之后继续啰嗦被截断；
        tail/fallback 是从半截推理里捡字母，准确率约等于 1/4 随机。"""
        sub = [r for r in rows if r[f"{mk}_{cond}_trunc"] == "1"]
        tc = [r for r in sub if r[f"{mk}_{cond}_pred"] == r["gold_answer"]]
        tcr = [r for r in tc if r[f"{mk}_{cond}_method"] in ("tag", "explicit")]
        return len(tc), len(tcr), len(sub)

    A("## 1. 准确率\n")
    A("所有条件共享 **4096** thinking 预算。**主指标「准确率」不因截断额外惩罚** —— "
      "只要在预算内给出了答案就算数，否则比较的会变成「谁的思维链更短」。\n")
    A("下表额外给出「撞顶率」和「已答完准确率」以便区分失败模式。注意撞顶样本中"
      "约 3/4 的答案是从半截推理里「捡」出来的字母（准确率 ≈ 25% 随机水平），"
      "「撞顶答对」列里标注了其中真正由模型明确给出答案的数量。\n")
    A("| 模型 | 指标 | no_image（盲答） | original（原图） | bbox（带红框） |")
    A("|---|---|---|---|---|")
    stats = {}
    for mk in common.MODELS:
        cells = []
        for cond in common.CONDITIONS:
            a, cnt, nc = acc(rows, mk, cond)
            stats[(mk, cond)] = (a, cnt, nc)
            cells.append(f"{a*100:.1f}% ({nc}/{cnt})")
        A(f"| {mk} | **准确率**（主指标） | " + " | ".join(cells) + " |")
        cells = []
        for cond in common.CONDITIONS:
            a, cnt, nc = acc_answered(mk, cond)
            cells.append(f"{a*100:.1f}% ({nc}/{cnt})")
        A(f"| {mk} | 准确率(仅已答完) | " + " | ".join(cells) + " |")
        cells = []
        for cond in common.CONDITIONS:
            t, cnt, nt = trunc_rate(mk, cond)
            cells.append(f"{t*100:.1f}% ({nt}/{cnt})")
        A(f"| {mk} | **撞顶率** | " + " | ".join(cells) + " |")
        cells = []
        for cond in common.CONDITIONS:
            tc, tcr, nt = trunc_reliable(mk, cond)
            cells.append(f"{tc} 条 (其中 {tcr} 条可靠)")
        A(f"| {mk} | 撞顶但答对 | " + " | ".join(cells) + " |")
    A("")

    # ---- 口径敏感性 ----
    RELIABLE_M = {"tag", "explicit"}

    def conv(mk, cond):
        """三种口径下的 (准确率, 样本数)。"""
        out = {}
        for mode in "ABC":
            n = ok = 0
            for r in rows:
                key = f"{mk}_{cond}"
                if r[f"{key}_correct"] == "":
                    continue
                gold = r["gold_answer"]
                pred = r[f"{key}_pred"]
                trunc = r[f"{key}_trunc"] == "1"
                meth = r[f"{key}_method"]
                if mode == "A":
                    n += 1
                    ok += 1 if (not trunc and pred == gold) else 0
                elif mode == "B":
                    n += 1
                    ok += 1 if pred == gold else 0
                else:  # C：只在模型确实给出答案时才计入
                    if (not trunc) or meth in RELIABLE_M:
                        n += 1
                        ok += 1 if pred == gold else 0
            out[mode] = (ok / n if n else 0.0, n)
        return out

    A("## 2. 口径敏感性：截断样本怎么算，会不会改变结论？\n")
    A("- **A** 撞顶一律计错（最保守，但把「没答完」等同于「答错」）\n")
    A("- **B** 撞顶但答案匹配金标算对 ← **本报告主口径**\n")
    A("- **C** 只在模型确实给出答案时才计入样本；撞顶且答案是从半截推理里捡来的整条剔除\n")
    A("")
    A("| 组 | A 撞顶计错 | **B 主口径** | C 仅计已作答 | C 的样本数 |")
    A("|---|---|---|---|---|")
    convs = {(mk, c): None for mk in common.MODELS for c in common.CONDITIONS}
    for mk in common.MODELS:
        for cond in common.CONDITIONS:
            cv = conv(mk, cond)
            convs[(mk, cond)] = cv
            A(f"| {mk} / {cond} | {cv['A'][0]*100:.1f}% | **{cv['B'][0]*100:.1f}%** | "
              f"{cv['C'][0]*100:.1f}% | {cv['C'][1]} |")
    A("")

    A("| 关键对比 | A | B（主口径） | C |")
    A("|---|---|---|---|")
    def mcnemar_conv(a, b, mode):
        bb = cc = 0
        for r in rows:
            ka, kb = f"{a[0]}_{a[1]}", f"{b[0]}_{b[1]}"
            if r[f"{ka}_correct"] == "" or r[f"{kb}_correct"] == "":
                continue
            gold = r["gold_answer"]
            def ok(key):
                pred = r[f"{key}_pred"]
                trunc = r[f"{key}_trunc"] == "1"
                if mode == "A":
                    return (not trunc) and pred == gold
                if mode == "B":
                    return pred == gold
                if not trunc or r[f"{key}_method"] in RELIABLE_M:
                    return pred == gold
                return None
            X, Y = ok(ka), ok(kb)
            if X is None or Y is None:
                continue
            if X and not Y:
                bb += 1
            elif Y and not X:
                cc += 1
        if bb + cc == 0:
            return 1.0
        chi2 = (abs(bb - cc) - 1) ** 2 / (bb + cc)
        return math.erfc(math.sqrt(chi2 / 2))

    for label, ka, kb in [
        ("4B: bbox − original", ("4B", "original"), ("4B", "bbox")),
        ("9B: bbox − original", ("9B", "original"), ("9B", "bbox")),
        ("9B − 4B @ no_image", ("4B", "no_image"), ("9B", "no_image")),
        ("9B − 4B @ bbox", ("4B", "bbox"), ("9B", "bbox")),
    ]:
        cells = [f"{mcnemar_conv(ka, kb, m):.3f} {sig(mcnemar_conv(ka, kb, m))}" for m in "ABC"]
        A(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    A("")
    A("> 注意：**红框定位提示（bbox − original）的增益三种口径下都不稳健** —— "
      "A 恰好压线显著，B/C 均不显著。这个结论本身值得写进论文。\n")

    A("## 3. 增益分解（主口径 B）\n")
    A("| 对比 | 含义 | 增量 | McNemar b/c | p | 显著性 |")
    A("|---|---|---|---|---|---|")
    comparisons = []
    for mk in common.MODELS:
        comparisons.append((
            f"{mk}: original − no_image", (mk, "no_image"), (mk, "original"), "图像本身的贡献"))
        comparisons.append((
            f"{mk}: bbox − original", (mk, "original"), (mk, "bbox"), "红框定位提示的额外贡献"))
    for cond in common.CONDITIONS:
        comparisons.append((
            f"9B − 4B @ {cond}", ("4B", cond), ("9B", cond), "模型规模差异"))
    for label, a, b, meaning in comparisons:
        aa = stats[a][0]
        ab = stats[b][0]
        bb, cc, chi2, p = mcnemar(rows, a, b)
        A(f"| {label} | {meaning} | {(ab-aa)*100:+.1f} pp | {bb}/{cc} | {p:.2e} | {sig(p)} |")
    A("")

    A("## 4. 逐样本对错模式（每行一个模式，000 = 三档全错）\n")
    for mk in common.MODELS:
        patterns = {}
        for r in rows:
            key = tuple(1 if r[f"{mk}_{c}_correct"] == "1" else 0 for c in common.CONDITIONS)
            patterns[key] = patterns.get(key, 0) + 1
        A(f"**{mk}**（列序: no_image, original, bbox）\n")
        A("| 模式 | 数量 | 占比 | 解读 |")
        A("|---|---|---|---|")
        for key, cnt in sorted(patterns.items(), key=lambda kv: -kv[1]):
            pat = "".join(map(str, key))
            interp = {
                "000": "三档全错 — 图像没能救回来",
                "001": "只有带红框才对 — 定位提示起了决定性作用",
                "011": "有图就对，红框无额外帮助 — 模型能自己找到目标",
                "111": "三档全对 — 可能在盲答（语言先验）或太简单",
                "100": "只有盲答对、有图反而错 — 图像构成干扰",
                "101": "盲答与红框都对、原图错 — 反常，值得逐条看",
                "110": "盲答与原图对、红框错 — 反常，值得逐条看",
                "010": "只有原图对 — 反常，值得逐条看",
            }.get(pat, "")
            A(f"| `{pat}` | {cnt} | {cnt/n*100:.1f}% | {interp} |")
        A("")

    A("## 5. 难度分层：图像增益是否随目标变小而增大？\n")
    areas = sorted(float(r["bbox_area"]) for r in rows if r["bbox_area"])
    if len(areas) >= 3:
        t1, t2 = areas[len(areas) // 3], areas[2 * len(areas) // 3]
        A(f"按归一化 bbox 面积三分位分组（阈值 {t1:.4f} / {t2:.4f}）\n")
        A("| 模型 | 组 | n | no_image | original | bbox | original−no_image |")
        A("|---|---|---|---|---|---|---|")
        for mk in common.MODELS:
            for name, lo, hi in [("小目标", -1, t1), ("中目标", t1, t2), ("大目标", t2, 2)]:
                sub = [
                    r for r in rows
                    if r["bbox_area"] and lo < float(r["bbox_area"]) <= hi
                ]
                if not sub:
                    continue
                cells = []
                for cond in common.CONDITIONS:
                    vals = [int(r[f"{mk}_{cond}_correct"]) for r in sub if r[f"{mk}_{cond}_correct"] != ""]
                    cells.append(f"{sum(vals)/len(vals)*100:.1f}%")
                gain = float(cells[1][:-1]) - float(cells[0][:-1])
                A(f"| {mk} | {name} | {len(sub)} | " + " | ".join(cells) + f" | {gain:+.1f} pp |")
        A("")

    out = os.path.join(common.RESULT_DIR, "report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"✓ {out}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
