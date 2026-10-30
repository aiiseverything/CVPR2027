#!/usr/bin/env python3
"""导出 7 个 benchmark 的样例：4B/9B 在 thinking 开下的 thinking + 输出，
外加 4B 在 thinking 关下已测数据集（MME / MMStar / MMBench）的输出。

产出: bench/results/examples.md
"""

import glob
import os
import re

import pandas as pd

ROOT = os.environ.get("ROOT", "/home/ml-user/workdir/CVPR2027/pre-lab")
OUT = os.path.join(ROOT, "vlmeval_out")
OUT_NT = os.path.join(ROOT, "vlmeval_out_nothink")
DEST = os.path.join(ROOT, "bench", "results", "examples.md")

MODELS = [("Qwen3.5-4B", "4B"), ("Qwen3.5-9B", "9B")]
# (显示名, 数据集名, 是否用 1K 子集)
BENCHES = [
    ("MME", "MME", False),
    ("MathVista_MINI", "MathVista_MINI", False),
    ("MathVerse_MINI (mathversemini)", "MathVerse_MINI_1K", False),
    ("MMStar", "MMStar", False),
    ("MMBench_DEV_EN_V11", "MMBench_DEV_EN_V11", False),
    ("MathVision_MINI", "MathVision_MINI", False),
    ("MMMU_DEV_VAL", "MMMU_DEV_VAL", False),
]
NOTHINK = ["MME", "MMStar", "MMBench_DEV_EN_V11"]


def latest(root, mdl, bench):
    c = [os.path.join(root, mdl, f"{mdl}_{bench}.xlsx")]
    c += glob.glob(f"{root}/{mdl}/*/{mdl}_{bench}.xlsx")
    c = [f for f in c if os.path.exists(f) and "auxmatch" not in f]
    return max(c, key=os.path.getmtime) if c else None


def split_think(resp):
    """拆成 (thinking, 最终输出)。"""
    s = str(resp)
    if "</think>" in s:
        head, _, tail = s.rpartition("</think>")
        head = head.split("<think>")[-1] if "<think>" in head else head
        return head.strip(), tail.strip()
    return "", s.strip()


def question_of(row):
    """拼出该样本的题干（含选项）。"""
    q = str(row.get("question", "")).strip()
    opts = []
    for k in "ABCD":
        if k in row and pd.notna(row[k]) and str(row[k]).strip():
            opts.append(f"{k}. {str(row[k]).strip()}")
    if opts:
        q += "\n" + "\n".join(opts)
    return q


def pick_index(df):
    """选一个可供对比的样本：优先挑两模型都答对、且回答长度适中的，便于展示。"""
    return int(df["index"].iloc[0]) if "index" in df.columns else 0


def main():
    L = []
    A = L.append
    A("# Qwen3.5-4B / 9B 各 benchmark 样例：thinking 与输出")
    A("")
    A("每个 benchmark 取**同一条样本**，并列 4B 与 9B 的思考过程与最终答案。")
    A("thinking 关的部分（4B 已测的 MME / MMStar / MMBench）附在最后。")
    A("")

    for disp, bench, _ in BENCHES:
        A("---")
        A("")
        A(f"## {disp}")
        A("")
        rows = {}
        for mdl, tag in MODELS:
            f = latest(OUT, mdl, bench)
            if not f:
                A(f"> {tag}: 无预测文件")
                continue
            df = pd.read_excel(f)
            idx = pick_index(df)
            r = df[df["index"] == idx].iloc[0] if "index" in df.columns else df.iloc[0]
            rows[tag] = (r, idx)

        if not rows:
            A("")
            continue

        any_r = next(iter(rows.values()))[0]
        gold = str(any_r.get("answer", "")).strip()
        A(f"**样本 index**: `{next(iter(rows.values()))[1]}`　**正确答案**: `{gold}`")
        A("")
        A("**题干**")
        A("")
        A("```")
        A(question_of(any_r)[:1200])
        A("```")
        A("")
        for mdl, tag in MODELS:
            if tag not in rows:
                continue
            r = rows[tag][0]
            think, ans = split_think(r.get("prediction", ""))
            A(f"### {tag}")
            A("")
            A(f"- **thinking**（{len(think)} 字符）")
            A("")
            A("```")
            A(think[:2500] + ("\n…（截断）" if len(think) > 2500 else ""))
            A("```")
            A("")
            A(f"- **最终输出**")
            A("")
            A("```")
            A(ans[:800] + ("\n…（截断）" if len(ans) > 800 else ""))
            A("```")
            A("")

    # ---- thinking 关 ----
    A("---")
    A("")
    A("# 附：thinking **关**（仅 4B，已测的 3 个数据集）")
    A("")
    for disp, bench, _ in BENCHES:
        if bench not in NOTHINK:
            continue
        f = latest(OUT_NT, "Qwen3.5-4B", bench)
        if not f:
            continue
        df = pd.read_excel(f)
        # 用与 thinking 开相同位置的样本便于对照
        f_on = latest(OUT, "Qwen3.5-4B", bench)
        idx = pick_index(pd.read_excel(f_on)) if f_on else pick_index(df)
        r = df[df["index"] == idx].iloc[0] if "index" in df.columns else df.iloc[0]

        A("---")
        A("")
        A(f"## {disp}（thinking 关）")
        A("")
        A(f"**样本 index**: `{idx}`　**正确答案**: `{str(r.get('answer','')).strip()}`")
        A("")
        A("**题干**")
        A("")
        A("```")
        A(question_of(r)[:1000])
        A("```")
        A("")
        A("**4B 输出**（thinking 关 → 无思考过程）")
        A("")
        A("```")
        A(str(r.get("prediction", ""))[:800])
        A("```")
        A("")

    os.makedirs(os.path.dirname(DEST), exist_ok=True)
    with open(DEST, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"✓ 写出 {DEST}  ({len(L)} 行)")


if __name__ == "__main__":
    main()
