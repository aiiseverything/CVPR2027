#!/usr/bin/env python3
"""把 MathVerse 的 checkpoint 转成 VLMEvalKit 判分所需的预测 xlsx。

背景：stop_mathverse_at_1k.sh 在 index 1000 处直接杀掉作业，VLMEvalKit 来不及
写最终 xlsx，只留下 `<模型>_MathVerse_MINI_checkpoint.pkl`。判分需要预测文件，
所以这里把 checkpoint 和数据集拼回去，产出标准格式的 xlsx。

只取前 1000 条（= mathversemini 的定义）。

用法: python3 bench/finalize_mathverse.py
"""

import glob
import os
import pickle
import sys

import pandas as pd

ROOT = os.environ.get("ROOT", "/home/ml-user/workdir/CVPR2027/pre-lab")
OUT = os.path.join(ROOT, "vlmeval_out")
LMUDATA = os.path.join(ROOT, "lmudata")
LIMIT = 1000
BENCH = "MathVerse_MINI"


def main():
    made = 0
    for mdl in ["Qwen3.5-4B", "Qwen3.5-9B"]:
        dst = os.path.join(OUT, mdl, f"{mdl}_{BENCH}.xlsx")
        if os.path.exists(dst):
            try:
                d = pd.read_excel(dst)
                if len(d) >= LIMIT:
                    print(f"  {mdl}: 已有 xlsx（{len(d)} 行），跳过")
                    continue
            except Exception:
                pass

        # 汇总所有 checkpoint（可能有多个时间戳目录）
        preds = {}
        for p in glob.glob(f"{OUT}/{mdl}/*/{mdl}_{BENCH}_checkpoint.pkl"):
            try:
                with open(p, "rb") as f:
                    d = pickle.load(f)
                for k, v in d.items():
                    kk = int(k) if str(k).isdigit() else k
                    if isinstance(kk, int) and kk <= LIMIT:
                        if kk not in preds or not str(preds[kk]).startswith("Failed to obtain"):
                            preds[kk] = v
            except Exception as e:
                print(f"    （跳过损坏的 {os.path.basename(p)}: {type(e).__name__}）")

        if not preds:
            print(f"  {mdl}: 无可用 checkpoint，跳过")
            continue

        # 拼数据集
        src = os.path.join(LMUDATA, f"{BENCH}.tsv")
        data = pd.read_csv(src, sep="\t")
        data = data[data["index"] <= LIMIT].copy()
        data["prediction"] = data["index"].map(
            lambda i: str(preds.get(int(i), "Failed to obtain answer via API."))
        )

        os.makedirs(os.path.dirname(dst), exist_ok=True)
        data.to_excel(dst, index=False)
        nfail = int((data["prediction"] == "Failed to obtain answer via API.").sum())
        print(f"  ✓ {mdl}: 写出 {len(data)} 行 -> {dst}  失败={nfail}")
        made += 1

    return made


if __name__ == "__main__":
    sys.exit(0 if main() >= 0 else 1)
