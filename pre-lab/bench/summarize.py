#!/usr/bin/env python3
"""从判分日志汇总最终成绩，输出 markdown 表。

MME 用 perception+reasoning 合计（VLMEvalKit 的主指标只报子项 OCR）；
其余 benchmark 取判分汇总表里的 primary_metric_value。
同时报告 API 失败数（0 才可信）。
"""

import glob
import os
import re
import sys

import pandas as pd

ROOT = os.environ.get("ROOT", "/home/ml-user/workdir/CVPR2027/pre-lab")
OUT = os.path.join(ROOT, "vlmeval_out")
LOGS = os.path.join(ROOT, "logs", "bench")
FAIL = "Failed to obtain answer via API."

BENCHMARKS = ["MME", "MathVista_MINI", "MathVerse_MINI", "MMStar",
              "MMBench_DEV_EN_V11", "MathVision_MINI", "MMMU_DEV_VAL"]
MODELS = [("Qwen3.5-4B", "4B"), ("Qwen3.5-9B", "9B")]


def mme_total(mdl):
    for f in glob.glob(f"{OUT}/{mdl}/*/{mdl}_MME_score.csv"):
        try:
            d = pd.read_csv(f)
            return float(d["perception"].iloc[0]) + float(d["reasoning"].iloc[0])
        except Exception:
            pass
    return None


def _strip_prefix(line):
    """去掉日志行前缀 '[时间] LEVEL - logger: '，留下正文。"""
    return re.sub(r"^\[[^\]]+\]\s*(?:INFO|ERROR|WARNING)\s*-\s*[^:]+:\s?", "", line)


def primary_metric(mdl, bench):
    """从判分日志的汇总表里取该 benchmark 那一行的 primary_metric_value。

    注意：日志每行都带 '[时间] INFO - logger:' 前缀，所以不能简单地
    用 'Run Summary Report:(.*?)(\\n\\[)' 截取 —— 那会在下一行前缀处就断掉。
    正确做法是先剥离前缀，再在窗口内找表格行。
    """
    for name in (f"judge_{mdl}_{bench}.log", f"judge_{mdl}_{bench}_1K.log"):
        p = os.path.join(LOGS, name)
        if not os.path.exists(p):
            continue
        lines = [_strip_prefix(l) for l in open(p, errors="ignore")]
        start = next((i for i, l in enumerate(lines) if "Run Summary Report" in l), None)
        if start is None:
            continue
        for line in lines[start:start + 40]:
            if not line.strip().startswith(bench):
                continue
            if "Incomplete" in line or "No infer result" in line:
                return None, line.strip()[:60]
            parts = re.split(r"\s{2,}", line.strip())
            if len(parts) >= 5 and parts[4] not in ("-", ""):
                return parts[4], None
        return None, "未找到该行"
    return None, "无判分日志"


def _latest(mdl, bench):
    """找该 (model, bench) 最新的预测文件 —— 既可能在扁平的软链路径，
    也可能在时间戳子目录里，取 mtime 最新的那个。"""
    cands = [os.path.join(OUT, mdl, f"{mdl}_{bench}.xlsx")]
    cands += [f for f in glob.glob(f"{OUT}/{mdl}/*/{mdl}_{bench}.xlsx")]
    cands = [f for f in cands if os.path.exists(f) and "auxmatch" not in f]
    return max(cands, key=os.path.getmtime) if cands else None


def mathverse_score(mdl):
    """MathVerse 按 5 个 problem_version 分别报分，取平均作为总成绩。
    实际跑的是 MathVerse_MINI_1K（= mathversemini，前 1000 行）。"""
    for f in glob.glob(f"{OUT}/{mdl}/*/{mdl}_MathVerse_MINI_1K_DeepSeek*_score.csv"):
        try:
            d = pd.read_csv(f)
            return d["Overall"].mean()
        except Exception:
            pass
    return None


def fail_count(mdl, bench):
    f = _latest(mdl, bench)
    if not f:
        return None
    try:
        d = pd.read_excel(f)
        return int((d["prediction"].astype(str) == FAIL).sum())
    except Exception:
        return None


def main():
    L = []
    A = L.append
    A("# Qwen3.5-4B / 9B 多模态评测汇总")
    A("")
    A("推理: vLLM OpenAI 服务端 · VLMEvalKit API 模式 ｜ "
      "thinking 开启 · temperature=1.0 · max_tokens=32768 ｜ "
      "判分: DeepSeek-V4.1-Flash（reasoning 关）")
    A("")
    A("| benchmark | 4B | 9B |")
    A("|---|---|---|")
    for b in BENCHMARKS:
        cells = []
        for mdl, tag in MODELS:
            if b == "MME":
                v = mme_total(mdl)
                cells.append(f"{v:.1f}" if v is not None else "—")
            elif b == "MathVerse_MINI":
                v = mathverse_score(mdl)
                cells.append(f"{v:.1f}" if v is not None else "—")
            else:
                v, note = primary_metric(mdl, b)
                cells.append(str(v) if v else f"— ({note})" if note else "—")
        A(f"| {b} | {cells[0]} | {cells[1]} |")
    A("")
    A("## API 失败数（必须为 0 才可信）")
    A("")
    A("| benchmark | 4B | 9B |")
    A("|---|---|---|")
    for b in BENCHMARKS:
        cells = []
        for mdl, _ in MODELS:
            n = fail_count(mdl, "MathVerse_MINI_1K" if b == "MathVerse_MINI" else b)
            cells.append("—" if n is None else (f"{n} ⚠" if n else "0 ✓"))
        A(f"| {b} | {cells[0]} | {cells[1]} |")
    A("")
    A("> 注：MME 为 perception + reasoning 合计（VLMEvalKit 主指标只报 OCR 子项）。")
    A("> thinking 模式下的数字与公开榜单的非 thinking 数字不可直接比较。")

    print("\n".join(L))


if __name__ == "__main__":
    main()
