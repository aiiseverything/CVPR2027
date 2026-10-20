#!/usr/bin/env python3
"""查看生成的例子。

    python probe/view.py                      # 列出各结果文件与条数
    python probe/view.py --n 3                # 每个 模型×条件 各看 3 条
    python probe/view.py --idx 4              # 看某个样本在 6 组下的横向对比
    python probe/view.py --model-key 4B --condition bbox --n 5
    python probe/view.py --idx 4 --full       # 不截断推理过程

结果文件: probe/out/{model}__{condition}.jsonl
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def show(rec, full=False, width=700):
    resp = rec.get("raw_response") or ""
    reason, sep, answer = resp.rpartition("</think>")
    ok = rec.get("pred_answer") == rec.get("gold_answer")
    mark = "✓" if ok else "✗"

    print("─" * 78)
    print(f"{mark} idx={rec['idx']}  [{rec['model']} / {rec['condition']}]  "
          f"金标={rec['gold_answer']}  预测={rec.get('pred_answer')}  "
          f"({rec.get('extract_method')}, {rec.get('n_tokens')} tok, {rec.get('finish_reason')})")
    print("─" * 78)
    if rec.get("condition") == "no_image":
        print("【输入图像】无")
    print("【问题】")
    print(rec["prompt"])
    print()
    print("【模型推理过程】" + ("" if sep else "  (无 </think>，整段为输出)"))
    body = reason if sep else resp
    print(body if full else (body[:width] + ("\n  …(截断，加 --full 看完整)" if len(body) > width else "")))
    if sep:
        print()
        print("【最终输出】" + answer.strip())
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-key", choices=list(common.MODELS))
    ap.add_argument("--condition", choices=common.CONDITIONS)
    ap.add_argument("--n", type=int, default=0, help="每个组合展示几条")
    ap.add_argument("--idx", type=int, help="只看这个样本 idx（跨 6 组对比）")
    ap.add_argument("--full", action="store_true", help="不截断推理过程")
    args = ap.parse_args()

    # 载入所有可用结果
    runs = {}
    for mk in common.MODELS:
        for cond in common.CONDITIONS:
            p = common.out_path(mk, cond)
            recs = common.read_jsonl(p)
            if recs:
                runs[(mk, cond)] = {r["idx"]: r for r in recs}

    if not runs:
        print("还没有任何结果。等 probe/out/ 里出现 .jsonl 后再来。")
        print(f"目录: {common.OUT_DIR}")
        return

    print(f"可用结果（{common.OUT_DIR}）:")
    for (mk, cond), d in sorted(runs.items()):
        n_ok = sum(1 for r in d.values() if r.get("pred_answer") == r.get("gold_answer"))
        print(f"  {mk:>3} / {cond:<9} {len(d):>5} 条   正确 {n_ok}"
              f" ({n_ok/len(d)*100:.0f}%)")
    print()

    # 单样本跨组对比
    if args.idx is not None:
        print(f"########## 样本 idx={args.idx} 的横向对比 ##########\n")
        for mk in common.MODELS:
            for cond in common.CONDITIONS:
                r = runs.get((mk, cond), {}).get(args.idx)
                if r:
                    show(r, args.full)
        return

    # 逐组展示
    keys = [(mk, c) for mk in common.MODELS for c in common.CONDITIONS]
    if args.model_key:
        keys = [k for k in keys if k[0] == args.model_key]
    if args.condition:
        keys = [k for k in keys if k[1] == args.condition]

    n = args.n or (1 if args.idx is None and not args.n else args.n)
    for key in keys:
        if key not in runs:
            continue
        recs = list(runs[key].values())[:n]
        print(f"\n##################### {key[0]} / {key[1]} #####################\n")
        for r in recs:
            show(r, args.full)


if __name__ == "__main__":
    main()
