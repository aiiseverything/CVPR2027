#!/usr/bin/env python3
"""提取「某模型答对、另一模型答错」的特例，产出带图片和 prompt 的 HTML 画廊。

    python3 probe/cases.py                          # 4B 对 / 9B 错，全部条件
    python3 probe/cases.py --condition bbox          # 只看 bbox 条件
    python3 probe/cases.py --winner 9B --loser 4B    # 反过来
    python3 probe/cases.py --exclude-trunc           # 排除输家撞顶（没答完）的样本

产出:
    probe/results/cases_{winner}_beats_{loser}.html   自包含（图片内嵌 base64）
    probe/results/cases_{winner}_beats_{loser}.csv    逐条明细
"""

import argparse
import base64
import csv
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

COND_NAME = {
    "no_image": "无图（盲答）",
    "original": "原图（无红框）",
    "bbox": "带红框全图 ← 学生/标准推理输入",
    "teacher": "裁剪区域（教师 privileged view）",
}


# 由 main() 按案例总数调整：案例多时用小图，避免 HTML 过大拖慢浏览器
THUMB_W = 600
THUMB_Q = 72


def thumb_b64(path, max_w=None, quality=None):
    """读图 -> 缩放 -> JPEG -> base64，用于内嵌 HTML。"""
    if not path or not os.path.exists(path):
        return None
    max_w = max_w or THUMB_W
    quality = quality or THUMB_Q
    try:
        from PIL import Image

        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > max_w:
                im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=quality)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def split_think(resp):
    """拆成 (推理过程, 最终输出)。"""
    if not resp:
        return "", ""
    head, sep, tail = resp.rpartition("</think>")
    if sep:
        return head, tail.strip()
    return resp, ""


def load(mk, cond):
    return {r["idx"]: r for r in common.read_jsonl(common.out_path(mk, cond))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winner", default="4B", choices=list(common.MODELS))
    ap.add_argument("--loser", default="9B", choices=list(common.MODELS))
    ap.add_argument("--condition", choices=common.CONDITIONS,
                    help="默认全部条件")
    ap.add_argument("--exclude-trunc", action="store_true",
                    help="排除输家撞顶（finish_reason=length）的样本，只看真正答错的")
    ap.add_argument("--max", type=int, default=30, help="每个条件最多展示多少条")
    args = ap.parse_args()

    conds = [args.condition] if args.condition else common.CONDITIONS
    samples = {s["idx"]: s for s in common.load_samples()}

    results = {}
    for cond in conds:
        w = load(args.winner, cond)
        l = load(args.loser, cond)
        if not w or not l:
            print(f"跳过 {cond}（结果缺失）")
            continue
        cases = []
        for idx in sorted(set(w) & set(l)):
            wr, lr = w[idx], l[idx]
            gold = wr["gold_answer"]
            # 口径 B（与 report.py 主口径一致）：截断不额外惩罚，
            # 只看抽取出的答案是否匹配金标。
            w_ok = wr["pred_answer"] == gold
            l_trunc = lr["finish_reason"] == "length"
            l_bad = lr["pred_answer"] != gold
            if not (w_ok and l_bad):
                continue
            if args.exclude_trunc and l_trunc:
                continue
            cases.append((idx, wr, lr, l_trunc))
        # 小目标排前面 —— 越小的框越考验细粒度感知，也越能说明问题
        cases.sort(key=lambda c: samples[c[0]].get("bbox_area") or 1.0)
        results[cond] = cases
        n_t = sum(1 for c in cases if c[3])
        print(f"{cond:<10} {args.winner} 对 / {args.loser} 错: {len(cases):>4} 条"
              f"  (其中 {args.loser} 撞顶 {n_t})")

    base = os.path.join(common.RESULT_DIR, f"cases_{args.winner}_beats_{args.loser}")
    os.makedirs(common.RESULT_DIR, exist_ok=True)

    # 自适应缩略图尺寸：案例多时用小图，防止 HTML 膨胀到几十 MB
    global THUMB_W, THUMB_Q
    n_show = sum(min(len(v), args.max) for v in results.values())
    if n_show > 120:
        THUMB_W, THUMB_Q = 430, 65
    elif n_show > 60:
        THUMB_W, THUMB_Q = 520, 70
    print(f"  生成 HTML: {n_show} 个案例, 缩略图 {THUMB_W}px/q{THUMB_Q}")

    # ---------- CSV ----------
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["condition", "idx", "gold", f"{args.winner}_pred", f"{args.loser}_pred",
                    f"{args.loser}_trunc", "bbox", "bbox_area", "img_bbox", "img_original",
                    f"{args.winner}_response", f"{args.loser}_response"])
        for cond in conds:
            for idx, wr, lr, lt in results.get(cond, []):
                s = samples.get(idx, {})
                w.writerow([cond, idx, wr["gold_answer"], wr["pred_answer"], lr["pred_answer"],
                            int(lt), s.get("bbox"), s.get("bbox_area"),
                            s.get("img_bbox"), s.get("img_original"),
                            wr["raw_response"], lr["raw_response"]])
    print(f"\n✓ {base}.csv")

    # ---------- HTML ----------
    h = []
    h.append("""<!doctype html><meta charset="utf-8">
<title>特例：%s 答对 / %s 答错</title>
<style>
 body{font:14px/1.6 -apple-system,"Segoe UI",Roboto,"Noto Sans CJK SC",sans-serif;
      max-width:1180px;margin:0 auto;padding:24px;background:#fafafa;color:#1a1a1a}
 h1{font-size:22px} h2{font-size:17px;margin-top:34px;padding-bottom:6px;
      border-bottom:2px solid #e0e0e0}
 .sum{background:#fff;border:1px solid #e4e4e4;border-radius:8px;padding:14px 18px;margin:16px 0}
 .case{background:#fff;border:1px solid #e4e4e4;border-radius:10px;
       padding:16px 18px;margin:18px 0}
 .hd{font-weight:600;margin-bottom:10px;font-size:15px}
 .badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:12px;
        font-weight:600;vertical-align:middle}
 .g{background:#e6f4ea;color:#137333} .b{background:#fce8e6;color:#c5221f}
 .t{background:#fef7e0;color:#b06000}
 .row{display:flex;gap:16px;flex-wrap:wrap}
 .col{flex:1 1 320px;min-width:280px}
 img{max-width:100%;border:1px solid #ddd;border-radius:6px;display:block}
 .cap{font-size:12px;color:#666;margin:4px 0 10px}
 pre{white-space:pre-wrap;word-wrap:break-word;background:#f6f8fa;border:1px solid #e4e4e4;
     border-radius:6px;padding:10px;font-size:12.5px;max-height:320px;overflow:auto;
     font-family:ui-monospace,Menlo,Consolas,monospace}
 .q{background:#eef4ff;border:1px solid #d6e2ff;border-radius:6px;padding:10px;
    font-size:13px;white-space:pre-wrap}
 details{margin-top:6px} summary{cursor:pointer;font-size:12.5px;color:#555}
 .ans{font-weight:700}
 .ok{color:#137333} .no{color:#c5221f}
</style>""")
    h.append(f"<h1>特例：<b>{args.winner}</b> 答对，而 <b>{args.loser}</b> 答错</h1>")

    total = sum(len(v) for v in results.values())
    h.append('<div class="sum"><b>总计 %d 条</b>（同一批 1000 样本、同 prompt、同 4096 thinking 预算）<br>'
             % total)
    for cond in conds:
        cs = results.get(cond, [])
        if not cs:
            continue
        nt = sum(1 for c in cs if c[3])
        h.append(f"&nbsp;&nbsp;· {COND_NAME.get(cond,cond)}：<b>{len(cs)}</b> 条"
                 f"（其中 {args.loser} 撞顶没答完 {nt} 条）<br>")
    h.append("</div>")

    for cond in conds:
        cs = results.get(cond, [])
        if not cs:
            continue
        h.append(f"<h2>{COND_NAME.get(cond,cond)} — {len(cs)} 条</h2>")
        for idx, wr, lr, lt in cs[: args.max]:
            s = samples.get(idx, {})
            gold = wr["gold_answer"]
            w_think, w_ans = split_think(wr["raw_response"])
            l_think, l_ans = split_think(lr["raw_response"])
            h.append('<div class="case">')
            h.append(f'<div class="hd">idx={idx} &nbsp; 正确答案 '
                     f'<span class="badge g">{gold}</span> &nbsp;|&nbsp; '
                     f'{args.winner} 答 <span class="badge g">{wr["pred_answer"]}</span> '
                     f'&nbsp; {args.loser} 答 '
                     f'<span class="badge b">{lr["pred_answer"]}</span>')
            # 截断只是标注，不再是排除理由：撞顶时那个字母就是它在预算内给出的答案
            if lt:
                h.append(' <span class="badge t">该回复撞 4096 上限被截断</span>')
            if wr["finish_reason"] == "length":
                h.append(f' <span class="badge t">（{args.winner} 的回复也被截断）</span>')
            if s.get("bbox"):
                a = s.get("bbox_area")
                h.append(f'<br><span style="font-weight:400;font-size:12.5px;color:#555">'
                         f'bbox={s["bbox"]}'
                         + (f' 归一化面积={a:.4f}' if a is not None else '') + '</span>')
            h.append("</div>")

            h.append('<div class="row">')
            # 左：模型实际看到的图
            h.append('<div class="col"><div class="cap"><b>模型看到的图</b></div>')
            if cond == "no_image":
                h.append('<div style="padding:40px;text-align:center;color:#999;'
                         'border:1px dashed #ccc;border-radius:6px">无图</div>')
            else:
                key = {"bbox": "img_bbox", "original": "img_original",
                       "teacher": "img_teacher"}.get(cond)
                b = thumb_b64(s.get(key))
                if b:
                    h.append(f'<img src="data:image/jpeg;base64,{b}">')
            h.append("</div>")
            # 右：参考裁剪图（题目所问的细节）
            if cond != "teacher":
                b = thumb_b64(s.get("img_teacher"))
                if b:
                    h.append('<div class="col"><div class="cap"><b>参考：题目所指细节（裁剪）'
                             '</b></div>'
                             f'<img src="data:image/jpeg;base64,{b}">')
                    h.append('<div class="cap">这是数据集的 teacher 裁剪图，仅作定位参考</div></div>')

            h.append("</div>")
            h.append('<div class="cap" style="margin-top:10px"><b>Prompt（两模型完全相同）</b></div>')
            h.append(f'<div class="q">{wr["prompt"]}</div>')

            h.append('<div class="row" style="margin-top:12px">')
            for role, rec, ans, think, cls, trunc in [
                (args.winner, wr, w_ans, w_think, "ok", False),
                (args.loser, lr, l_ans, l_think, "no", lt),
            ]:
                h.append('<div class="col">')
                shown = f'<span class="ans {cls}">{ans if ans else "(未抽到答案)"}</span>'
                tr_note = (" · <span style='color:#b06000'>撞顶被截断，以上为截断前文本</span>"
                           if trunc else "")
                h.append(f'<div class="cap"><b>{role}</b> 最终输出: {shown} '
                         f'· {rec.get("n_tokens")} tok · {rec.get("finish_reason")}'
                         f'{tr_note}'
                         f' · 抽取方式 {rec.get("extract_method")}</div>')
                h.append("<details><summary>展开推理过程"
                         + ("（被截断）" if trunc else "") + "</summary>"
                         f"<pre>{think}</pre></details>")
                h.append("</div>")
            h.append("</div></div>")

    with open(base + ".html", "w", encoding="utf-8") as f:
        f.write("\n".join(h))
    print(f"✓ {base}.html")


if __name__ == "__main__":
    main()
