#!/usr/bin/env python3
"""从 Vision-OPD-6K 抽固定种子样本，两个模型共用同一批。

产出 probe/samples.jsonl，每条含：
  idx          在 train.jsonl 中的原始行号（跨条件/跨模型的唯一连接键）
  gold_answer  正确答案字母
  bbox         [x1,y1,x2,y2]
  bbox_area    归一化面积（占原图比例），用于事后分析难度与目标大小的关系
  img_*        三种条件下的图像绝对路径
"""

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", default=os.path.join(common.PROBE_DIR, "samples.jsonl"))
    args = ap.parse_args()

    records = common.load_train_jsonl()
    print(f"train.jsonl 共 {len(records)} 条")

    rng = random.Random(args.seed)
    idxs = sorted(rng.sample(range(len(records)), min(args.n, len(records))))
    print(f"抽样 {len(idxs)} 条 (seed={args.seed})")

    samples = []
    for idx in idxs:
        r = records[idx]
        x1, y1, x2, y2 = r["bbox"]
        # 原图尺寸用于归一化 bbox 面积
        orig_abs = os.path.join(common.DATA_DIR, r["original_images"][0])
        try:
            from PIL import Image

            with Image.open(orig_abs) as im:
                W, H = im.size
        except Exception:
            W = H = None

        samples.append(
            {
                "idx": idx,
                "gold_answer": r["answer"],
                "bbox": r["bbox"],
                "bbox_area": ((x2 - x1) * (y2 - y1) / (W * H)) if (W and H) else None,
                "orig_wh": [W, H] if (W and H) else None,
                "problem": r["problem"],
                "img_bbox": os.path.join(common.DATA_DIR, r["images"][0]),
                "img_original": orig_abs,
                "img_teacher": os.path.join(common.DATA_DIR, r["teacher_images"][0]),
            }
        )

    with open(args.out, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"✓ 写入 {args.out}")


if __name__ == "__main__":
    main()
