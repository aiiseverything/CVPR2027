#!/usr/bin/env python3
"""
把 Vision-OPD-6K 的 train.jsonl 转成训练用 train.parquet。

字段 schema 与官方 repo/scripts/prepare_data.py 的 build_record() 保持一致
（该文件已被阅读以确认格式；这里独立实现，不执行上游代码）。

用法: python3 make_parquet.py --data-dir data/Vision-OPD-6K
"""

import argparse
import json
import os
import sys

import datasets

REMOVE_HINT = (
    "Only focus on the objects inside the red bounding box in the image "
    "to answer this question."
)


def clean_question(problem: str) -> str:
    text = (problem or "").replace("<image>", "").strip()
    text = text.replace(f"\n\n{REMOVE_HINT}", "")
    text = text.replace(REMOVE_HINT, "")
    return text.strip()


def build_record(item: dict, data_dir: str) -> dict:
    image_path = os.path.join(data_dir, item["images"][0])
    teacher_path = os.path.join(data_dir, item["teacher_images"][0])
    return {
        "data_source": "zwz_rl_vqa_bbox_teacher",
        "prompt": [{"role": "user", "content": item["problem"]}],
        "images": [{"path": image_path}],
        "bbox_images": [{"path": teacher_path}],
        "ability": "visual_question_answering",
        "reward_model": {
            "style": "none",
            "ground_truth": item.get("answer", ""),
        },
        "extra_info": {
            "answer": item.get("answer", ""),
            "question": clean_question(item.get("problem", "")),
            "source_extra_info": item.get("extra_info", {}),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/Vision-OPD-6K")
    ap.add_argument("--output", default=None, help="默认 <data-dir>/train.parquet")
    args = ap.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    jsonl_path = os.path.join(data_dir, "train.jsonl")
    if not os.path.exists(jsonl_path):
        sys.exit(f"✗ 找不到 {jsonl_path}")

    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(build_record(json.loads(line), data_dir))

    out = args.output or os.path.join(data_dir, "train.parquet")
    datasets.Dataset.from_list(records).to_parquet(out)
    print(f"✓ 已写入 {len(records)} 条 -> {out}")


if __name__ == "__main__":
    main()
