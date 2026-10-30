#!/usr/bin/env python3
"""Optionally add original RefCOCO train images and their expressions for v3.

Run only when the existing grounding pool cannot meet the difficulty quota.
The pinned annotation parquet is LFS-verified; its bbox is checked against the
original raw COCO annotation. Existing VQAv2 images may be reused only when the
COCO image ID, decoded dimensions, and stored pixel digest agree. All new files
stay in perception_v3_sources/. No GPU or model inference is performed.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
from pathlib import Path
import random
import shutil
import struct
import time

from PIL import Image
import pyarrow.parquet as pq
import requests

import expand_perception_v3_sources as common

HERE, OUT = common.HERE, common.OUT
SPEC = {"repo": "jxu124/refcoco", "original_repo": "jxu124/refcoco",
        "revision": "9fba8200c5326e996f789191f095bd464ef1d09e", "prefix": "data/train-", "task": "bbox"}
common.SOURCES["refcoco"] = SPEC


def read_records(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def checked_bbox(row):
    if row["split"] != "train":
        raise ValueError("Non-train RefCOCO annotation")
    info, annotation = json.loads(row["raw_image_info"]), json.loads(row["raw_anns"])
    if int(info["id"]) != int(row["image_id"]) or int(annotation["image_id"]) != int(row["image_id"]):
        raise ValueError("RefCOCO image identity disagreement")
    x, y, width, height = annotation["bbox"]
    box = [x, y, x + width, y + height]
    if len(row["bbox"]) != 4 or any(abs(a - b) >= 1e-5 for a, b in zip(row["bbox"], box)):
        raise ValueError("Mirrored xyxy bbox disagrees with original COCO xywh")
    if width <= 0 or height <= 0 or info["width"] <= 0 or info["height"] <= 0:
        raise ValueError("Invalid original bbox or image dimensions")
    return [1000 * box[0] / info["width"], 1000 * box[1] / info["height"],
            1000 * box[2] / info["width"], 1000 * box[3] / info["height"]], info


def image_index():
    index = {}
    paths = [HERE / "normalized" / "vqav2.jsonl", HERE / "normalized" / "refcoco.jsonl",
             OUT / "normalized" / "vqav2.jsonl", OUT / "normalized" / "refcoco.jsonl"]
    for path in paths:
        for row in read_records(path):
            if row.get("image_id") and len(row.get("images", [])) == 1:
                index.setdefault(int(row["image_id"]), (row["images"][0], row["image_sha256"][0]))
    return index


def get_image(image_id, info, reusable):
    expected_size = (int(info["width"]), int(info["height"]))
    if image_id in reusable:
        path, expected_hash = reusable[image_id]
        if Path(path).is_file():
            with Image.open(path) as image:
                dimensions = image.size
            if dimensions == expected_size:
                actual = common.v2.pixel_hash(path)
                if actual != expected_hash:
                    raise ValueError(f"Existing reusable COCO image changed: {path}")
                return {"path": str(Path(path).resolve()), "pixel_sha256": actual,
                        "source": "existing_same_COCO_image_ID_and_dimensions", "file_sha256": common.sha256(path)}
    name = f"COCO_train2014_{image_id:012d}.jpg"
    target = OUT / "images" / "refcoco" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        last_error = None
        for attempt in range(6):
            url = f"{'https' if attempt < 3 else 'http'}://images.cocodataset.org/train2014/{name}"
            try:
                response = requests.get(url, timeout=(20, 90))
                response.raise_for_status()
                blob = response.content
                with Image.open(io.BytesIO(blob)) as image:
                    if image.size != expected_size:
                        raise ValueError("Downloaded COCO image dimensions disagree with original annotation")
                    image.load()
                temporary = target.with_suffix(".writing")
                temporary.write_bytes(blob)
                temporary.replace(target)
                break
            except Exception as exc:
                last_error = exc
                if attempt < 5:
                    time.sleep(min(attempt + 1, 5))
        if not target.exists():
            raise RuntimeError(f"Could not obtain original COCO train image {image_id}: {last_error}")
    with Image.open(target) as image:
        if image.size != expected_size:
            raise ValueError(f"Cached COCO image dimensions changed: {target}")
    return {"path": str(target.resolve()), "pixel_sha256": common.v2.pixel_hash(str(target)),
            "source": f"https://images.cocodataset.org/train2014/{name}", "file_sha256": common.sha256(target)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-images", type=int, default=1000, help="Additional distinct train images per completed invocation")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261019)
    parser.add_argument("--plan-only", action="store_true", help="Fetch metadata only; do not download parquet/images")
    args = parser.parse_args()
    if args.new_images < 1 or args.workers < 1:
        parser.error("new-images and workers must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "refcoco_expansion.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    files = common.source_listing("refcoco")
    if len(files) != 1:
        raise ValueError("Expected the single pinned RefCOCO train annotation parquet")
    common.log(event="refcoco_plan", annotation_bytes=files[0]["size"],
               pinned_card_train_referring_objects=42404, additional_images=args.new_images)
    if args.plan_only:
        return
    if shutil.disk_usage(OUT).free < (60 << 30):
        raise RuntimeError("Keep at least 60 GiB available for formal checkpoints/export before expanding images")
    source = common.download_shard("refcoco", files[0], min(args.workers, 4))
    rows = pq.read_table(source).to_pylist()
    grouped, image_info = defaultdict(list), {}
    for row in rows:
        box, info = checked_bbox(row)
        image_id = int(row["image_id"])
        if image_id in image_info and image_info[image_id] != info:
            raise ValueError("Conflicting original image metadata")
        image_info[image_id] = info
        grouped[image_id].append((row, box))
    original = read_records(HERE / "normalized" / "refcoco.jsonl")
    target = OUT / "normalized" / "refcoco.jsonl"
    previous = read_records(target)
    used = {int(r["image_id"]) for r in original + previous}
    forbidden = set(json.loads((HERE / "raw" / "refcoco_selection.json").read_text())["all_original_val_image_ids"])
    available = sorted(set(grouped) - used - forbidden)
    random.Random(args.seed).shuffle(available)
    chosen = available[:args.new_images]
    if not chosen:
        raise RuntimeError("No unused original RefCOCO train images remain")
    reusable = image_index()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        values = list(pool.map(lambda i: get_image(i, image_info[i], reusable), chosen))
    image_results = dict(zip(chosen, values))
    added = []
    for image_id in chosen:
        image = image_results[image_id]
        for row, bbox in grouped[image_id]:
            for sentence in row["sentences"]:
                question = (f'Locate the object described as "{sentence["sent"]}". Return its bounding box '
                            'as [x1, y1, x2, y2] in coordinates normalized to 0-1000.')
                added.append({"id": f'refcoco:{row["ref_id"]}:{sentence["sent_id"]}', "source": "refcoco",
                              "split": "train", "source_id": str(sentence["sent_id"]), "image_id": str(image_id),
                              "images": [image["path"]], "image_sha256": [image["pixel_sha256"]],
                              "question": question, "question_sha256": common.v2.question_hash(question),
                              "task": "bbox", "answers": [], "bbox": bbox, "bbox_xyxy": bbox,
                              "bbox_xyxy_1000": bbox, "reward_type": "bbox",
                              "source_provenance": {"repo": SPEC["repo"], "revision": SPEC["revision"],
                                                    "file": files[0]["path"], "file_sha256": files[0]["lfs"]["oid"],
                                                    "image": image}})
    all_records = previous + added
    if len({r["id"] for r in all_records}) != len(all_records):
        raise ValueError("Repeated original RefCOCO sentence IDs")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".writing")
    with temporary.open("w") as stream:
        for row in all_records:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(target)
    receipt = {"completed_at": common.now(), "source": SPEC, "path": str(target.resolve()),
               "sha256": common.sha256(target), "rows": len(all_records), "added_rows": len(added),
               "added_images": len(chosen), "total_train_referring_objects": len(rows),
               "total_train_images": len(grouped), "total_train_expressions": sum(len(r["sentences"]) for r in rows),
               "unused_train_images_remaining": len(available) - len(chosen),
               "selected_image_ids": chosen, "source_sha256": files[0]["lfs"]["oid"],
               "normalizer_sha256": common.sha256(Path(__file__)),
               "difficulty_filter_applied": False, "evaluation_exclusion_applied": "original val image IDs only; run v3 pool full exclusion"}
    common.atomic_json(target.with_suffix(".manifest.json"), receipt)
    common.log(event="refcoco_expansion_complete", rows=len(all_records), added_rows=len(added), added_images=len(chosen),
               unused_train_images_remaining=receipt["unused_train_images_remaining"])


if __name__ == "__main__":
    main()
