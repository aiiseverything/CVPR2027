#!/usr/bin/env python3
"""Read only image headers and stream teacher parquet to estimate vision budgets.

Uses the installed resize functions without importing torch/Transformers. No
image pixels are decoded. Estimates are not measured tensors or training RSS.
The default output is separate from the preserved historical audit.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import json
import hashlib
import math
import os
from pathlib import Path
import resource
import time
import typing

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_name] = "1"

import pyarrow.parquet as pq
from PIL import Image


SOURCE_WEIGHTS_PER_200 = {
    "perception": {"tallyqa": 80, "clevr": 80, "refcoco": 40},
    "reasoning": {"geometry3k": 80, "mmk12": 40, "virl39k": 80},
    "generalist": {"vqav2": 80, "docvqa": 10, "chartqa": 10,
                   "geometry3k": 16, "mmk12": 8, "virl39k": 16,
                   "tallyqa": 16, "clevr": 16, "refcoco": 8, "if_format": 20},
}


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resize_function(path, names, environment):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in nodes} != names:
        raise RuntimeError(f"Missing resize helpers in {path}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), environment)
    return environment["smart_resize"]


def main():
    project = Path(__file__).resolve().parents[1]
    workspace = project.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, default=project / "data_pipeline/prepared")
    parser.add_argument("--model", type=Path, default=workspace / "pre-lab/models/Qwen3.5-4B")
    parser.add_argument("--site-packages", type=Path, default=project / ".venv-train/lib/python3.10/site-packages")
    parser.add_argument("--output", type=Path, default=project / "logs/visual_batch_budget_audit_recomputed.json")
    parser.add_argument("--batch-size", type=int, default=40, help="Audit batch size; default preserves the historical batch40 audit.")
    parser.add_argument("--max-prompts", type=int, default=12000, help="Audit only this leading exposure prefix of each unmodified parquet.")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_prompts <= 0 or args.max_prompts % args.batch_size:
        parser.error("batch-size and max-prompts must be positive, and max-prompts must be divisible by batch-size")
    common = json.loads((project / "training/config/common.json").read_text())
    processor = json.loads((args.model / "preprocessor_config.json").read_text())
    patch = processor["patch_size"]
    merge = processor["merge_size"]
    temporal = processor["temporal_patch_size"]
    factor = patch * merge
    minimum = processor["size"]["shortest_edge"]
    maximum = processor["size"]["longest_edge"]
    # RGB channels * temporal patch duplication * bytes per FP32 value.
    bytes_per_pixel = 3 * temporal * 4
    first_path = args.site_packages / "qwen_vl_utils/vision_process.py"
    second_path = args.site_packages / "transformers/models/qwen2_vl/image_processing_qwen2_vl.py"
    first_resize = resize_function(first_path, {"smart_resize", "round_by_factor", "ceil_by_factor", "floor_by_factor"},
                                   {"math": math, "Optional": typing.Optional, "Tuple": typing.Tuple, "MAX_RATIO": 200})
    second_resize = resize_function(second_path, {"smart_resize"}, {"math": math})
    started = time.monotonic()
    dimensions = {}
    results = {}
    perception_first_pixels = None
    for role in ("perception", "reasoning", "generalist"):
        batches = []
        image_histogram = Counter()
        cap_mismatches = 0
        raw_image_max = {"pixels": 0}
        total_rows = 0
        source_totals = Counter()
        parquet_path = args.prepared / f"{role}.parquet"
        parquet_sha_before = file_sha256(parquet_path)
        table = pq.ParquetFile(parquet_path)
        if table.metadata.num_rows < args.max_prompts:
            raise ValueError(f"{parquet_path} has fewer than {args.max_prompts} rows")
        for batch_index, batch in enumerate(table.iter_batches(batch_size=args.batch_size, columns=["images", "extra_info"], use_threads=False)):
            if total_rows >= args.max_prompts:
                break
            batch = batch.slice(0, min(batch.num_rows, args.max_prompts - total_rows))
            pixels = raw_pixels = image_count = upscale_count = 0
            sources = Counter()
            prompt_details = []
            for row in batch.to_pylist():
                images = row["images"]
                n = len(images)
                image_histogram[n] += 1
                image_count += n
                sources[row["extra_info"]["source"]] += 1
                prompt_pixels = 0
                for item in images:
                    path = item["path"]
                    if path not in dimensions:
                        with Image.open(path) as header:
                            dimensions[path] = header.size
                    width, height = dimensions[path]
                    raw_pixels += width * height
                    if width * height > raw_image_max["pixels"]:
                        raw_image_max = {"pixels": width * height, "width": width, "height": height,
                                         "record_id": row["extra_info"]["record_id"]}
                    expected_cap = min(common["max_image_pixels"], common["max_total_image_pixels"] // n)
                    cap_mismatches += int(item["max_pixels"] != expected_cap or item["min_pixels"] != common["min_image_pixels"])
                    h1, w1 = first_resize(height, width, factor, item["min_pixels"], item["max_pixels"])
                    h2, w2 = second_resize(h1, w1, factor, minimum, maximum)
                    prompt_pixels += h2 * w2
                    upscale_count += int(h2 * w2 > h1 * w1)
                pixels += prompt_pixels
                prompt_details.append([prompt_pixels, row["extra_info"]["record_id"], n])
            if perception_first_pixels is None:
                perception_first_pixels = pixels
            batches.append({"step_1based": batch_index + 1, "rows": batch.num_rows,
                            "processed_pixels": pixels, "vision_tokens": pixels // factor**2,
                            "ratio_to_perception_first": pixels / perception_first_pixels,
                            "fp32_pixel_values_group8_gib": pixels * bytes_per_pixel * common["group_size"] / 2**30,
                            "raw_pixels": raw_pixels, "images": image_count, "sources": dict(sources),
                            "top3_prompts": sorted(prompt_details, reverse=True)[:3],
                            "hf_upscale_count": upscale_count})
            total_rows += batch.num_rows
            source_totals.update(sources)
        parquet_sha_after = file_sha256(parquet_path)
        if parquet_sha_before != parquet_sha_after:
            raise RuntimeError(f"Prepared data changed during audit: {parquet_path}")
        ordered = sorted(batches, key=lambda value: value["processed_pixels"])
        results[role] = {"rows": total_rows, "batch_count": len(batches), "cap_mismatches": cap_mismatches,
                         "prepared_rows": table.metadata.num_rows, "prepared_sha256_before": parquet_sha_before,
                         "prepared_sha256_after": parquet_sha_after, "prepared_unchanged_during_audit": True,
                         "source_counts": dict(source_totals),
                         "source_proportions": {key: count / total_rows for key, count in source_totals.items()},
                         "expected_source_weights_per_200": SOURCE_WEIGHTS_PER_200[role],
                         "source_ratios_match_exactly": set(source_totals) == set(SOURCE_WEIGHTS_PER_200[role]) and
                             all(source_totals[key] * 200 == weight * total_rows for key, weight in SOURCE_WEIGHTS_PER_200[role].items()),
                         "images_per_prompt_histogram": dict(image_histogram), "raw_max_image": raw_image_max,
                         "hf_upscaled_images_including_repeated_rows": sum(b["hf_upscale_count"] for b in batches),
                         "first": batches[0], "minimum": ordered[0], "median_upper_order_statistic": ordered[len(ordered) // 2],
                         "p95_nearest_rank": ordered[math.ceil(.95 * len(ordered)) - 1], "maximum": ordered[-1],
                         "maximum_raw_pixel_batch": max(batches, key=lambda value: value["raw_pixels"]),
                         "batches": batches}
        print(json.dumps({"teacher": role, "rows": total_rows, "batch_count": len(batches),
                          "source_counts": dict(source_totals),
                          "source_ratios_match_exactly": results[role]["source_ratios_match_exactly"],
                          "first": batches[0], "maximum": ordered[-1]}, ensure_ascii=False), flush=True)
    generalist_counts = results["generalist"]["source_counts"]
    results["generalist"]["category_counts"] = {
        "general_vqa_caption": generalist_counts["vqav2"],
        "document_chart": generalist_counts["docvqa"] + generalist_counts["chartqa"],
        "reasoning": sum(generalist_counts[key] for key in ("geometry3k", "mmk12", "virl39k")),
        "perception": sum(generalist_counts[key] for key in ("tallyqa", "clevr", "refcoco")),
        "if_format": generalist_counts["if_format"],
    }
    result = {"status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "method": "PIL header.size only; two exact installed pure smart_resize functions; no decode or tensors",
              "source_functions": [str(first_path), str(second_path)], "gpu_used": False,
              "batch_size": args.batch_size, "max_prompts_per_teacher": args.max_prompts,
              "steps_per_teacher": args.max_prompts // args.batch_size, "group_size": common["group_size"],
              "rollout_candidates_per_step": args.batch_size * common["group_size"],
              "world_size_checked": 4,
              "rollout_candidates_divisible_by_4": args.batch_size * common["group_size"] % 4 == 0,
              "rollout_candidates_per_rank_if_4_equal_shards": args.batch_size * common["group_size"] / 4,
              "common_config_batch_size_at_audit": common["batch_size"],
              "data_exposure_note": "Only the requested leading prompt exposures are audited; original prepared parquet remains intact. Exposure count does not imply distinct source examples.",
              "image_grid_thw_formula": f"[1, final_height/{patch}, final_width/{patch}]",
              "vision_tokens_formula": f"sum(final_height*final_width)/{factor**2}",
              "fp32_pixel_values_bytes_formula": f"sum(final_height*final_width)*{bytes_per_pixel}",
              "hf_processor_min_pixels": minimum, "hf_processor_max_pixels": maximum,
              "limitations": ["Derived image grids, not materialized or measured tensors.",
                              "FP32 byte estimate is one complete copy only; runtime copy count and RSS are unknown.",
                              "Batch order follows parquet and current data.shuffle=False; other sampling changes require re-audit."],
              "unique_image_headers": len(dimensions), "elapsed_seconds": time.monotonic() - started,
              "max_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "teachers": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
