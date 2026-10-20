#!/usr/bin/env python3
"""CPU-only checks. Passing these is not a GPU training smoke test."""

import argparse
import importlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
REPO = WORKSPACE / "pre-lab/repo"
sys.path[:0] = [str(WORKSPACE), str(REPO)]

EXPECTED = {
    "torch": "2.10.0", "transformers": "5.5.0", "vllm": "0.18.0",
    "ray": "2.53.0", "hydra-core": "1.3.2", "omegaconf": "2.3.0",
    "torchdata": "0.11.0", "tensordict": "0.10.0",
    "flash-linear-attention": "0.4.2", "flash-attn": "2.8.3",
}


def check_imports():
    errors, versions = [], {}
    for package, expected in EXPECTED.items():
        try:
            actual = importlib.metadata.version(package)
            versions[package] = actual
            if actual.split("+")[0] != expected:
                errors.append(f"{package}: expected {expected}, found {actual}")
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing package: {package}")
    if errors:
        return {"versions": versions, "errors": errors}
    modules = [
        "flash_attn", "fla", "transformers.models.qwen3_5.modeling_qwen3_5",
        "verl.models.transformers.qwen3_5", "verl.trainer.main_ppo",
        "verl.workers.fsdp_workers", "verl.workers.rollout.vllm_rollout.vllm_async_server",
    ]
    for module in modules:
        try:
            importlib.import_module(module)
        except Exception as exc:
            errors.append(f"Import {module}: {type(exc).__name__}: {exc}")
    return {"versions": versions, "errors": errors}


def check_dataset(path, expected_rows):
    import pyarrow.parquet as pq

    path = Path(path)
    if not path.is_file():
        return {"errors": [f"Missing prepared dataset: {path}"]}
    table = pq.read_table(path)
    required = {"prompt", "images", "data_source", "reward_model", "extra_info"}
    missing = required - set(table.column_names)
    errors = [f"Missing columns: {sorted(missing)}"] if missing else []
    if expected_rows is not None and table.num_rows != expected_rows:
        errors.append(f"Expected {expected_rows} prompt exposures, found {table.num_rows}")
    if missing:
        return {"rows": table.num_rows, "errors": errors}
    tasks, sources, image_paths = {}, {}, set()
    for idx, row in enumerate(table.to_pylist()):
        try:
            if not row["prompt"] or not isinstance(row["prompt"], list):
                raise ValueError("prompt must be nonempty chat message list")
            reward = row["reward_model"]
            gt = json.loads(reward["ground_truth"])
            task = gt["task"]
            if task not in {"exact", "math", "vqa", "bbox", "json", "anls", "relaxed_numeric"}:
                raise ValueError(f"unsupported task: {task}")
            if task == "bbox":
                box = gt["bbox"]
                if len(box) != 4 or not all(0 <= float(x) <= 1000 for x in box):
                    raise ValueError("bbox must have four coordinates in [0,1000]")
            elif not gt.get("answers"):
                raise ValueError("answers must not be empty")
            tasks[task] = tasks.get(task, 0) + 1
            sources[row["data_source"]] = sources.get(row["data_source"], 0) + 1
            for image in row["images"] or []:
                if image.get("path"):
                    image_paths.add(image["path"])
                elif not image.get("bytes"):
                    raise ValueError("image must contain local path or bytes")
        except Exception as exc:
            errors.append(f"Row {idx}: {exc}")
            if len(errors) >= 25:
                break
    for image in image_paths:
        if not Path(image).is_file():
            errors.append(f"Missing local image: {image}")
            if len(errors) >= 25:
                break
    return {"rows": table.num_rows, "tasks": tasks, "sources": sources, "local_images": len(image_paths), "errors": errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true")
    parser.add_argument("--teacher", choices=["perception", "reasoning", "generalist"])
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    # Never allocate GPU memory in preflight, even when called independently.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    report = {} if args.data_only else check_imports()
    if args.teacher:
        report["dataset"] = check_dataset(
            WORKSPACE / f"experiments/sharedexpert_c/data_pipeline/prepared/{args.teacher}.parquet", 12000
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(report.get("errors") or report.get("dataset", {}).get("errors")))


if __name__ == "__main__":
    sys.exit(main())
