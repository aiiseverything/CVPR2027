#!/usr/bin/env python3
"""Register one new export against unchanged frozen high-resolution inputs, on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sqlite3
import sys

from protocol import (DATA, PROTOCOL, REGISTERED_MODEL_NAMES, atomic_json,
                      content_for, digest, file_sha, load_suite, prompt_for, prompt_tokens)
from run import (BENCHMARKS, apply_subset, local_lock_path, model_provenance,
                 offline_environment, selection_snapshot)

SOURCE_FILES = {
    "suite.jsonl": "suite_sha256", "selection.json": "selection_sha256",
    "source_manifest.json": "source_manifest_sha256",
    "dataset_manifest.json": "dataset_manifest_sha256",
    "questions.jsonl": "questions_sha256",
}
LINK_FILES = (*SOURCE_FILES, "images.sqlite")


def checked_sources(source, meta):
    for filename, key in SOURCE_FILES.items():
        if file_sha(source / filename) != meta[key]:
            raise ValueError(f"Frozen source artifact changed: {filename}")
    if not (source / "images.sqlite").is_file():
        raise ValueError("Frozen image database is missing")


def selected_rows(rows, source, benchmarks=None, ids=None):
    selection = selection_snapshot(source)
    if benchmarks is not None:
        if not benchmarks or not set(benchmarks) <= BENCHMARKS:
            raise ValueError("Unknown or empty benchmark subset")
        rows = [row for row in rows if row["benchmark"] in benchmarks]
    if ids is not None:
        rows, _, _ = apply_subset(rows, selection, ids)
    if not rows:
        raise ValueError("Registration requires a non-empty selected subset")
    return rows


def rendered_prompt(processor, row):
    messages = [{"role": "user", "content": content_for(row["prompt"], {"type": "image"})}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=PROTOCOL["enable_thinking"])


def validate_processor(model, source, meta, rows):
    """Compare all selected prompts and token counts; preprocess two real images."""
    from transformers import AutoConfig, AutoProcessor
    from PIL import Image

    reference = Path(meta["models"]["baseline"]["path"])
    for filename, expected in meta["models"]["baseline"]["metadata_sha256"].items():
        if file_sha(reference / filename) != expected:
            raise ValueError(f"Frozen baseline metadata changed: {filename}")
    config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    context = getattr(getattr(config, "text_config", config), "max_position_embeddings")
    if context < PROTOCOL["max_model_len"]:
        raise ValueError("Export cannot support the frozen high-resolution context length")
    processor = AutoProcessor.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    baseline = AutoProcessor.from_pretrained(reference, local_files_only=True, trust_remote_code=False)
    if getattr(processor, "image_processor", None) is None:
        raise ValueError("Export does not load an image processor")
    for row in rows:
        if row["prompt"] != prompt_for(row):
            raise ValueError(f"Frozen prompt construction mismatch: {row['id']}")
        text, reference_text = rendered_prompt(processor, row), rendered_prompt(baseline, row)
        if (text != reference_text or processor.tokenizer.encode(text, add_special_tokens=False)
                != baseline.tokenizer.encode(reference_text, add_special_tokens=False)):
            raise ValueError(f"Export chat template/tokenization differs from frozen baseline: {row['id']}")
        actual, visual = prompt_tokens(processor, row)
        if (actual, visual) != (row["prompt_tokens"], row["visual_tokens"]):
            raise ValueError(f"Export prompt/visual token counts differ from frozen row: {row['id']}")
        if not 0 < actual <= PROTOCOL["max_prompt_tokens"]:
            raise ValueError(f"Frozen row exceeds prompt token budget: {row['id']}")
    checked = []
    representatives = {row["id"]: row for row in (
        min(rows, key=lambda row: row["visual_tokens"]),
        max(rows, key=lambda row: row["visual_tokens"]))}
    Image.MAX_IMAGE_PIXELS = None
    with sqlite3.connect((source / "images.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        for row in representatives.values():
            result = db.execute("SELECT data FROM images WHERE sha256=?", (row["image"]["sha256"],)).fetchone()
            if result is None or hashlib.sha256(result[0]).hexdigest() != row["image"]["sha256"]:
                raise ValueError(f"Missing or corrupt packed image: {row['id']}")
            with Image.open(io.BytesIO(result[0])) as original:
                if original.size != (row["image"]["width"], row["image"]["height"]):
                    raise ValueError(f"Packed image dimensions differ: {row['id']}")
                image = original.convert("RGB")
            actual = processor(text=[rendered_prompt(processor, row)], images=[image],
                               size=PROTOCOL["image_size"], return_tensors="pt")
            count = int(actual["input_ids"].shape[-1])
            if count != row["prompt_tokens"]:
                raise ValueError(f"Actual CPU image preprocessing differs from frozen count: {row['id']}")
            checked.append({"id": row["id"], "actual_prompt_tokens": count,
                            "image_grid_thw": actual["image_grid_thw"].tolist()})
            del actual, image, result
            gc.collect()
    return {"context_capacity": context, "processor_class": type(processor).__name__,
            "processor_verification": "checked",
            "processor_checks": {"n_rows": len(rows), "chat_template_and_token_ids": "matched_frozen_baseline",
                                 "prompt_and_visual_tokens": "matched_all_selected_frozen_rows",
                                 "actual_cpu_preprocessing": checked}}


def register(source_root, root, model, name, benchmarks=None, ids=None):
    source, root, model = Path(source_root).resolve(), Path(root).resolve(), Path(model).resolve()
    if name not in REGISTERED_MODEL_NAMES:
        raise ValueError("Unsupported new model name")
    if root == source:
        raise ValueError("Registration requires a private root; the original frozen metadata is immutable")
    os.environ.update(offline_environment())
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    rows, original_meta = load_suite(source)
    source_meta_sha256 = file_sha(source / "suite.meta.json")
    checked_sources(source, original_meta)
    rows = selected_rows(rows, source, benchmarks, ids)
    marker = model / "sharedexpert_export_complete.json"
    if not marker.is_file() or json.loads(marker.read_text()).get("step") != 100:
        raise ValueError("Registration requires the completed final step100 export")
    provenance = model_provenance(model)
    model_info = {"path": str(model),
                  "metadata_sha256": {key: value for key, value in provenance["metadata_sha256"].items()
                                      if Path(key).suffix in {".json", ".jinja"}},
                  "weight_files": provenance["weight_files"]}
    identity = {"schema_version": "highres-model-registration-v1", "name": name,
                "source_root": str(source), "source_meta_sha256": source_meta_sha256,
                "validated_ids_sha256": digest(sorted(row["id"] for row in rows)),
                "n_validated": len(rows), "counts": dict(Counter(row["benchmark"] for row in rows)),
                "export_identity_sha256": digest(model_info),
                "environment": {"python": sys.executable,
                                "transformers": importlib.metadata.version("transformers"),
                                "tokenizers": importlib.metadata.version("tokenizers")},
                "gpu_used": False}
    root.mkdir(parents=True, exist_ok=True)
    lock = local_lock_path(root, "register-highres-model").open("a")
    with lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for filename in LINK_FILES:
            target = root / filename
            if target.is_symlink() and target.resolve() == (source / filename).resolve():
                continue
            if target.exists() or target.is_symlink():
                raise ValueError(f"Private view contains an unrelated artifact: {target}")
            target.symlink_to(source / filename)
        meta_path = root / "suite.meta.json"
        if meta_path.is_symlink():
            raise ValueError("Private suite metadata must not be a symlink")
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            receipt = existing.get("registered_models", {}).get(name, {})
            if (all(receipt.get(key) == value for key, value in identity.items())
                    and receipt.get("model_sha256") == digest(existing.get("models", {}).get(name))):
                base = {**existing, "models": {key: value for key, value in existing["models"].items() if key != name}}
                base.pop("registered_models", None)
                if base == original_meta:
                    return receipt
            raise ValueError("Private registration already exists with different source/model/subset; refusing replacement")
        model_info.update(validate_processor(model, source, original_meta, rows))
        checked_sources(source, original_meta)
        if file_sha(source / "suite.meta.json") != source_meta_sha256 or model_provenance(model) != provenance:
            raise ValueError("Source metadata or exported model changed during registration")
        receipt = {**identity, "model_sha256": digest(model_info),
                   "created_at": datetime.now(timezone.utc).isoformat()}
        meta = {**original_meta, "models": {**original_meta["models"], name: model_info},
                "registered_models": {name: receipt}}
        atomic_json(meta_path, meta)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DATA)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--name", choices=REGISTERED_MODEL_NAMES, required=True)
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(BENCHMARKS))
    parser.add_argument("--ids", type=Path)
    args = parser.parse_args()
    print(json.dumps(register(args.source_root, args.root, args.model, args.name,
                              args.benchmarks, args.ids), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
