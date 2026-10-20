#!/usr/bin/env python3
"""Freeze CPU-only, deterministic benchmark subsets and image-level exclusion hashes."""
from __future__ import annotations

import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from common import HERE, ROOT, PROTOCOL, SCORING_PROTOCOL, digest, file_sha, pixel_sha, question_sha, read_jsonl, message_content, image_size_for_count

csv.field_size_limit(100_000_000)


def rank(key):
    return hashlib.sha256(f"{PROTOCOL['seed']}:{key}".encode()).hexdigest()


def stratified(rows, count):
    if count <= 0 or count >= len(rows):
        return rows
    groups = defaultdict(list)
    for row in rows:
        groups[row.get("category", "all")].append(row)
    for values in groups.values():
        values.sort(key=lambda r: rank(r["index"]))
    # Proportional allocation, largest remainder, deterministic ties.
    quotas = {k: count * len(v) / len(rows) for k, v in groups.items()}
    sizes = {k: int(v) for k, v in quotas.items()}
    for k in sorted(groups, key=lambda k: (-(quotas[k] - sizes[k]), k))[:count - sum(sizes.values())]:
        sizes[k] += 1
    return [r for k in sorted(groups) for r in groups[k][:sizes[k]]]


def image_values(row, index):
    value = row.get("image", "")
    visited = set()
    while value in index and len(value) < 80:
        if value in visited:
            raise ValueError("Cyclic image reference")
        visited.add(value)
        value = index[value]["image"]
    return ast.literal_eval(value) if value.startswith("[") else [value]


def save_images(values, asset_dir):
    paths, hashes, file_hashes, sizes = [], [], [], []
    for value in values:
        if len(value) < 240 and Path(value).is_file():
            blob = Path(value).read_bytes()
        else:
            blob = base64.b64decode(value, validate=True)
        sha = hashlib.sha256(blob).hexdigest()
        with Image.open(io.BytesIO(blob)) as im:
            extension = ".png" if im.format == "PNG" else ".jpg" if im.format == "JPEG" else ".img"
            path = asset_dir / (sha + extension)
            if not path.exists():
                path.write_bytes(blob)
            paths.append(str(path.resolve()))
            hashes.append(pixel_sha(im))
            file_hashes.append(sha)
            sizes.append(list(im.size))
    return paths, hashes, file_hashes, sizes


def prompt_tokens(processor, prompt, sizes):
    # Exact image-token count through the installed processor's own sizing routine.
    messages = [{"role": "user", "content": message_content(prompt, [{"type": "image"} for _ in sizes])}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    text_n = len(processor.tokenizer.encode(text, add_special_tokens=False))
    # Installed transformers' count helper accepts min/max_pixels but ignores size.
    # Actual preprocessing/API requests use the equivalent public size dictionary.
    size = image_size_for_count(len(sizes))
    info = processor._get_num_multimodal_tokens(image_sizes=[(h, w) for w, h in sizes], min_pixels=size["shortest_edge"], max_pixels=size["longest_edge"])
    return text_n + sum(info["num_image_tokens"]) - len(sizes)


def make_row(name, row, index, asset_dir, processor):
    choices = {c: row[c] for c in "ABCDEFGHIJ" if row.get(c)}
    if not choices and row.get("choices"):
        choices = {chr(65 + i): x for i, x in enumerate(ast.literal_eval(row["choices"]))}
    task = "yesno" if name == "MME_perception" else "mcq" if choices else "exact"
    answer = row.get("answer_option") or row["answer"]
    question = row["question"]
    prompt = (row.get("hint", "") + "\n" if row.get("hint") else "") + question
    if choices and "Choices:" not in question:
        prompt += "\n" + "\n".join(f"{c}. {value}" for c, value in choices.items())
    prompt += "\nPut only your final " + ("option letter" if task == "mcq" else "yes/no answer" if task == "yesno" else "answer") + " inside <answer>...</answer>."
    paths, hashes, file_hashes, sizes = save_images(image_values(row, index), asset_dir)
    token_count = prompt_tokens(processor, prompt, sizes)
    group = str(int(row["index"]) % 1_000_000) if name == "MMBench_DEV_EN_V11" else row["index"]
    if name == "MME_perception":
        group = hashes[0]
    return {
        "id": f"{name}:{row['index']}", "benchmark": name,
        "dimension": "perception" if name == "MME_perception" else "general" if name in ["MMStar", "MMBench_DEV_EN_V11"] else "reasoning",
        "source_index": row["index"], "group": group, "category": row.get("category", "all"),
        "split": row.get("split", "official-evaluation"), "question": question,
        "question_sha256": question_sha(question), "prompt": prompt,
        "images": paths, "image_sha256": hashes, "image_file_sha256": file_hashes,
        "image_sizes": sizes, "prompt_tokens": token_count,
        "task": task, "answer": answer, "choices": choices,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", type=Path, default=ROOT / "pre-lab/lmudata")
    ap.add_argument("--processor", type=Path, default=ROOT / "pre-lab/models/Qwen3.5-4B")
    ap.add_argument("--output", type=Path, default=HERE / "suite.jsonl")
    ap.add_argument("--refcoco-jsonl", type=Path, help="Heldout records: id, question, images, bbox_xyxy_1000, split")
    ap.add_argument("--refcoco-limit", type=int, default=500)
    ap.add_argument("--require-grounding", action="store_true")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to replace frozen suite: {args.output}; use a new --output path")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    assets = args.output.parent / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    output, exclusions, source_hashes = [], [], {}
    plans = [("MathVista_MINI", 500), ("MathVision_MINI", 0), ("MMMU_DEV_VAL", 450), ("MME", 0), ("MMStar", 500), ("MMBench_DEV_EN_V11", 250)]
    for name, count in plans:
        path = args.data_root / f"{name}.tsv"
        source_hashes[name] = file_sha(path)
        with path.open() as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        index = {r["index"]: r for r in rows}
        if name == "MMMU_DEV_VAL":
            rows = [r for r in rows if r["split"] == "validation" and r["question_type"] == "multiple-choice"]
        if name == "MME":
            rows = [r for r in rows if r["category"] in {"count", "position", "color", "existence"}]
        if name == "MMBench_DEV_EN_V11":
            group_ids = sorted({str(int(r["index"]) % 1_000_000) for r in rows}, key=rank)[:count]
            rows = [r for r in rows if str(int(r["index"]) % 1_000_000) in group_ids]
        else:
            rows = stratified(rows, count)
        name_out = "MME_perception" if name == "MME" else name
        current = [make_row(name_out, r, index, assets, processor) for r in rows]
        # A circular MMBench or MME pair must be retained/dropped together.
        bad = {r["group"] for r in current if r["prompt_tokens"] > PROTOCOL["max_prompt_tokens"] or len(r["images"]) > PROTOCOL["max_images"]}
        output.extend(r for r in current if r["group"] not in bad)
        exclusions.extend({"id": r["id"], "reason": "prompt_budget_or_paired_with_excluded_row", "prompt_tokens": r["prompt_tokens"]} for r in current if r["group"] in bad)
        print(f"{name}: selected {len(current)}, retained {sum(r['group'] not in bad for r in current)}", flush=True)
    if args.refcoco_jsonl:
        source_hashes["RefCOCO_heldout"] = file_sha(args.refcoco_jsonl)
        rows = sorted(read_jsonl(args.refcoco_jsonl), key=lambda r: rank(r["id"]))[:args.refcoco_limit]
        for row in rows:
            if "train" in row["split"].lower():
                raise ValueError("RefCOCO evaluation cannot use a training split")
            paths, hashes, file_hashes, sizes = save_images(row.get("images") or [row["image"]], assets)
            prompt = row["question"] + "\nReturn the bounding box [x1,y1,x2,y2], with coordinates normalized to 0–1000, inside <answer>...</answer>."
            n = prompt_tokens(processor, prompt, sizes)
            prepared = {
                **row, "id": "RefCOCO_heldout:" + str(row["id"]), "benchmark": "RefCOCO_heldout", "dimension": "perception",
                "group": hashes[0], "category": row["split"], "prompt": prompt, "prompt_tokens": n,
                "images": paths, "image_sha256": hashes, "image_file_sha256": file_hashes, "image_sizes": sizes,
                "question_sha256": question_sha(row["question"]), "task": "grounding", "answer": json.dumps(row["bbox_xyxy_1000"]),
            }
            if n <= PROTOCOL["max_prompt_tokens"]:
                output.append(prepared)
            else:
                exclusions.append({"id": prepared["id"], "reason": "prompt_budget", "prompt_tokens": n})
    elif args.require_grounding:
        raise SystemExit("Grounding required but --refcoco-jsonl missing")
    output.sort(key=lambda r: r["id"])
    args.output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in output))
    meta = {
        "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL), "suite_sha256": file_sha(args.output),
        "scoring_protocol": SCORING_PROTOCOL, "scoring_protocol_sha256": digest(SCORING_PROTOCOL),
        "processor": str(args.processor.resolve()), "processor_config_sha256": file_sha(args.processor / "preprocessor_config.json"),
        "source_sha256": source_hashes, "counts": dict(Counter(r["benchmark"] for r in output)), "excluded": exclusions,
        "grounding_included": bool(args.refcoco_jsonl),
        "image_hash_format": "sha256(struct.pack('>II', width, height) + RGB pixel bytes)",
        "limitations": ["Strict local extraction scores, not official LLM-judge scores", "Subsets selected before training; preserve circular MMBench groups", "Hash exclusion catches exact pixels only, not all re-encoded or cropped derivatives"],
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"suite": str(args.output), "counts": meta["counts"], "excluded": len(exclusions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
