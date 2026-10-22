#!/usr/bin/env python3
"""Freeze BLINK val and MMVP into the shared evaluation row format.

The two source TSVs are VLMEvalKit release files (MD5 recorded in
suite_blink_mmvp.meta.json). Images are written to a dedicated asset
directory with the same pixel/file hash format as the main suite. Prompt
construction, image budget, prompt-token counting and exclusion rules are
reused from prepare_suite.py; no protocol value is changed.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import PROTOCOL, SCORING_PROTOCOL, digest, file_sha, question_sha  # noqa: E402
from prepare_suite import prompt_tokens, save_images  # noqa: E402

csv.field_size_limit(100_000_000)


def image_values(row):
    value = row.get("image", "")
    return ast.literal_eval(value) if value.startswith("[") else [value]


def make_row(benchmark, row, assets, processor, dimension, split, category):
    choices = {c: row[c] for c in "ABCDEFGHIJ" if row.get(c)}
    if not choices:
        raise ValueError(f"{benchmark}:{row['index']} has no options")
    answer = str(row["answer"]).strip().upper()
    if answer not in choices:
        raise ValueError(f"{benchmark}:{row['index']} answer {answer!r} not in {sorted(choices)}")
    question = row["question"]
    prompt = question + "\n" + "\n".join(f"{c}. {value}" for c, value in choices.items())
    prompt += "\nPut only your final option letter inside <answer>...</answer>."
    paths, hashes, file_hashes, sizes = save_images(image_values(row), assets)
    return {
        "id": f"{benchmark}:{row['index']}", "benchmark": benchmark, "dimension": dimension,
        "source_index": row["index"], "group": row["index"], "category": category,
        "split": split, "question": question, "question_sha256": question_sha(question),
        "prompt": prompt, "images": paths, "image_sha256": hashes,
        "image_file_sha256": file_hashes, "image_sizes": sizes,
        "prompt_tokens": prompt_tokens(processor, prompt, sizes),
        "task": "mcq", "answer": answer, "choices": choices,
    }


def read_tsv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--blink-tsv", type=Path, required=True)
    ap.add_argument("--mmvp-tsv", type=Path, required=True)
    ap.add_argument("--processor", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=HERE / "suite_blink_mmvp.jsonl")
    ap.add_argument("--assets", type=Path, default=HERE / "assets_blink_mmvp")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"Refusing to replace frozen suite: {args.output}")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.processor, local_files_only=True)
    args.assets.mkdir(parents=True, exist_ok=True)

    output, exclusions = [], []
    sources = [
        ("BLINK_val", args.blink_tsv, "val", lambda row: row["category"]),
        ("MMVP", args.mmvp_tsv, "official-evaluation", lambda row: "all"),
    ]
    for benchmark, path, split, category in sources:
        rows = read_tsv(path)
        if len({row["index"] for row in rows}) != len(rows):
            raise SystemExit(f"Duplicate indices in {path}")
        for row in rows:
            prepared = make_row(benchmark, row, args.assets, processor, "perception", split, category(row))
            if prepared["prompt_tokens"] > PROTOCOL["max_prompt_tokens"] or len(prepared["images"]) > PROTOCOL["max_images"]:
                exclusions.append({"id": prepared["id"], "reason": "prompt_budget",
                                   "prompt_tokens": prepared["prompt_tokens"],
                                   "n_images": len(prepared["images"])})
                continue
            output.append(prepared)
        print(f"{benchmark}: {len(rows)} source rows, {sum(r['benchmark'] == benchmark for r in output)} retained", flush=True)

    output.sort(key=lambda row: (row["benchmark"], row["source_index"]))
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in output))
    meta = {
        "protocol": PROTOCOL, "protocol_sha256": digest(PROTOCOL),
        "suite_sha256": file_sha(args.output),
        "scoring_protocol": SCORING_PROTOCOL, "scoring_protocol_sha256": digest(SCORING_PROTOCOL),
        "processor": str(args.processor.resolve()),
        "processor_config_sha256": file_sha(args.processor / "preprocessor_config.json"),
        "source_sha256": {"BLINK_val": file_sha(args.blink_tsv), "MMVP": file_sha(args.mmvp_tsv)},
        "source_md5_expected": {"BLINK_val": "d5e8af148b10ac69f535ff7b23f3f989",
                                "MMVP": "8cb732b141a0cba5b42159df2839e557"},
        "counts": dict(Counter(row["benchmark"] for row in output)),
        "excluded": exclusions, "grounding_included": False,
        "image_hash_format": "sha256(struct.pack('>II', width, height) + RGB pixel bytes)",
        "limitations": ["Strict local extraction scores, not official LLM-judge scores",
                        "BLINK val uses VLMEvalKit TSV rows; MMVP uses all 300 rows in its TSV",
                        "Hash exclusion catches exact pixels only, not all re-encoded derivatives"],
    }
    args.output.with_suffix(".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"suite": str(args.output), "counts": meta["counts"], "excluded": len(exclusions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
