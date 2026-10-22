#!/usr/bin/env python3
"""Stage a merged training export as a complete HF model directory.

The archive is treated as untrusted data: it is only read/extracted, never
imported or executed. A truncated archive is rejected before any weight file
is published. Files absent from the archive (for example an older export that
omitted tokenizer metadata) are copied from the base model, but files present
in the archive always win.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tarfile
from pathlib import Path

FALLBACK_FILES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "video_preprocessor_config.json",
    "chat_template.jinja",
]


def find_member(tf, name):
    matches = [m for m in tf.getmembers() if m.name.rsplit("/", 1)[-1] == name and m.isfile()]
    if len(matches) > 1:
        raise SystemExit(f"Expected at most one {name} in archive, found {len(matches)}")
    return matches[0] if matches else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tar", type=Path, required=True)
    ap.add_argument("--base-model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--weights-name", default="model.safetensors")
    args = ap.parse_args()

    archive_size = args.tar.stat().st_size
    with tarfile.open(args.tar, "r:*") as tf:
        weights = find_member(tf, args.weights_name)
        if weights is None:
            raise SystemExit(f"Archive does not contain {args.weights_name}")
        end = weights.offset_data + weights.size
        if end > archive_size:
            raise SystemExit(
                f"Truncated archive: {args.weights_name} needs {weights.size:,} bytes ending at "
                f"offset {end:,}, but {args.tar} is {archive_size:,} bytes "
                f"(missing {end - archive_size:,} bytes). Re-copy the complete tar first.")

        args.output.mkdir(parents=True, exist_ok=True)
        extracted = []
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = member.name.rsplit("/", 1)[-1]
            target = args.output / name
            part = target.with_name(target.name + ".part")
            source = tf.extractfile(member)
            if source is None:
                continue
            with part.open("wb") as out:
                shutil.copyfileobj(source, out, length=16 << 20)
            if part.stat().st_size != member.size:
                raise SystemExit(f"Extraction incomplete for {name}: {part.stat().st_size:,}/{member.size:,}")
            part.replace(target)
            extracted.append(name)

    for name in FALLBACK_FILES:
        target = args.output / name
        if target.exists():
            continue
        if (args.base_model / name).is_file():
            shutil.copy2(args.base_model / name, target)

    required = ["config.json", "tokenizer_config.json", "model.safetensors"]
    missing = [name for name in required if not (args.output / name).is_file()]
    if not any((args.output / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        missing.append("processor_config.json or preprocessor_config.json")
    if missing:
        raise SystemExit(f"Staged checkpoint is missing {missing}")
    summary = {
        "output": str(args.output.resolve()),
        "weights_bytes": (args.output / args.weights_name).stat().st_size,
        "extracted_from_tar": sorted(extracted),
        "base_model": str(args.base_model.resolve()),
        "files": sorted(p.name for p in args.output.iterdir()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
