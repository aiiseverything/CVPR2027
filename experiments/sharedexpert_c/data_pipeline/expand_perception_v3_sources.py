#!/usr/bin/env python3
"""Expand perception-v3 candidates from pinned original VQAv2/DocVQA train shards.

All new artifacts stay under perception_v3_sources/. Existing normalized data,
images, v2 inputs, and rewards remain untouched. Downloads are checked against
the pinned repository's upstream LFS SHA256 before any annotations are read.
Only original questions passing the unchanged v2 category rules are normalized;
this is a CPU candidate extraction, not a claim of passing model difficulty.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import fcntl
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import struct
import time

import pyarrow.parquet as pq
from PIL import Image
import requests

import prepare_perception_v2 as v2

HERE = Path(__file__).resolve().parent
OUT = HERE / "perception_v3_sources"
ENDPOINT = "https://hf-mirror.com"
SOURCES = {
    "vqav2": {"repo": "lmms-lab-encoder/VQAv2",
              "original_repo": "lmms-lab/VQAv2",
              "revision": "32665d35052eb4a6d4414851c3c829a72754915a",
              "prefix": "data/train-", "task": "vqa"},
    "docvqa": {"repo": "lmms-lab-encoder/DocVQA",
               "original_repo": "lmms-lab/DocVQA",
               "revision": "539088ef8a8ada01ac8e2e6d4e372586748a265e",
               "prefix": "DocVQA/train-", "task": "anls"},
}


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def log(**fields):
    print(json.dumps({"time": now(), **fields}, ensure_ascii=False), flush=True)


def source_listing(source):
    spec = SOURCES[source]
    metadata = OUT / "metadata" / f"{source}.{spec['revision']}.json"
    if metadata.is_file():
        stored = json.loads(metadata.read_text())
        if stored["revision"] != spec["revision"] or stored["repo"] != spec["repo"]:
            raise ValueError(f"Metadata identity mismatch: {metadata}")
        return stored["files"]
    url = f"{ENDPOINT}/api/datasets/{spec['repo']}/tree/{spec['revision']}?recursive=true"
    listing = []
    while url:
        response = requests.get(url, timeout=(20, 90))
        response.raise_for_status()
        listing.extend(response.json())
        url = response.links.get("next", {}).get("url")
    selected = sorted((x for x in listing if x.get("type") == "file"
                       and x["path"].startswith(spec["prefix"])
                       and x["path"].endswith(".parquet")), key=lambda x: x["path"])
    if not selected:
        raise ValueError(f"No pinned train shards: {source}")
    for item in selected:
        lfs = item.get("lfs") or {}
        if not re.fullmatch(r"[a-f0-9]{64}", lfs.get("oid", "")):
            raise ValueError(f"Missing upstream LFS SHA256: {item['path']}")
        if item["size"] != lfs.get("size"):
            raise ValueError(f"Upstream size disagreement: {item['path']}")
    atomic_json(metadata, {"repo": spec["repo"], "original_repo": spec["original_repo"],
                          "revision": spec["revision"], "fetched_at": now(), "files": selected})
    return selected


def download_shard(source, item, range_workers):
    spec = SOURCES[source]
    out = OUT / "raw" / source / item["path"]
    size, expected = item["size"], item["lfs"]["oid"]
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and out.stat().st_size == size and sha256(out) == expected:
        log(event="verified_cached_shard", source=source, file=item["path"])
        return out
    if shutil.disk_usage(OUT).free < 2 * size + (2 << 30):
        raise RuntimeError("Insufficient free disk for verified shard assembly")
    url = f"{ENDPOINT}/datasets/{spec['repo']}/resolve/{spec['revision']}/{item['path']}?download=true"
    chunks = out.with_name(out.name + ".ranges")
    chunks.mkdir(exist_ok=True)
    chunk_size = 8 << 20
    starts = list(range(0, size, chunk_size))

    def fetch(start):
        end = min(size, start + chunk_size) - 1
        part = chunks / f"{start:012d}.part"
        if part.is_file() and part.stat().st_size == end - start + 1:
            return part
        error = None
        for attempt in range(8):
            temp = part.with_name(part.name + ".downloading")
            try:
                with requests.get(url + f"&range={start}-{attempt}",
                                  headers={"Range": f"bytes={start}-{end}"},
                                  stream=True, timeout=(20, 90)) as response:
                    response.raise_for_status()
                    if response.status_code != 206:
                        raise ValueError(f"Expected HTTP206 range, got {response.status_code}")
                    if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                        raise ValueError("Wrong Content-Range")
                    with temp.open("wb") as stream:
                        for block in response.iter_content(1 << 20):
                            if block:
                                stream.write(block)
                if temp.stat().st_size != end - start + 1:
                    raise ValueError("Wrong range byte count")
                temp.replace(part)
                return part
            except Exception as exc:
                error = exc
                if temp.exists():
                    temp.unlink()
                if attempt != 7:
                    time.sleep(min(10, attempt + 1))
        raise RuntimeError(f"Range download failed: {source}/{item['path']}@{start}: {error}")

    with ThreadPoolExecutor(max_workers=range_workers) as pool:
        for future in as_completed([pool.submit(fetch, start) for start in starts]):
            future.result()
    assembly = out.with_name(out.name + ".assembling")
    with assembly.open("wb") as stream:
        for start in starts:
            with (chunks / f"{start:012d}.part").open("rb") as part:
                shutil.copyfileobj(part, stream, 8 << 20)
    actual = sha256(assembly)
    if assembly.stat().st_size != size or actual != expected:
        # Leave forensic bytes unpublished. No annotation is consumed on failure.
        raise ValueError(f"Upstream LFS verification failed: {item['path']}: {actual} != {expected}")
    assembly.replace(out)
    shutil.rmtree(chunks)
    log(event="download_verified", source=source, file=item["path"], bytes=size, sha256=actual)
    return out


def annotation_record(source, row):
    if source == "vqav2":
        sid, answers = str(row["question_id"]), [str(a["answer"]) for a in row["answers"]]
        if len(answers) != 10:
            raise ValueError("VQAv2 requires all ten original answer annotations")
        extras = {"image_id": str(row["image_id"])}
    elif source == "docvqa":
        if row["data_split"] != "train":
            raise ValueError("Non-train DocVQA record")
        sid, answers, extras = str(row["questionId"]), [str(a) for a in row["answers"]], {}
    else:
        raise ValueError(source)
    question = row["question"].replace("<image>", "").strip()
    if not answers:
        raise ValueError("Missing original answers")
    return {"id": f"{source}:{sid}", "source": source, "source_id": sid, "split": "train",
            "question": question, "question_sha256": v2.question_hash(question),
            "task": SOURCES[source]["task"], "answers": answers, **extras}


def normalize_shard(source, item, path, base_ids):
    target = OUT / "normalized_shards" / source / (path.stem + ".jsonl")
    manifest = target.with_suffix(".manifest.json")
    policy_hash = sha256(HERE / "prepare_perception_v2.py")
    identity = {"source": source, "repo": SOURCES[source]["repo"],
                "revision": SOURCES[source]["revision"], "source_file": item["path"],
                "source_sha256": item["lfs"]["oid"], "source_bytes": item["size"],
                "category_policy_sha256": policy_hash,
                "normalizer_sha256": sha256(Path(__file__))}
    if target.is_file() and manifest.is_file():
        saved = json.loads(manifest.read_text())
        if all(saved.get(k) == v for k, v in identity.items()) and saved.get("sha256") == sha256(target):
            return saved
        raise ValueError(f"Existing normalized shard identity changed: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".writing")
    stats, categories, image_cache, seen = Counter(), Counter(), {}, set()
    pf = pq.ParquetFile(path)
    with temp.open("w") as output:
        for batch in pf.iter_batches(batch_size=32):
            for row in batch.to_pylist():
                stats["source_rows"] += 1
                record = annotation_record(source, row)
                if record["id"] in base_ids:
                    stats["already_in_original_normalized"] += 1
                    continue
                if record["id"] in seen:
                    raise ValueError(f"Duplicate original question ID in shard: {record['id']}")
                seen.add(record["id"])
                classification = v2.classify(record)
                if classification is None:
                    stats["excluded_by_unchanged_category_rules"] += 1
                    continue
                blob = row["image"].get("bytes")
                if not blob:
                    raise ValueError("Expected embedded original image bytes in pinned parquet")
                image_digest = hashlib.sha256(blob).hexdigest()
                if image_digest not in image_cache:
                    image_path = OUT / "images" / source / (image_digest + ".img")
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    with Image.open(io.BytesIO(blob)) as original:
                        rgb = original.convert("RGB")
                        pixels = hashlib.sha256(struct.pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()
                    if image_path.exists():
                        if sha256(image_path) != image_digest:
                            raise ValueError(f"Existing image bytes changed: {image_path}")
                    else:
                        image_tmp = image_path.with_suffix(".writing")
                        image_tmp.write_bytes(blob)
                        image_tmp.replace(image_path)
                    image_cache[image_digest] = (str(image_path.resolve()), pixels)
                image_path, pixels = image_cache[image_digest]
                record.update(images=[image_path], image_sha256=[pixels],
                              source_provenance={"repo": identity["repo"], "revision": identity["revision"],
                                                 "file": item["path"], "file_sha256": identity["source_sha256"]})
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                categories[classification[0]] += 1
                stats["normalized_candidates"] += 1
    temp.replace(target)
    result = {**identity, "path": str(target.resolve()), "sha256": sha256(target),
              "counts": dict(stats), "categories": dict(categories),
              "difficulty_filter_applied": False, "evaluation_exclusion_applied": False}
    atomic_json(manifest, result)
    log(event="normalized_shard", source=source, file=item["path"], counts=dict(stats), categories=dict(categories))
    return result


def combine(source):
    shards = sorted((OUT / "normalized_shards" / source).glob("*.jsonl"))
    target = OUT / "normalized" / (source + ".jsonl")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".writing")
    seen, categories, manifests = {}, Counter(), []
    with temporary.open("w") as stream:
        for shard in shards:
            manifest = json.loads(shard.with_suffix(".manifest.json").read_text())
            if sha256(shard) != manifest["sha256"]:
                raise ValueError(f"Normalized shard hash mismatch: {shard}")
            manifests.append(manifest)
            for line in shard.read_text().splitlines():
                rec = json.loads(line)
                if rec["id"] in seen:
                    if seen[rec["id"]] != (rec["question_sha256"], rec["answers"], rec["image_sha256"]):
                        raise ValueError(f"Conflicting original labels: {rec['id']}")
                    continue
                seen[rec["id"]] = (rec["question_sha256"], rec["answers"], rec["image_sha256"])
                stream.write(line + "\n")
                categories[v2.classify(rec)[0]] += 1
    temporary.replace(target)
    result = {"source": source, "path": str(target.resolve()), "sha256": sha256(target),
              "rows": len(seen), "categories": dict(categories), "shards": manifests,
              "difficulty_filter_applied": False, "evaluation_exclusion_applied": False,
              "note": "Apply v3 pool leakage exclusions, then 8-rollout correctness filtering before training."}
    atomic_json(target.with_suffix(".manifest.json"), result)
    log(event="supplemental_source_complete", source=source, rows=len(seen), categories=dict(categories), path=str(target))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqav2-start", type=int, default=4)
    parser.add_argument("--vqav2-stop", type=int, default=20, help="Exclusive shard index")
    parser.add_argument("--docvqa-start", type=int, default=2)
    parser.add_argument("--docvqa-stop", type=int, default=4, help="Exclusive shard index")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--range-workers", type=int, default=4)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.range_workers < 1:
        parser.error("workers must be positive")
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / "expansion.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = []
    for source in SOURCES:
        start, stop = getattr(args, source + "_start"), getattr(args, source + "_stop")
        files = source_listing(source)
        if not 0 <= start <= stop <= len(files):
            raise ValueError(f"Invalid shard interval {source}: {start}:{stop}")
        jobs.extend((source, item) for item in files[start:stop])
    plan = {"created_at": now(), "bytes": sum(item["size"] for _, item in jobs),
            "jobs": [{"source": source, **item} for source, item in jobs],
            "sources": SOURCES, "args": vars(args), "normalizer_sha256": sha256(Path(__file__))}
    atomic_json(OUT / "plan.json", plan)
    log(event="source_plan", jobs=len(jobs), bytes=plan["bytes"])
    if args.plan_only:
        return
    base_ids = {source: {json.loads(line)["id"] for line in (HERE / "normalized" / (source + ".jsonl")).read_text().splitlines()}
                for source in SOURCES}
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(download_shard, source, item, args.range_workers): (source, item)
                   for source, item in jobs}
        for future in as_completed(pending):
            source, item = pending[future]
            path = future.result()
            results.append(normalize_shard(source, item, path, base_ids[source]))
    combined = {source: combine(source) for source in SOURCES}
    atomic_json(OUT / "complete.json", {"completed_at": now(), "sources": combined,
                                        "plan_sha256": sha256(OUT / "plan.json")})


if __name__ == "__main__":
    main()
