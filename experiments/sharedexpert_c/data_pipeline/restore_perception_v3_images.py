#!/usr/bin/env python3
"""Restore the perception-v3 supplemental image cache from verified raw shards.

The interrupted 2026-10-29 screening run references images below
``perception_v3_sources/images`` (a symlink to /root/autodl-tmp/...), which was
not carried over to this server. The pinned raw parquet shards are intact, so
this tool re-extracts exactly the images referenced by the frozen candidate
pools, verifies both the blob SHA256 and the decoded RGB pixel hash stored in
those pools, and writes a separate receipt.

The migration receipt records only the first migration batch; later expansion
rounds wrote directly into the live directory, so the authoritative missing set
is derived from the candidate records themselves.

It never rewrites candidate pools, normalized records, the screening journal,
the run fingerprint, or any model/sampling/scoring input.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SOURCES = HERE / "perception_v3_sources"
POOLS = [HERE / "prepared/perception_v3_pool",
         HERE / "prepared/perception_v3_pool_round2",
         HERE / "prepared/perception_v3_pool_round3"]


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pixel_hash(path):
    from PIL import Image
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return hashlib.sha256(struct.pack(">II", *rgb.size) + rgb.tobytes()).hexdigest()


def atomic_write(path, blob):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    with temporary.open("wb") as f:
        f.write(blob)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def load_raw_jobs():
    """Every raw shard from all expansion rounds, pinned by plan or normalized manifest.

    ``plan.json`` only describes the most recent expansion, while the referenced
    image set spans several rounds. Each earlier round has a normalized-shard
    manifest that records the upstream parquet SHA256 and byte count, so every
    raw file still has an independent expected identity.
    """
    plan = json.loads((SOURCES / "plan.json").read_text())
    planned = {}
    for item in plan["jobs"]:
        path = SOURCES / "raw" / item["source"] / item["path"]
        planned[str(path.resolve())] = {"source": item["source"], "bytes": item["size"],
                                        "sha256": item["lfs"]["oid"], "provenance": "plan.json"}
    jobs, missing_identity = [], []
    raw_root = SOURCES / "raw"
    for path in sorted(raw_root.glob("*/*.parquet")) + sorted(raw_root.glob("*/*/*.parquet")):
        source = path.relative_to(raw_root).parts[0]
        manifest_path = SOURCES / "normalized_shards" / source / (path.stem + ".manifest.json")
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else None
        expected = planned.get(str(path.resolve()))
        if manifest is not None:
            identity = {"source": source, "bytes": manifest["source_bytes"],
                        "sha256": manifest["source_sha256"],
                        "provenance": "normalized_shards/%s/%s" % (source, manifest_path.name)}
            if expected is not None and (expected["bytes"], expected["sha256"]) != \
                    (identity["bytes"], identity["sha256"]):
                raise RuntimeError(f"plan.json and normalized manifest disagree for {path}")
            expected = identity
        if expected is None:
            missing_identity.append(str(path))
            continue
        jobs.append({"path": path, **{k: expected[k] for k in ("source", "bytes", "sha256")},
                     "provenance": expected["provenance"]})
    if missing_identity:
        raise RuntimeError(f"Raw shards without a pinned identity: {missing_identity[:5]}")
    return jobs


def verify_raw(jobs):
    failures = []
    for job in jobs:
        path = job["path"]
        if not path.is_file():
            failures.append({"path": str(path), "reason": "missing"})
        elif path.stat().st_size != job["bytes"]:
            failures.append({"path": str(path), "reason": "size",
                             "actual": path.stat().st_size, "expected": job["bytes"]})
        elif sha256_file(path) != job["sha256"]:
            failures.append({"path": str(path), "reason": "sha256"})
    return failures


def load_needed(destination):
    """Referenced supplemental images, grouped by source and blob digest."""
    referenced = defaultdict(set)
    records = 0
    for pool in POOLS:
        for line in (pool / "candidates.records.jsonl").open():
            record = json.loads(line)
            records += 1
            for path, pixels in zip(record["images"], record["image_sha256"], strict=True):
                resolved = str(Path(path).resolve())
                if Path(resolved).parent.parent == destination:
                    referenced[resolved].add(pixels)
    migration = json.loads((SOURCES / "image_migration.json").read_text())
    if migration.get("stage") != "complete":
        raise RuntimeError("Image migration receipt is not complete")
    if Path(migration["destination"]).resolve() != destination:
        raise RuntimeError("Migration receipt destination differs from the images symlink")
    needed = defaultdict(dict)
    for item in migration["files"]:
        source, name = item["relative_path"].split("/", 1)
        needed[source][name.removesuffix(".img")] = item["bytes"]
    for path in referenced:
        source, digest = Path(path).parent.name, Path(path).stem
        needed[source].setdefault(digest, None)
    return referenced, needed, records, migration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--skip-raw-verification", action="store_true")
    parser.add_argument("--receipt", type=Path, default=SOURCES / "image_restore.json")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")

    import pyarrow.parquet as pq

    destination = (SOURCES / "images").resolve()
    destination.mkdir(parents=True, exist_ok=True)
    referenced, needed, records, migration = load_needed(destination)
    print(json.dumps({"stage": "needed", "referenced_files": len(referenced),
                      "migration_files": len(migration["files"]),
                      "cache_files": sum(map(len, needed.values())),
                      "destination": str(destination)}), flush=True)

    present = 0
    for source, files in needed.items():
        for digest, size in files.items():
            path = destination / source / f"{digest}.img"
            if path.is_file() and sha256_file(path) == digest and \
                    (size is None or path.stat().st_size == size):
                present += 1

    jobs = load_raw_jobs()
    raw_failures = [] if args.skip_raw_verification else verify_raw(jobs)
    if raw_failures:
        raise RuntimeError(f"Raw shard verification failed: {raw_failures[:5]}")

    extracted, seen = 0, Counter()
    missing_bytes = 0
    for job in jobs:
        source = job["source"]
        wanted = needed.get(source, {})
        if not wanted:
            continue
        parquet = pq.ParquetFile(job["path"])
        for batch in parquet.iter_batches(batch_size=32, columns=["image"]):
            for row in batch.to_pylist():
                image = row.get("image")
                blob = image.get("bytes") if isinstance(image, dict) else None
                if not isinstance(blob, (bytes, bytearray)):
                    missing_bytes += 1
                    continue
                digest = hashlib.sha256(blob).hexdigest()
                if digest not in wanted:
                    continue
                seen[source] += 1
                size = wanted[digest]
                if size is not None and len(blob) != size:
                    raise RuntimeError(f"Extracted blob size differs for {digest}")
                path = destination / source / f"{digest}.img"
                if path.is_file() and sha256_file(path) == digest:
                    continue
                atomic_write(path, blob)
                extracted += 1

    absent, bad, total_bytes = [], [], 0
    for source, files in needed.items():
        for digest, size in files.items():
            path = destination / source / f"{digest}.img"
            if not path.is_file():
                absent.append(str(path))
            elif sha256_file(path) != digest or (size is not None and path.stat().st_size != size):
                bad.append(str(path))
            else:
                total_bytes += path.stat().st_size
    if absent or bad:
        raise RuntimeError(f"Restored image verification failed: absent={absent[:5]} bad={bad[:5]}")

    # Recompute the decoded RGB hash exactly as prepare_perception_v2.pixel_hash does.
    checked, mismatches = 0, []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        paths = sorted(referenced)
        for path, actual in zip(paths, pool.map(pixel_hash, paths, chunksize=16), strict=True):
            checked += 1
            if actual not in referenced[path]:
                mismatches.append({"path": path, "actual": actual,
                                   "expected": sorted(referenced[path])})
    if mismatches:
        raise RuntimeError(f"Decoded pixel hash mismatch: {mismatches[:5]}")

    # Final gate: every image path in every candidate pool must now exist.
    missing_candidates = []
    for pool in POOLS:
        for line in (pool / "candidates.records.jsonl").open():
            record = json.loads(line)
            for path in record["images"]:
                if not Path(path).is_file():
                    missing_candidates.append({"record": record["id"], "path": path})
    if missing_candidates:
        raise RuntimeError(f"Candidate images still missing: {missing_candidates[:5]}")

    receipt = {
        "schema": 1,
        "status": "complete",
        "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "destination": str(destination),
        "migration_receipt_sha256": sha256_file(SOURCES / "image_migration.json"),
        "tool_sha256": sha256_file(Path(__file__)),
        "referenced_files": len(referenced),
        "migration_files": len(migration["files"]),
        "cache_files": sum(map(len, needed.values())),
        "cache_bytes": total_bytes,
        "already_present": present,
        "extracted": extracted,
        "source_blobs_scanned": dict(seen),
        "rows_without_image_bytes": missing_bytes,
        "pixel_verified_files": checked,
        "candidate_records_checked": records,
        "candidate_image_paths_verified": sum(len(json.loads(line)["images"])
                                             for pool in POOLS
                                             for line in (pool / "candidates.records.jsonl").open()),
        "raw_shards": [{"path": str(job["path"]), "source": job["source"],
                        "sha256": job["sha256"], "bytes": job["bytes"],
                        "provenance": job["provenance"]} for job in jobs],
        "raw_verification": "skipped" if args.skip_raw_verification else "passed",
        "scope": "Only the referenced supplemental vqav2/docvqa image cache; no pool, journal, or fingerprint changed",
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.receipt.with_name(args.receipt.name + ".writing")
    temporary.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(args.receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == "__main__":
    main()
