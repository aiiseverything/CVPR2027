#!/usr/bin/env python3
"""Recreate the pinned-source provenance sidecars needed by build.py finalize.

The original sidecars lived on the previous training server and are not in git.
This script checks every downloaded payload against the tracked
prepared/source_validation.json manifest (upstream LFS SHA256 and byte size)
and rewrites equivalent local manifests. It never invents a hash: all values
come from the tracked manifest or from the downloaded bytes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
MANIFEST = ROOT / "prepared" / "source_validation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def local_path(entry: dict) -> Path:
    """Map a manifest path from the old server onto this workspace's raw tree."""
    raw_path = entry["path"].split("/raw/", 1)[-1]
    if raw_path.startswith("rlvr/"):
        raw_path = raw_path[len("rlvr/"):]
    return RAW / raw_path


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest["files"]

    primary, reused, cauldron, images_expected = [], [], [], None
    for entry in entries:
        path = local_path(entry)
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_bytes = path.stat().st_size
        actual_sha = sha256(path)
        if actual_bytes != entry["bytes"] or actual_sha != entry["sha256"]:
            raise RuntimeError(f"payload mismatch: {path} ({actual_bytes} vs {entry['bytes']}, {actual_sha} vs {entry['sha256']})")
        repo, revision = entry["repo"], entry["revision"]
        if "cauldron/" in entry["path"]:
            rel = entry["path"].split("raw/cauldron/", 1)[-1] if "raw/cauldron/" in entry["path"] else entry["path"].split("cauldron/", 1)[-1]
            cauldron.append({"path": rel, "size": entry["bytes"], "lfs": {"oid": entry["sha256"], "size": entry["bytes"]}})
        elif entry["path"].split("/raw/", 1)[-1].startswith(("rlvr/geometry3k", "rlvr/mmk12")):
            reused.append({"path": str(path.resolve()), "expected_etag": entry["sha256"], "bytes": entry["bytes"],
                           "repo": repo, "revision": revision})
        elif entry["path"].endswith("virl39k/images.zip"):
            images_expected = {"lfs": {"oid": entry["sha256"], "size": entry["bytes"]}, "size": entry["bytes"]}
        else:
            primary.append({"file": str(path.resolve()), "upstream_lfs_sha256": entry["sha256"],
                            "bytes": entry["bytes"], "repo": repo, "revision": revision})

    if images_expected is None:
        raise RuntimeError("images.zip is missing from the source manifest")
    if not cauldron or not reused or not primary:
        raise RuntimeError("provenance groups are incomplete")

    (RAW / "primary_lfs_verification.json").write_text(json.dumps(primary, indent=2))
    (RAW / "reused_source_verification.json").write_text(json.dumps(reused, indent=2))
    (RAW / "cauldron_lfs_manifest.json").write_text(json.dumps(cauldron, indent=2))
    (RAW / "virl39k" / "images_expected.json").write_text(json.dumps(images_expected, indent=2))

    # Rebuild the exact ViRL39K qids that derive from the original MMK12 test
    # split. build.py removes them before normalization; audit_prepared.py
    # independently asserts their absence from the prepared mixture.
    import pyarrow.parquet as pq

    test_ids = set()
    for path_ in sorted((RAW / "mmk12" / "data").glob("test-*.parquet")):
        test_ids.update(str(row["id"]) for row in pq.read_table(path_, columns=["id"]).to_pylist())
    if not test_ids:
        raise RuntimeError("MMK12 test split is empty or missing")
    release = pq.read_table(RAW / "virl39k" / "39Krelease.parquet", columns=["qid", "source"]).to_pylist()
    excluded, per_source = set(), {}
    for row in release:
        source = row["source"]
        if source not in ("MMK12", "MMK12Converted"):
            continue
        qid = row["qid"]
        origin = qid[len(source) + 1:] if qid.startswith(source + "-") else qid
        if origin in test_ids:
            excluded.add(qid)
            per_source[source] = per_source.get(source, 0) + 1
    audit = {"excluded_original_test_qids": sorted(excluded),
             "mmk12_test_ids": len(test_ids),
             "excluded_by_source": per_source,
             "derivation": "ViRL39K qid source prefix in {MMK12, MMK12Converted} and original id in local MMK12 test parquet"}
    (RAW / "virl39k_original_split_audit.json").write_text(json.dumps(audit, indent=2))

    print(json.dumps({"primary": len(primary), "reused": len(reused), "cauldron": len(cauldron),
                      "virl_images_expected": images_expected["lfs"]["size"],
                      "excluded_original_test_qids": len(excluded), "mmk12_test_ids": len(test_ids)}, indent=2))


if __name__ == "__main__":
    main()
