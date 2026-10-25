"""Pinned, full (never Lite) sources for the six high-resolution benchmarks."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "data" / "highres_perception_v1"
DEFAULT_SOURCE_ROOT = Path("/root/autodl-tmp/sharedexpert/highres_perception_sources")
BENCHMARKS = {
    "vstar": {"repo": "lmms-lab-encoder/vstar-bench", "revision": "b44023b4dca749ed8a76b85eb576627d05a1c174", "paths": ["data/test-00000-of-00001.parquet"], "expected_questions": 191},
    "zoom": {"repo": "inclusionAI/ZoomBench", "revision": "b788097e57d30510c6877824833234a73bf80d25", "paths": ["data/test.parquet"], "expected_questions": 845},
    "hrbench_4k": {"repo": "DreamMr/HR-Bench", "revision": "83b9013d6293b85dc507e87199ca52517536939c", "paths": ["hr_bench_4k.parquet"], "expected_questions": 800},
    "hrbench_8k": {"repo": "DreamMr/HR-Bench", "revision": "83b9013d6293b85dc507e87199ca52517536939c", "paths": ["hr_bench_8k.parquet"], "expected_questions": 800},
    "mme_rw_en": {"repo": "yifanzhang114/MME-RealWorld-Lmms-eval", "revision": "f157390129e8d2b60757eb68d340aade9446b52b", "paths": [f"data/train-{i:05d}-of-00070.parquet" for i in range(70)], "expected_questions": 23609},
    "mme_rw_cn": {"repo": "yifanzhang114/MME-RealWorld-CN-Lmms-eval", "revision": "37ac6f4e57ef211b37d1ee04a8f1268f1411b3e0", "paths": [f"data/train-{i:05d}-of-00013.parquet" for i in range(13)], "expected_questions": 5917},
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root):
    path = Path(root) / "source_manifest.json"
    manifest = json.loads(path.read_text())
    if manifest["benchmarks"] != BENCHMARKS or len(manifest["files"]) != 87:
        raise ValueError("Source manifest does not match the frozen six-benchmark specification")
    expected = {(b, p) for b, spec in BENCHMARKS.items() for p in spec["paths"]}
    actual = {(f["benchmark"], f["path"]) for f in manifest["files"]}
    if actual != expected:
        raise ValueError("Source manifest has missing or unexpected files")
    for f in manifest["files"]:
        spec = BENCHMARKS[f["benchmark"]]
        if f["repo"] != spec["repo"] or f["revision"] != spec["revision"]:
            raise ValueError("Source manifest revision mismatch")
        if len(f["sha256"]) != 64 or f["size"] <= 0:
            raise ValueError("Source file has no valid SHA256/size")
        if f["local_path"] != str(Path("sources") / f["benchmark"] / f["path"]):
            raise ValueError("Unexpected local source path")
    return manifest


def create_manifest(root, endpoint="https://hf-mirror.com"):
    root = Path(root)
    if (root / "source_manifest.json").exists():
        return load_manifest(root)
    cache = {}
    files = []
    session = requests.Session()
    session.trust_env = False
    for benchmark, spec in BENCHMARKS.items():
        key = (spec["repo"], spec["revision"])
        if key not in cache:
            url = f"{endpoint.rstrip('/')}/api/datasets/{key[0]}/revision/{key[1]}"
            for attempt in range(6):
                try:
                    response = session.get(url, params={"blobs": "true"}, timeout=(20, 60))
                    response.raise_for_status()
                    info = response.json()
                    if info["sha"] != key[1]:
                        raise ValueError("Metadata API returned a different revision")
                    cache[key] = {f["rfilename"]: f for f in info["siblings"]}
                    break
                except requests.RequestException:
                    if attempt == 5:
                        raise
                    time.sleep(min(2 ** attempt, 20))
        for path in spec["paths"]:
            entry = cache[key][path]
            lfs = entry.get("lfs", {})
            sha = lfs.get("sha256", lfs.get("oid", ""))
            if not sha or len(sha) != 64:
                raise ValueError(f"No upstream SHA256 for {key[0]}/{path}")
            files.append({"benchmark": benchmark, "repo": key[0], "revision": key[1], "path": path,
                          "size": entry["size"], "sha256": sha,
                          "local_path": str(Path("sources") / benchmark / path)})
    manifest = {"schema_version": 1, "created_at": utc_now(), "benchmarks": BENCHMARKS,
                "file_count": len(files), "total_bytes": sum(f["size"] for f in files), "files": files}
    atomic_json(root / "source_manifest.json", manifest)
    load_manifest(root)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = create_manifest(args.root)
    print(json.dumps({"manifest": str(args.root / "source_manifest.json"), "files": result["file_count"], "bytes": result["total_bytes"]}))
