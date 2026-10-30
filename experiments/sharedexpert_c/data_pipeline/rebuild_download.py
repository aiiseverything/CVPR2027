#!/usr/bin/env python3
"""Download the official sources needed to rebuild the generalist mixture.

Uses the Hugging Face mirror because huggingface.co is unreachable from this
host. Files are placed under data_pipeline/raw/ in the layout expected by
build.py (with Geometry3K/MMK12 standing in for the missing old RLVR tree).
"""
from __future__ import annotations

import concurrent.futures as cf
import fcntl
import fnmatch
import hashlib
import json
import shutil
import subprocess
import time
import requests
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
HF = "https://hf-mirror.com"
HEADERS = {"User-Agent": "sharedexpert-rebuild/1.0"}
JOBS = []


def add_hf(repo, rev, patterns, dest, limit=None):
    url = f"{HF}/api/datasets/{repo}/tree/{rev}?recursive=true"
    response = requests.get(url, headers=HEADERS, timeout=60)
    response.raise_for_status()
    listing = response.json()
    selected = [x for x in listing if x.get("type") == "file"
                and any(fnmatch.fnmatch(x["path"], p) for p in patterns)]
    selected.sort(key=lambda x: x["path"])
    if limit:
        selected = selected[:limit]
    for item in selected:
        JOBS.append((f"{HF}/datasets/{repo}/resolve/{rev}/{item['path']}?download=true",
                     RAW / dest / item["path"], item.get("size", 0)))


def download_ranges(url, out, size, workers=16, chunk=8 << 20):
    """Chunked HTTP ranges: the mirror drops long single connections."""
    root = out.with_name(out.name + ".ranges")
    root.mkdir(parents=True, exist_ok=True)
    starts = list(range(0, size, chunk))
    done = 0

    def one(start):
        end = min(start + chunk, size) - 1
        part = root / f"{start:012d}.part"
        if part.is_file() and part.stat().st_size == end - start + 1:
            return
        for attempt in range(12):
            try:
                fresh = url + ("&" if "?" in url else "?") + f"fresh={start}-{attempt}"
                result = subprocess.run(["curl", "-sS", "-L", "--fail", "-r", f"{start}-{end}",
                                         "--connect-timeout", "15", "--max-time", "120",
                                         "-o", str(part), fresh],
                                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                if result.returncode or not part.is_file() or part.stat().st_size != end - start + 1:
                    raise RuntimeError(f"range {start} curl {result.returncode}")
                return
            except Exception:
                if part.exists():
                    part.unlink()
                time.sleep(min(10, attempt + 1))
        raise RuntimeError(f"range failed after retries: {start}")

    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, start) for start in starts]
        for index, future in enumerate(cf.as_completed(futures), 1):
            future.result()
            if index % 32 == 0:
                print(json.dumps({"range_progress": index, "of": len(starts), "file": str(out)}), flush=True)
    temporary = out.with_name(out.name + ".assembling")
    with temporary.open("wb") as handle:
        for start in starts:
            handle.write((root / f"{start:012d}.part").read_bytes())
    if temporary.stat().st_size != size:
        raise RuntimeError(f"assembled size mismatch {temporary.stat().st_size} != {size}")
    temporary.replace(out)
    import shutil as _shutil
    _shutil.rmtree(root, ignore_errors=True)
    return {"url": url, "path": str(out), "bytes": size, "sha256": sha256_file(out)}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def download(job):
    url, out, expected = job
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.is_file() and expected and out.stat().st_size == expected:
        return {"url": url, "path": str(out), "bytes": out.stat().st_size,
                "sha256": sha256_file(out)}
    if expected and expected > (64 << 20):
        return download_ranges(url, out, expected)
    part = out.with_name(out.name + ".part")
    if expected and part.is_file() and part.stat().st_size == expected:
        part.replace(out)
        return {"url": url, "path": str(out), "bytes": expected, "sha256": sha256_file(out)}
    command = ["curl", "-L", "--fail", "--retry", "5", "--retry-delay", "3",
               "--connect-timeout", "20", "--max-time", "3600", "-C", "-",
               "-o", str(part), url]
    for attempt in range(6):
        try:
            result = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            if result.returncode in (33, 416) and part.exists():
                part.unlink()
                raise RuntimeError(f"server does not support resume (curl {result.returncode}); partial removed")
            if result.returncode:
                raise RuntimeError(f"curl exit {result.returncode}: {result.stderr.decode()[:200]}")
            size = part.stat().st_size
            if expected and size != expected:
                raise ValueError(f"size mismatch {size} != {expected}")
            part.replace(out)
            return {"url": url, "path": str(out), "bytes": size, "sha256": sha256_file(out)}
        except Exception as exc:
            print(json.dumps({"retry": attempt + 1, "file": str(out),
                              "error": type(exc).__name__, "detail": str(exc)[:200]}), flush=True)
            if attempt == 5:
                raise
            time.sleep(min(20, 2 ** attempt))


def main():
    lock = open('/tmp/sharedexpert-download.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    add_hf("hiyouga/geometry3k", "fd21e533e1e50d0662a2bf7b223e60511bd5f8b7",
           ["data/train-*.parquet"], "geometry3k", 1)
    add_hf("FanqingM/MMK12", "372a609268ea79b5e78d90ab173e02c37b486163",
           ["data/train-*.parquet", "data/test-*.parquet"], "mmk12")
    add_hf("TIGER-Lab/ViRL39K", "812ec617dea4bc8a4e751663b88e4ebb7de4d00e",
           ["39Krelease.parquet", "images.zip"], "virl39k")
    add_hf("HuggingFaceM4/ChartQA", "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5",
           ["data/train-*.parquet"], "chartqa")
    add_hf("lmms-lab-encoder/VQAv2", "32665d35052eb4a6d4414851c3c829a72754915a",
           ["data/train-0000[0-3]-of-00136.parquet"], "vqav2")
    add_hf("lmms-lab-encoder/DocVQA", "539088ef8a8ada01ac8e2e6d4e372586748a265e",
           ["DocVQA/train-0000[0-1]-of-00012.parquet"], "docvqa")
    add_hf("HuggingFaceM4/the_cauldron", "847a98a779b1652d65111daf20c972dfcd333605",
           ["tallyqa/*"], "cauldron")
    # The original protocol used random.Random(20261019).shuffle(range(24))[:3],
    # which selects exactly these three official shards (indices 1, 14, 21).
    add_hf("HuggingFaceM4/the_cauldron", "847a98a779b1652d65111daf20c972dfcd333605",
           ["clevr/train-00001-of-00024-8711717f841a0ad7.parquet",
            "clevr/train-00014-of-00024-6ff949aec9343112.parquet",
            "clevr/train-00021-of-00024-911403b3f4a7ad9f.parquet"], "cauldron")
    JOBS.append(("https://raw.githubusercontent.com/manoja328/TallyQA_dataset/"
                 "46cdc649ec79c3dcc2720ff227ad07d7ee51da6f/tallyqa.zip",
                 RAW / "tallyqa_annotations.zip", 0))
    JOBS.append(("https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip",
                 RAW / "clevr_no_images.zip", 0))
    (RAW / "refcoco/data").mkdir(parents=True, exist_ok=True)
    for source in (ROOT.parent / "build_sources/refcoco/data").glob("*.parquet"):
        target = RAW / "refcoco/data" / source.name
        if not target.exists():
            shutil.copy2(source, target)
    print(json.dumps({"jobs": len(JOBS), "targets": [str(j[1]) for j in JOBS]}), flush=True)
    failures = []
    ordered = sorted(JOBS, key=lambda job: -(job[2] or 0))
    with (RAW / "rebuild_downloads.jsonl").open("a") as log:
        for index, job in enumerate(ordered, 1):
            try:
                record = download(job)
                log.write(json.dumps(record) + "\n")
                log.flush()
                print(json.dumps({"done": index, "of": len(JOBS), "bytes": record["bytes"],
                                  "path": record["path"]}), flush=True)
            except Exception as exc:
                failures.append((str(job[1]), str(exc)))
                print(json.dumps({"failed": str(job[1]), "error": str(exc)[:300]}), flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} downloads failed: {failures[:3]}")
    print("ALL_DOWNLOADS_DONE", flush=True)


if __name__ == "__main__":
    main()
