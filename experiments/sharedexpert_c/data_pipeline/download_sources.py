#!/usr/bin/env python3
"""Download pinned, publicly released source data; never executes dataset scripts."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import time
import requests
import socket
import urllib3.util.connection
urllib3.util.connection.allowed_gai_family = lambda: socket.AF_INET

ROOT = Path(__file__).resolve().parent
HF_ENDPOINT = "https://hf-mirror.com"
HF_REPOS = {
    "virl39k": ("TIGER-Lab/ViRL39K", "812ec617dea4bc8a4e751663b88e4ebb7de4d00e", ["39Krelease.parquet", "images.zip", "README.md"]),
    "refcoco": ("jxu124/refcoco", "9fba8200c5326e996f789191f095bd464ef1d09e", ["data/*.parquet", "README.md"]),
    "chartqa": ("HuggingFaceM4/ChartQA", "b605b6e08b57faf4359aeb2fe6a3ca595f99b6c5", ["data/train-*", "README.md"]),
    "cauldron": ("HuggingFaceM4/the_cauldron", "847a98a779b1652d65111daf20c972dfcd333605", ["tallyqa/*", "vqav2/*", "docvqa/*", "README.md"]),
    "vqav2": ("lmms-lab/VQAv2", "32665d35052eb4a6d4414851c3c829a72754915a", ["data/train-00000-*", "data/train-00001-*", "data/train-00002-*", "data/train-00003-*", "README.md"]),
    "docvqa": ("lmms-lab/DocVQA", "539088ef8a8ada01ac8e2e6d4e372586748a265e", ["DocVQA/train-00000-*", "DocVQA/train-00001-*", "README.md"]),
}
DIRECT = {
    "clevr": ("https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip", "clevr/CLEVR_v1.0.zip"),
    "coco": ("https://images.cocodataset.org/zips/train2014.zip", "coco/train2014.zip"),
    "tallyqa_annotations": ("https://raw.githubusercontent.com/manoja328/TallyQA_dataset/46cdc649ec79c3dcc2720ff227ad07d7ee51da6f/tallyqa.zip", "tallyqa/tallyqa.zip"),
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def download(job):
    url, out, expected_size = job
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and (not expected_size or out.stat().st_size == expected_size):
        return {"url": url, "path": str(out), "bytes": out.stat().st_size, "sha256": sha256(out)}
    part = out.with_name(out.name + ".part")
    for attempt in range(6):
        try:
            with requests.get(url, stream=True, timeout=(30, 180)) as r:
                r.raise_for_status()
                size = int(r.headers.get("Content-Length", 0))
                with open(part, "wb") as f:
                    for chunk in r.iter_content(8 << 20):
                        if chunk:
                            f.write(chunk)
            if expected_size and part.stat().st_size != expected_size:
                raise ValueError("Downloaded size mismatch")
            if size and part.stat().st_size != size:
                raise ValueError("HTTP Content-Length mismatch")
            part.replace(out)
            return {"url": url, "path": str(out), "bytes": out.stat().st_size, "sha256": sha256(out)}
        except Exception as e:
            print(json.dumps({"retry": attempt + 1, "file": str(out), "error": type(e).__name__}), flush=True)
            if attempt == 5:
                raise
            time.sleep(min(30, 2 ** attempt))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=list(HF_REPOS) + list(DIRECT))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--cauldron-limit", type=int, default=0, help="0 downloads all shards; nonzero is an explicitly recorded fixed shard subset")
    args = ap.parse_args()
    import fnmatch
    jobs = []
    meta = {}
    for key in args.sources:
        if key in DIRECT:
            url, rel = DIRECT[key]
            jobs.append((url, ROOT / "raw" / rel, 0))
            continue
        repo, rev, patterns = HF_REPOS[key]
        response = requests.get(f"{HF_ENDPOINT}/api/datasets/{repo}/revision/{rev}", timeout=(15, 60))
        response.raise_for_status()
        listing = [{"path": x["rfilename"], "type": "file"} for x in response.json()["siblings"]]
        selected = [x for x in listing if x["type"] == "file" and any(fnmatch.fnmatch(x["path"], p) for p in patterns)]
        if key == "cauldron" and args.cauldron_limit:
            counts = {}
            filtered = []
            for x in sorted(selected, key=lambda x: x["path"]):
                group = x["path"].split("/")[0]
                counts[group] = counts.get(group, 0) + 1
                if counts[group] <= args.cauldron_limit:
                    filtered.append(x)
            selected = filtered
        meta[key] = {"repo": repo, "revision": rev, "files": [x["path"] for x in selected]}
        for x in selected:
            jobs.append((f"{HF_ENDPOINT}/datasets/{repo}/resolve/{rev}/{x['path']}?download=true", ROOT / "raw" / key / x["path"], x.get("size", 0)))
    (ROOT / "raw" / ("source_revisions_" + "_".join(args.sources) + ".json")).write_text(json.dumps(meta, indent=2))
    manifest = ROOT / "raw" / "downloads.jsonl"
    failures = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool, open(manifest, "a") as f:
        mapping = {pool.submit(download, job): job for job in jobs}
        for future in cf.as_completed(mapping):
            try:
                rec = future.result()
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(json.dumps({"complete": rec["path"], "bytes": rec["bytes"]}), flush=True)
            except Exception as e:
                failures.append(str(mapping[future][1]))
                print(json.dumps({"failed": str(mapping[future][1]), "error": type(e).__name__}), flush=True)
    if failures:
        raise SystemExit("Missing source files: " + ", ".join(failures))

if __name__ == "__main__":
    main()
