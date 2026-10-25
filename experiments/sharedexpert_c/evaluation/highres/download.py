"""Resumable, SHA256-verified downloads; no snapshot or image-file expansion."""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import errno
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests

from sources import DEFAULT_ROOT, DEFAULT_SOURCE_ROOT, atomic_json, create_manifest, file_sha256, load_manifest, utc_now


def log(event, **values):
    print(json.dumps({"time": utc_now(), "event": event, **values}), flush=True)


def reserve_bounded_space(stream, expected_size, minimum_free):
    """Bound XFS allocation without extending the resumable logical EOF."""
    descriptor = stream.fileno()
    before = os.fstat(descriptor).st_size
    libc = ctypes.CDLL(None, use_errno=True)
    fallocate = libc.fallocate
    fallocate.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_longlong, ctypes.c_longlong]
    fallocate.restype = ctypes.c_int
    block = os.fstatvfs(descriptor).f_frsize
    end = ((expected_size + block - 1) // block) * block
    # No valid byte can exist beyond the manifest's total size. Remove only
    # speculative extents after that boundary, never downloaded data.
    if fallocate(descriptor, 3, end, 1 << 40) != 0:  # PUNCH_HOLE | KEEP_SIZE
        error = ctypes.get_errno()
        if error not in {errno.EOPNOTSUPP, errno.ENOSYS}:
            raise OSError(error, "Could not trim out-of-range speculative allocation")
    allocated = os.fstat(descriptor).st_blocks * 512
    available = os.fstatvfs(descriptor).f_bavail * block
    if available < max(0, expected_size - allocated) + minimum_free:
        raise OSError(errno.ENOSPC, "Insufficient disk headroom for the complete pinned shard")
    if fallocate(descriptor, 1, 0, expected_size) != 0:  # KEEP_SIZE: EOF unchanged
        error = ctypes.get_errno()
        raise OSError(error, "Could not reserve bounded shard capacity")
    if os.fstat(descriptor).st_size != before:
        raise ValueError("KEEP_SIZE unexpectedly changed the resume offset")


def migrate_source(old_root, root, entry):
    """Move only this pipeline's source cache after byte-for-byte verification."""
    if old_root is None:
        return
    old = old_root / entry["local_path"]
    target = root / entry["local_path"]
    for source, destination, full in ((old, target, True),
            (old.with_suffix(old.suffix + ".part"), target.with_suffix(target.suffix + ".part"), False)):
        if not source.is_file() or destination.exists():
            continue
        before = source.stat()
        if full and (before.st_size != entry["size"] or file_sha256(source) != entry["sha256"]):
            raise ValueError("Old complete source does not match its pinned SHA256")
        expected = entry["sha256"] if full else file_sha256(source)
        temporary = destination.with_name(destination.name + ".moving")
        log("migration_start", file=entry["local_path"], partial=not full, bytes=before.st_size)
        with source.open("rb") as inp, temporary.open("wb") as out:
            shutil.copyfileobj(inp, out, length=8 * 1024 * 1024)
            out.flush()
            os.fsync(out.fileno())
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or file_sha256(temporary) != expected:
            raise ValueError("Source changed or cross-disk migration checksum failed; original retained")
        os.replace(temporary, destination)
        source.unlink()
        log("migration_complete", file=entry["local_path"], partial=not full, bytes=before.st_size, sha256=expected)
    receipt = old.with_suffix(old.suffix + ".verified.json")
    if receipt.exists() and target.exists():
        shutil.copy2(receipt, target.with_suffix(target.suffix + ".verified.json"))


def download_file(root, entry, retries, endpoint, minimum_free, old_root=None):
    target = root / entry["local_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    receipt = target.with_suffix(target.suffix + ".verified.json")
    migrate_source(old_root, root, entry)
    if target.exists() and target.stat().st_size == entry["size"] and file_sha256(target) == entry["sha256"]:
        atomic_json(receipt, {"verified_at": utc_now(), "sha256": entry["sha256"], "size": entry["size"]})
        log("already_verified", file=entry["local_path"])
        return entry["local_path"]
    url = f"{endpoint.rstrip('/')}/datasets/{entry['repo']}/resolve/{entry['revision']}/{quote(entry['path'], safe='/')}"
    session = requests.Session()
    session.trust_env = False
    for attempt in range(retries):
        try:
            offset = part.stat().st_size if part.exists() else 0
            if offset > entry["size"]:
                part.unlink()
                offset = 0
            if offset < entry["size"]:
                headers = {"Accept-Encoding": "identity"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                # Cache-busting avoids expired signed CDN redirects on a resumed download.
                with session.get(url, params={"download": "true", "attempt": f"{os.getpid()}-{attempt}-{int(time.time())}"},
                                  headers=headers, timeout=(30, 90), stream=True) as response:
                    response.raise_for_status()
                    if response.status_code == 206:
                        if not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                            raise ValueError("Invalid Content-Range for resumed download")
                    elif response.status_code == 200:
                        offset = 0
                    else:
                        raise ValueError(f"Unexpected response status {response.status_code}")
                    log("download_start", file=entry["local_path"], offset=offset, size=entry["size"], attempt=attempt + 1)
                    written = offset
                    last_report = time.monotonic()
                    with part.open("ab" if offset else "wb") as stream:
                        reserve_bounded_space(stream, entry["size"], minimum_free)
                        for chunk in response.iter_content(4 * 1024 * 1024):
                            if not chunk:
                                continue
                            if shutil.disk_usage(target.parent).free < minimum_free:
                                raise OSError("Source disk reached the reserved free-space floor")
                            written += len(chunk)
                            if written > entry["size"]:
                                raise ValueError("Download exceeded frozen source size")
                            stream.write(chunk)
                            if time.monotonic() - last_report > 60:
                                log("download_progress", file=entry["local_path"], bytes=written, size=entry["size"])
                                last_report = time.monotonic()
                        stream.flush()
                        os.fsync(stream.fileno())
            if part.stat().st_size != entry["size"]:
                raise ValueError(f"Incomplete download: {part.stat().st_size}/{entry['size']}")
            digest = file_sha256(part)
            if digest != entry["sha256"]:
                part.unlink()
                raise ValueError(f"SHA256 mismatch for {entry['local_path']}")
            os.replace(part, target)
            atomic_json(receipt, {"verified_at": utc_now(), "sha256": digest, "size": entry["size"]})
            log("verified", file=entry["local_path"], sha256=digest, size=entry["size"])
            return entry["local_path"]
        except (requests.RequestException, OSError, ValueError) as error:
            # Do not print URLs: signed redirects may contain access tokens.
            log("download_retry", file=entry["local_path"], attempt=attempt + 1, error_type=type(error).__name__)
            if attempt + 1 == retries:
                raise RuntimeError(f"Failed after {retries} attempts: {entry['local_path']}") from None
            time.sleep(min(2 ** (attempt + 1), 120))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--retries", type=int, default=12)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--reuse-root", type=Path)
    parser.add_argument("--minimum-free-gib", type=float, default=2.0)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    if args.root.resolve().name == "highres_v1":
        parser.error("The cancelled full-suite root must not be reused for perception-only evaluation")
    args.source_root.mkdir(parents=True, exist_ok=True)
    link = args.root / "sources"
    if link.is_symlink():
        if link.resolve() != args.source_root.resolve():
            raise ValueError("Sources symlink points to a different cache")
    elif link.exists():
        raise ValueError("New perception sources must be an explicit temporary-disk symlink")
    else:
        link.symlink_to(args.source_root.resolve(), target_is_directory=True)
    lock = (args.root / "download.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = args.root / "download_state.json"
    state = {"pid": os.getpid(), "started_at": utc_now(), "status": "manifest", "completed": [], "failed": []}
    atomic_json(state_path, state)
    if args.reuse_root and not (args.root / "source_manifest.json").exists():
        old_manifest = load_manifest(args.reuse_root)
        atomic_json(args.root / "source_manifest.json", old_manifest)
    manifest = create_manifest(args.root, args.endpoint)
    minimum_free = int(args.minimum_free_gib * 2 ** 30)
    cached = 0
    reserved = 0
    for entry in manifest["files"]:
        target = args.root / entry["local_path"]
        candidates = [target, target.with_suffix(target.suffix + ".part")]
        cached += min(entry["size"], max((p.stat().st_size for p in candidates if p.exists()), default=0))
        reserved += min(entry["size"], max((p.stat().st_blocks * 512 for p in candidates if p.exists()), default=0))
    available = shutil.disk_usage(args.source_root).free
    needed = manifest["total_bytes"] - reserved
    if available < needed + minimum_free:
        state.update(status="failed", error="Insufficient temporary disk headroom", available_bytes=available, remaining_bytes=needed)
        atomic_json(state_path, state)
        raise RuntimeError("Complete frozen sources plus reserved headroom do not fit the source disk")
    state.update(status="downloading", file_count=len(manifest["files"]), total_bytes=manifest["total_bytes"])
    state.update(endpoint=args.endpoint, trust_environment_proxy=False, source_root=str(args.source_root.resolve()),
                 minimum_free_bytes=minimum_free, remaining_download_bytes=manifest["total_bytes"] - cached,
                 additional_disk_allocation_bytes=needed, bounded_keep_size_allocation=True)
    atomic_json(state_path, state)
    log("manifest_ready", files=len(manifest["files"]), total_bytes=manifest["total_bytes"])
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        jobs = {executor.submit(download_file, args.root, entry, args.retries, args.endpoint,
                                minimum_free, args.reuse_root): entry for entry in manifest["files"]}
        for future in concurrent.futures.as_completed(jobs):
            entry = jobs[future]
            try:
                state["completed"].append(future.result())
            except Exception as error:
                state["failed"].append({"file": entry["local_path"], "error_type": type(error).__name__})
            state["updated_at"] = utc_now()
            atomic_json(state_path, state)
    if state["failed"]:
        state["status"] = "failed"
        atomic_json(state_path, state)
        raise SystemExit(1)
    state.update(status="complete", completed_at=utc_now())
    atomic_json(state_path, state)
    atomic_json(args.root / "download_complete.json", {"completed_at": utc_now(), "files": len(manifest["files"]),
                "source_manifest_sha256": file_sha256(args.root / "source_manifest.json")})
    log("download_complete", files=len(manifest["files"]))
    if args.prepare:
        subprocess.run([sys.executable, str(Path(__file__).with_name("prepare.py")), "--root", str(args.root),
                        "--selection", str(args.root / "selection.json"), "--wait-selection"], check=True)


if __name__ == "__main__":
    main()
