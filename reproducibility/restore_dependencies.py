#!/usr/bin/env python3
"""Verify pinned dependency checkouts and apply only the recorded project patches.

Run `git submodule update --init --recursive` first. Existing unrelated changes
are never reset. --check performs only read operations, including on live runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args, check=True):
    return subprocess.run(
        ["git", "-C", str(path), *args],
        check=check, capture_output=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify only; never apply patches")
    args = parser.parse_args()
    dependencies = json.loads((ROOT / "reproducibility/dependencies.json").read_text())["dependencies"]
    for entry in dependencies:
        path = ROOT / entry["path"]
        patch = ROOT / entry["patch"]
        if hashlib.sha256(patch.read_bytes()).hexdigest() != entry["patch_sha256"]:
            raise RuntimeError(f"Patch hash changed: {entry['patch']}")
        actual = git(path, "rev-parse", "HEAD").stdout.decode().strip()
        if actual != entry["commit"]:
            raise RuntimeError(f"{entry['name']}: checkout differs from pinned commit")
        diff = git(path, "diff", "--binary", "HEAD", "--").stdout
        if diff == patch.read_bytes():
            print(f"{entry['name']}: exact pinned source and patch verified")
            continue
        if diff:
            raise RuntimeError(f"{entry['name']}: unrelated/different local changes; refusing to overwrite")
        if args.check:
            raise RuntimeError(f"{entry['name']}: patch not applied; run this script without --check")
        git(path, "apply", "--check", str(patch))
        git(path, "apply", str(patch))
        if git(path, "diff", "--binary", "HEAD", "--").stdout != patch.read_bytes():
            raise RuntimeError(f"{entry['name']}: post-apply verification failed")
        print(f"{entry['name']}: recorded patch applied and verified")


if __name__ == "__main__":
    main()
