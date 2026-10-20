#!/usr/bin/env python3
"""Build only FA in the dedicated environment, with a cgroup memory guard."""

import argparse
import difflib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import tarfile
import urllib.request
from pathlib import Path


def upstream_source():
    root = Path(__file__).resolve().parent.parent / "build_sources"
    root.mkdir(exist_ok=True)
    metadata = json.load(urllib.request.urlopen("https://pypi.org/pypi/flash-attn/2.8.3/json", timeout=30))
    release = next(x for x in metadata["urls"] if x["packagetype"] == "sdist")
    archive = root / release["filename"]
    if not archive.exists():
        with urllib.request.urlopen(release["url"], timeout=60) as incoming, archive.open("wb") as output:
            while chunk := incoming.read(1 << 20):
                output.write(chunk)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != release["digests"]["sha256"]:
        raise RuntimeError("Official source archive SHA256 mismatch")
    target = root / "flash_attn-2.8.3"
    if not (target / "setup.py").is_file():
        with tarfile.open(archive) as package:
            package.extractall(root, filter="data")
    return target, {"archive": str(archive), "archive_sha256": digest, "source_url": release["url"]}


def stop_build_session(session_id):
    # ninja/nvcc create additional process groups within this isolated session.
    # Killing only pip's process group can leave multi-gigabyte cicc orphans.
    for signum in (signal.SIGTERM, signal.SIGKILL):
        for proc in Path("/proc").iterdir():
            if not proc.name.isdigit():
                continue
            try:
                fields = (proc / "stat").read_text().split()
                if int(fields[5]) == session_id:
                    os.kill(int(proc.name), signum)
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
        if signum == signal.SIGTERM:
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", help="Existing upstream FA2.8.3 source tree; keeps completed objects")
    parser.add_argument("--guard-pid", type=int, help="Guard an already-running isolated FA pip process without restarting it")
    parser.add_argument("--qwen-restricted", action="store_true", help="Compile with upstream feature-disable switches for this Qwen model; needs --source")
    parser.add_argument("--qwen-heads-only", action="store_true", help="Explicitly support only BF16 head64/head256; retains forward/backward/split and both causal modes")
    args = parser.parse_args()
    limit_path = Path("/sys/fs/cgroup/memory.max")
    used_path = limit_path.with_name("memory.current")
    limit = int(limit_path.read_text()) if limit_path.is_file() and limit_path.read_text().strip().isdigit() else 0
    threshold = min(int(limit * 0.90), limit - (6 << 30)) if limit else 0
    if args.guard_pid:
        pid = args.guard_pid
        proc = Path(f"/proc/{pid}")
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if not any(phrase in command for phrase in ("pip install --no-deps --no-build-isolation",
                                                   "pip wheel --no-deps --no-build-isolation")) or "flash-attn_" not in command:
            raise RuntimeError("Guard target is not the expected FlashAttention build")
        if os.getpgid(pid) != pid:
            raise RuntimeError("Guard target must own an isolated process group")
        start = (proc / "stat").read_text().split()[21]
        print({"guard_pid": pid, "start_time": start, "abort_above": threshold}, flush=True)
        while proc.exists():
            try:
                if (proc / "stat").read_text().split()[21] != start:
                    return
            except FileNotFoundError:
                return
            if threshold and int(used_path.read_text()) > threshold:
                print("Cgroup RAM exceeded build guard; stopping only guarded build group", flush=True)
                stop_build_session(pid)
                raise SystemExit(1)
            time.sleep(2)
        return
    if args.qwen_restricted:
        clean_source, archive_report = upstream_source()
        source = Path(args.source) if args.source else clean_source
        args.source = str(source)
        setup = source / "setup.py"
        pristine_setup = clean_source / "setup.py.sharedexpert-original"
        if not pristine_setup.exists():
            pristine_setup.write_text((clean_source / "setup.py").read_text())
        original = pristine_setup.read_text()
        flags = ["FLASHATTENTION_DISABLE_" + feature for feature in
                 ("DROPOUT", "ALIBI", "SOFTCAP", "UNEVEN_K", "LOCAL")]
        modified = original
        for flag in flags:
            modified = modified.replace('# "-D' + flag + '",', '"-D' + flag + '",')
        # The C++ API contains explicit rejection checks for unavailable features.
        # Apply the same flags there so unsupported calls fail instead of silently
        # ignoring dropout/local windows/etc. Backward kernels remain enabled.
        cxx_flags = "[" + ", ".join(repr("-D" + flag) for flag in flags) + "]"
        marker = 'compiler_c17_flag=["-O3", "-std=c++17"]'
        if marker + " + " not in modified:
            modified = modified.replace(marker, marker + " + " + cxx_flags)
        patches = []
        if args.qwen_heads_only:
            import re
            modified = "\n".join(line for line in modified.split("\n") if
                not ("csrc/flash_attn/src/flash_" in line and ".cu\"" in line and
                     not re.search(r"_hdim(?:64|256)_bf16(?:_causal)?_sm80.cu", line)))
            header_rel = "csrc/flash_attn/src/static_switch.h"
            pristine_header = clean_source / (header_rel + ".sharedexpert-original")
            if not pristine_header.exists():
                pristine_header.write_text((clean_source / header_rel).read_text())
            original_header = pristine_header.read_text()
            restricted_header = original_header[:original_header.index("#define FP16_SWITCH")]
            restricted_header += '''// SharedExpert Qwen-only dispatch: unsupported inputs fail explicitly.
#define FP16_SWITCH(COND, ...) \\
  [&] { \\
    TORCH_CHECK(!(COND), "This FlashAttention build supports BF16 only"); \\
    using elem_type = cutlass::bfloat16_t; \\
    return __VA_ARGS__(); \\
  }()

#define HEADDIM_SWITCH(HEADDIM, ...) \\
  [&] { \\
    TORCH_CHECK((HEADDIM) == 64 || (HEADDIM) == 256, "This FlashAttention build supports head dimensions 64 and 256 only"); \\
    if ((HEADDIM) == 64) { \\
      constexpr static int kHeadDim = 64; \\
      return __VA_ARGS__(); \\
    } else { \\
      constexpr static int kHeadDim = 256; \\
      return __VA_ARGS__(); \\
    } \\
  }()
'''
            (source / header_rel).write_text(restricted_header)
            patches.extend(difflib.unified_diff(original_header.splitlines(True), restricted_header.splitlines(True),
                                               fromfile="a/" + header_rel, tofile="b/" + header_rel))
        if modified == original and not all('"-D' + flag + '",' in original for flag in flags):
            raise RuntimeError("Could not apply upstream feature switches")
        setup.write_text(modified)
        report = {**archive_report, "source": str(source.resolve()), "version": "2.8.3", "feature_disable_macros": flags,
                  "backward_enabled": True, "cuda_archs": "80", "max_jobs": 2,
                  "qwen_heads_only": args.qwen_heads_only,
                  "setup_original_sha256": hashlib.sha256(original.encode()).hexdigest(),
                  "setup_build_sha256": hashlib.sha256(modified.encode()).hexdigest(),
                  "allowed_model": "Qwen3.5-4B: attention head dimensions 64/256, dropout 0, no ALiBi/softcap/local attention"}
        logs = Path(__file__).resolve().parent.parent / "logs"
        patches.extend(difflib.unified_diff(original.splitlines(True), modified.splitlines(True),
                                           fromfile="a/setup.py", tofile="b/setup.py"))
        patch = "".join(patches)
        (logs / "flash-attention-source.patch").write_text(patch)
        report["patch_sha256"] = hashlib.sha256(patch.encode()).hexdigest()
        (logs / "flash-attention-build.json").write_text(json.dumps(report, indent=2))
    env = dict(os.environ, PATH=str(Path(sys.executable).parent) + ":" + os.environ.get("PATH", ""),
               MAX_JOBS="2", NVCC_THREADS="2", FLASH_ATTN_CUDA_ARCHS="80",
               FLASH_ATTENTION_FORCE_BUILD="TRUE", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    wheels = Path(__file__).resolve().parent.parent / "build_sources/wheels"
    wheels.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
               "--wheel-dir", str(wheels), args.source or "flash-attn==2.8.3"]
    if threshold and int(used_path.read_text()) > threshold - (12 << 30):
        raise RuntimeError("Insufficient cgroup RAM headroom for two FA compiler jobs")
    print({"command": command, "max_jobs": 2, "nvcc_threads": 2, "cuda_archs": "80",
           "cgroup_limit": limit, "abort_above": threshold}, flush=True)
    process = subprocess.Popen(command, env=env, start_new_session=True)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        while process.poll() is None:
            if threshold and int(used_path.read_text()) > threshold:
                raise RuntimeError("Cgroup RAM exceeded build guard; stopping only this build process group")
            time.sleep(2)
        if process.returncode:
            raise SystemExit(process.returncode)
        wheel = max(wheels.glob("flash_attn-2.8.3-*.whl"), key=lambda path: path.stat().st_mtime)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", str(wheel)], check=True, env=env)
        report_path = Path(__file__).resolve().parent.parent / "logs/flash-attention-build.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        report.update(wheel=str(wheel), wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest())
        report_path.write_text(json.dumps(report, indent=2))
    finally:
        if process.poll() is None:
            # nvcc's cicc descendants can ignore TERM; KILL the isolated build
            # group after a short grace period so no compiler orphan survives.
            stop_build_session(process.pid)
            process.wait()


if __name__ == "__main__":
    main()
