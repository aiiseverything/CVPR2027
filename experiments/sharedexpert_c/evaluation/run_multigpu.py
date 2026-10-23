#!/usr/bin/env python3
"""Start one owned local vLLM server per GPU, rollout the frozen suite in parallel, score on CPU."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import time
from contextlib import ExitStack
from pathlib import Path

from common import HERE, ROOT, PROTOCOL, digest, file_sha, load_suite, local_lock_path, read_jsonl
from run_api import request_json
from run_checkpoint import completed_cache, validate_hf_export_metadata


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="Merged, complete HF model directory")
    ap.add_argument("--name", required=True, choices=["baseline", "baseline-9b", "generalist", "perception", "reasoning"])
    ap.add_argument("--gpus", type=int, nargs="+", required=True)
    ap.add_argument("--suite", type=Path, default=HERE / "suite.jsonl")
    ap.add_argument("--output-root", type=Path, default=HERE / "results")
    ap.add_argument("--concurrency", type=int, default=0, help="Total client concurrency; default 12 per GPU")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        raise SystemExit("GPU list contains duplicates")
    concurrency = args.concurrency or 12 * len(args.gpus)
    py = ROOT / "pre-lab/venv/bin/python"
    served_name = f"sharedexpert-{args.name}"
    plan = []
    for gpu in args.gpus:
        port = free_port()
        plan.append({
            "gpu": gpu,
            "port": port,
            "server": [str(py), "-m", "vllm.entrypoints.openai.api_server", "--model", str(args.model.resolve()),
                       "--served-model-name", served_name, "--host", "127.0.0.1", "--port", str(port),
                       "--max-model-len", str(PROTOCOL["max_model_len"]), "--gpu-memory-utilization", "0.88",
                       "--max-num-seqs", "16", "--dtype", "bfloat16",
                       "--limit-mm-per-prompt", json.dumps({"image": PROTOCOL["max_images"]}),
                       "--mm-processor-kwargs", json.dumps({"size": PROTOCOL["image_size"]})],
        })
    eval_cmd = [str(py), str(HERE / "run_api.py"), "--suite", str(args.suite),
                "--base-urls", *[f"http://127.0.0.1:{entry['port']}/v1" for entry in plan],
                "--model", served_name, "--name", args.name, "--output-root", str(args.output_root),
                "--concurrency", str(concurrency)]
    if args.dry_run:
        print(json.dumps({"gpus": args.gpus, "concurrency": concurrency, "servers": plan,
                          "evaluation": eval_cmd}, indent=2))
        return
    rows, suite_meta = load_suite(args.suite)
    metadata = validate_hf_export_metadata(args.model)
    print(json.dumps({"export_metadata_loaded": metadata, "servers": len(plan)}), flush=True)
    if not list(args.model.glob("*.safetensors")):
        raise SystemExit("No exported safetensors; merge checkpoint first")
    out = args.output_root / "full" / args.name
    provenance = {"path": str(args.model.resolve()), "config_sha256": file_sha(args.model / "config.json"),
                  "weight_files": [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                                   for p in sorted(args.model.glob("*.safetensors"))]}
    if completed_cache(out, rows, suite_meta, provenance, args.name, served_name):
        subprocess.run([str(py), str(HERE / "summarize.py"), "--suite", str(args.suite),
                        "--output-root", str(args.output_root)], check=True)
        print(f"{args.name}: exact checkpoint and complete evaluation cache verified; no inference needed", flush=True)
        return
    with ExitStack() as stack:
        # Lock every GPU before starting any server; sorted order avoids deadlock.
        for gpu in sorted(args.gpus):
            lock = stack.enter_context(local_lock_path(HERE, f"gpu-{gpu}").open("w"))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise SystemExit(f"GPU {gpu} is locked by another evaluation launcher")
        for gpu in args.gpus:
            gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                                                "--format=csv,noheader"], text=True).strip()
            processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                                 "--format=csv,noheader"], text=True)
            if any(line.startswith(gpu_uuid) for line in processes.splitlines()):
                raise SystemExit(f"GPU {gpu} already has a compute process; refusing to interfere")
        out.mkdir(parents=True, exist_ok=True)
        provenance_path = out / "checkpoint.json"
        if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
            raise SystemExit("Output belongs to another checkpoint; use a different --output-root")
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        env_base = dict(os.environ, VLLM_USE_FLASHINFER_SAMPLER="0", OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
        bundled_cuda = ROOT / "pre-lab/venv/lib/python3.10/site-packages/nvidia/cu13"
        if (bundled_cuda / "bin/nvcc").is_file():
            env_base["CUDA_HOME"] = str(bundled_cuda)
            env_base["PATH"] = str(bundled_cuda / "bin") + os.pathsep + env_base.get("PATH", "")
        servers, logs = [], []

        def interrupted(signum, frame):
            # Default SIGTERM bypasses finally; turn it into a controlled exit so
            # every independently-sessioned vLLM process group is reclaimed.
            raise SystemExit(128 + signum)

        previous_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
        try:
            deadline = time.monotonic() + 3600

            def wait_ready(port, gpu, proc):
                while time.monotonic() < deadline:
                    for other_gpu, _, other in servers:
                        if other.poll() is not None:
                            raise RuntimeError(f"vLLM on GPU {other_gpu} exited {other.returncode}; "
                                               f"see {out / f'vllm-gpu{other_gpu}.log'}")
                    try:
                        models = request_json(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                        if served_name in {m["id"] for m in models["data"]}:
                            return
                    except Exception:
                        pass
                    time.sleep(5)
                raise TimeoutError(f"vLLM on GPU {gpu} did not become ready within 60 minutes")

            # Sequential startup keeps concurrent model-loading peaks inside the
            # container's 64 GiB memory cgroup; inference then runs on all GPUs.
            for entry in plan:
                gpu, port = entry["gpu"], entry["port"]
                log = (out / f"vllm-gpu{gpu}.log").open("a")
                logs.append(log)
                proc = subprocess.Popen(entry["server"], env=dict(env_base, CUDA_VISIBLE_DEVICES=str(gpu)),
                                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                servers.append((gpu, port, proc))
                print(f"starting vLLM on GPU {gpu} (port {port})", flush=True)
                wait_ready(port, gpu, proc)
                print(f"vLLM on GPU {gpu} ready", flush=True)
            before = len(read_jsonl(out / "predictions.jsonl"))
            print(f"{args.name}: {len(rows)} samples, {before} cached, {len(servers)} endpoints, "
                  f"client concurrency={concurrency}; scoring runs in this CPU client", flush=True)
            subprocess.run(eval_cmd, check=True, env=env_base)
        finally:
            for gpu, _, proc in servers:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait(timeout=10)
                except ProcessLookupError:
                    pass
            for log in logs:
                log.close()
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)


if __name__ == "__main__":
    main()
