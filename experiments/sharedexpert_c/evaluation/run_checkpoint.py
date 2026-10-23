#!/usr/bin/env python3
"""Start an owned local vLLM server for an exported HF checkpoint, evaluate, clean up."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

from common import HERE, ROOT, PROTOCOL, digest, file_sha, load_suite, read_jsonl, local_lock_path
from model_names import MODEL_NAMES
from run_api import request_json


def validate_hf_export_metadata(model):
    """Load local config/processor metadata without allocating model weights.

    Transformers 5 saves the image/video processor components inside
    processor_config.json; older exports use preprocessor_config.json.
    AutoProcessor resolves either layout, including the saved tokenizer/template.
    """
    model = Path(model)
    for required in ("config.json", "tokenizer_config.json"):
        if not (model / required).is_file():
            raise ValueError(f"Not a complete HF checkpoint: missing {model / required}")
    processor_files = [name for name in ("processor_config.json", "preprocessor_config.json")
                       if (model / name).is_file()]
    if not processor_files:
        raise ValueError(f"Not a complete HF checkpoint: missing processor_config.json or preprocessor_config.json in {model}")
    from transformers import AutoConfig, AutoProcessor
    config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if getattr(processor, "image_processor", None) is None or getattr(processor, "tokenizer", None) is None:
        raise ValueError(f"Export does not load a multimodal image processor and tokenizer: {model}")
    return {"model_type": config.model_type, "architectures": getattr(config, "architectures", None),
            "processor_class": type(processor).__name__, "processor_metadata": processor_files}


def completed_cache(out, rows, suite_meta, provenance, name, served_name):
    """Only an exact, successful full-suite cache can avoid starting a server."""
    required = [out / filename for filename in ("checkpoint.json", "run.json", "predictions.jsonl")]
    if not all(path.is_file() for path in required):
        return False
    if json.loads(required[0].read_text()) != provenance:
        raise SystemExit("Output belongs to another checkpoint; use a different --output-root")
    info = json.loads(required[1].read_text())
    expected = {"suite_sha256": suite_meta["suite_sha256"],
                "protocol_sha256": suite_meta["protocol_sha256"], "protocol": PROTOCOL,
                "name": name, "served_model": served_name, "n_requested": len(rows), "smoke": False}
    if any(info.get(key) != value for key, value in expected.items()):
        return False
    records = {record["id"]: record for record in read_jsonl(required[2])}
    return all(row["id"] in records and not records[row["id"]].get("error") and
               "raw_response" in records[row["id"]] and
               records[row["id"]].get("row_sha256") == digest(row) for row in rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True, help="Merged, complete HF model directory")
    ap.add_argument("--name", required=True, choices=MODEL_NAMES)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--suite", type=Path, default=HERE / "suite.jsonl")
    ap.add_argument("--output-root", type=Path, default=HERE / "results")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="Smoke only: evaluate N representative samples into smoke/<name>")
    ap.add_argument("--server-config", type=Path, default=HERE / "local_server.json",
                    help="Environment-only vLLM knobs (GPU count/memory differ from the original host)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    # The original host had the evaluation virtualenv under pre-lab/venv. On
    # the new host the evaluation environment lives in .venv-eval, so use the
    # interpreter that launched this script instead of a hard-coded path.
    py = Path(sys.executable)
    server_config = {"gpu_memory_utilization": 0.88, "max_num_seqs": 16, "dtype": "bfloat16",
                     "env": {"VLLM_USE_FLASHINFER_SAMPLER": "0", "OMP_NUM_THREADS": "8"}}
    if args.server_config.is_file():
        server_config.update(json.loads(args.server_config.read_text()))
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    served_name = f"sharedexpert-{args.name}"
    server_cmd = [str(py), "-m", "vllm.entrypoints.openai.api_server", "--model", str(args.model.resolve()), "--served-model-name", served_name,
                  "--host", "127.0.0.1", "--port", str(port), "--max-model-len", str(PROTOCOL["max_model_len"]),
                  "--gpu-memory-utilization", str(server_config["gpu_memory_utilization"]),
                  "--max-num-seqs", str(server_config["max_num_seqs"]), "--dtype", str(server_config["dtype"]),
                  "--limit-mm-per-prompt", json.dumps({"image": PROTOCOL["max_images"]}),
                  "--mm-processor-kwargs", json.dumps({"size": PROTOCOL["image_size"]})]
    eval_cmd = [str(py), str(HERE / "run_api.py"), "--suite", str(args.suite), "--base-url", f"http://127.0.0.1:{port}/v1",
                "--model", served_name, "--name", args.name, "--output-root", str(args.output_root), "--concurrency", str(args.concurrency)]
    if args.limit:
        eval_cmd += ["--limit", str(args.limit)]
    if args.dry_run:
        print(json.dumps({"CUDA_VISIBLE_DEVICES": str(args.gpu), "server": server_cmd, "evaluation": eval_cmd}, indent=2))
        return
    rows, suite_meta = load_suite(args.suite)
    metadata = validate_hf_export_metadata(args.model)
    print(json.dumps({"export_metadata_loaded": metadata}), flush=True)
    if not list(args.model.glob("*.safetensors")):
        raise SystemExit("No exported safetensors; merge checkpoint first")
    out = args.output_root / ("smoke" if args.limit else "full") / args.name
    provenance = {"path": str(args.model.resolve()), "config_sha256": file_sha(args.model / "config.json"),
                  "weight_files": [{"name": p.name, "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns} for p in sorted(args.model.glob("*.safetensors"))]}
    if not args.limit and completed_cache(out, rows, suite_meta, provenance, args.name, served_name):
        # Recompute current scoring from the raw generations without requiring a
        # free GPU; the user may already have restored their original services.
        subprocess.run([str(py), str(HERE / "summarize.py"), "--suite", str(args.suite),
                        "--output-root", str(args.output_root)], check=True)
        print(f"{args.name}: exact checkpoint and complete evaluation cache verified; no inference needed", flush=True)
        return
    lock_path = local_lock_path(HERE, f'gpu-{args.gpu}')
    with lock_path.open("w") as gpu_lock:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_uuid = subprocess.check_output(["nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
        processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True)
        if any(line.startswith(gpu_uuid) for line in processes.splitlines()):
            raise SystemExit(f"GPU {args.gpu} already has a compute process; refusing to interfere")
        out.mkdir(parents=True, exist_ok=True)
        provenance_path = out / "checkpoint.json"
        if provenance_path.exists() and json.loads(provenance_path.read_text()) != provenance:
            raise SystemExit("Output belongs to another checkpoint; use a different --output-root")
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
                   **{str(k): str(v) for k, v in server_config.get("env", {}).items()})
        bundled_cuda = ROOT / "pre-lab/venv/lib/python3.10/site-packages/nvidia/cu13"
        if (bundled_cuda / "bin/nvcc").is_file():
            env["CUDA_HOME"] = str(bundled_cuda)
            env["PATH"] = str(bundled_cuda / "bin") + os.pathsep + env.get("PATH", "")
        with (out / "vllm.log").open("a") as log:
            server = subprocess.Popen(server_cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            def interrupted(signum, frame):
                # Default SIGTERM bypasses finally; turn it into a controlled
                # exit so our independently-sessioned vLLM is always reclaimed.
                raise SystemExit(128 + signum)
            previous_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
            try:
                deadline = time.monotonic() + 1200
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        raise RuntimeError(f"vLLM exited {server.returncode}; see {out / 'vllm.log'}")
                    try:
                        models = request_json(f"http://127.0.0.1:{port}/v1/models", timeout=5)
                        if served_name in {m["id"] for m in models["data"]}:
                            break
                    except Exception:
                        time.sleep(5)
                else:
                    raise TimeoutError("vLLM did not become ready within 20 minutes")
                subprocess.run(eval_cmd, check=True, env=env)
            finally:
                # Signal only the process group created by this launcher.
                try:
                    os.killpg(server.pid, signal.SIGTERM)
                    server.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait(timeout=10)
                except ProcessLookupError:
                    pass
                finally:
                    for sig, previous in previous_handlers.items():
                        signal.signal(sig, previous)


if __name__ == "__main__":
    main()
