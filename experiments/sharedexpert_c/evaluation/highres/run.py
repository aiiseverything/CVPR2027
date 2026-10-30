#!/usr/bin/env python3
"""Resume one complete high-resolution evaluation using an owned local server."""
from __future__ import annotations

import argparse
import base64
from collections import Counter, deque
import concurrent.futures
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import threading
import urllib.error
import urllib.request

from protocol import (DATA, EVALUATION, HERE, MODEL_PATHS, PROTOCOL, RESULTS, REGISTERED_MODEL_NAMES,
                      atomic_json, content_for, digest, file_sha, load_suite,
                      prompt_tokens)
from scheduling import configuration as scheduling_configuration, can_admit, row_cost
import throughput

_IMAGE_ENCODERS = threading.BoundedSemaphore(16)

sys.path.insert(0, str(EVALUATION))
from common import local_lock_path  # Only reuse the host-local locking namespace.

AUTHORIZED_MODELS = ("baseline", "reasoning-step100", "perception-step100", "perception-v2-step100")
BENCHMARKS = {"vstar", "zoom", "hrbench_4k", "hrbench_8k", "mme_rw_en", "mme_rw_cn"}


def selection_snapshot(root, require_ready=True):
    """Read one atomic selection snapshot; unfinished manifests are never run."""
    path = Path(root) / "selection.json"
    if not path.is_file():
        if require_ready:
            raise ValueError(f"Missing perception selection manifest: {path}")
        return None
    payload = path.read_bytes()
    selection = json.loads(payload)
    if selection.get("ready") is not True:
        if require_ready:
            raise ValueError("Perception selection is not ready")
        return None
    if selection.get("schema_version") != "highres-selection-v1":
        raise ValueError("Unsupported perception selection schema")
    benchmarks = selection.get("benchmarks")
    if not isinstance(benchmarks, dict) or set(benchmarks) != BENCHMARKS:
        raise ValueError("Selection must explicitly define all six benchmarks")
    counts = {}
    for name, item in benchmarks.items():
        count = item.get("expected_count")
        if type(count) is not int or count < 1:
            raise ValueError(f"{name}: expected_count must be a positive integer")
        counts[name] = count
        filters = item.get("filters", [])
        if not isinstance(filters, list):
            raise ValueError(f"{name}: filters must be an explicit AND list")
        if not filters and "include_source_ids" not in item:
            raise ValueError(f"{name}: explicit perception filters or source ID inclusion are required")
        for rule in filters:
            if (not isinstance(rule, dict) or not isinstance(rule.get("field"), str)
                    or not rule["field"] or rule.get("op") not in {"eq", "startswith", "in"}
                    or "value" not in rule):
                raise ValueError(f"{name}: invalid selection filter")
            if rule["op"] == "in" and not isinstance(rule["value"], list):
                raise ValueError(f"{name}: 'in' filter requires a list")
            if rule["op"] == "startswith" and not isinstance(rule["value"], str):
                raise ValueError(f"{name}: 'startswith' filter requires a string")
        for key in ("include_source_ids", "exclude_source_ids"):
            if key in item and (not isinstance(item[key], list)
                                or any(not isinstance(x, str) for x in item[key])
                                or len(item[key]) != len(set(item[key]))):
                raise ValueError(f"{name}: {key} must contain unique original source ID strings")
    return {"sha256": hashlib.sha256(payload).hexdigest(), "expected_counts": counts,
            "n_requested": sum(counts.values()), "path": str(path.resolve())}


def apply_subset(rows, selection, path):
    """Narrow frozen rows to an explicit, hash-recorded subset manifest."""
    path = Path(path)
    payload = path.read_bytes()
    manifest = json.loads(payload)
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != "highres-subset-v1"
            or manifest.get("ready") is not True):
        raise ValueError("Unsupported or unready high-resolution subset manifest")
    ids = manifest.get("ids")
    if (not isinstance(ids, list) or not ids
            or any(not isinstance(value, str) or not value for value in ids)
            or len(ids) != len(set(ids))):
        raise ValueError("Subset manifest must list unique non-empty row IDs")
    missing = sorted(set(ids) - {row["id"] for row in rows})
    if missing:
        raise ValueError(f"Subset IDs are absent from the frozen suite or benchmark filter: {missing[:5]}")
    wanted = set(ids)
    rows = [row for row in rows if row["id"] in wanted]
    counts = dict(Counter(row["benchmark"] for row in rows))
    subset = {"name": manifest.get("name"), "path": str(path.resolve()),
              "sha256": hashlib.sha256(payload).hexdigest(), "n_ids": len(ids),
              "counts": counts}
    selection = {**selection,
                 "source_expected_counts": selection.get("source_expected_counts", selection["expected_counts"]),
                 "expected_counts": counts, "n_requested": len(rows), "subset": subset}
    return rows, selection, subset


def offline_environment():
    """Inference/preparation use only local files and never need a laptop proxy."""
    env = {key: value for key, value in os.environ.items()
           if key.lower() not in {"http_proxy", "https_proxy", "all_proxy", "no_proxy"}}
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1",
               HF_HUB_DISABLE_TELEMETRY="1", VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
               NO_PROXY="127.0.0.1,localhost,::1", PYTHONUNBUFFERED="1")
    return env


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def request_json(url, payload=None, timeout=60):
    request = urllib.request.Request(url,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    # A local inference request must not pass through HTTP_PROXY/ALL_PROXY.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=timeout) as response:
        return json.load(response)


def gpu_status(gpu):
    """Read resource ownership without allocating CUDA or touching another job."""
    def query(arguments):
        return subprocess.check_output(["nvidia-smi", *arguments], text=True, timeout=30).strip()
    fields = query(["-i", str(gpu), "--query-gpu=uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
    uuid, memory, utilization = [x.strip() for x in fields.split(",")]
    apps = query(["--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
    pids = [int(line.split(",")[1].strip()) for line in apps.splitlines()
            if line.split(",")[0].strip() == uuid]
    memory = int(memory)
    warning = None
    if utilization not in {"0", "[N/A]", "N/A"} and not pids and memory <= 256:
        warning = "Nonzero GPU utilization without compute PIDs or substantial VRAM; attempting the authorized formal server start."
    return {"gpu": gpu, "uuid": uuid, "memory_used_mib": memory,
            "utilization_percent": utilization, "compute_pids": pids,
            "busy": bool(pids) or memory > 256, "warning": warning, "checked_at": now()}


def model_provenance(model):
    required = ["config.json", "tokenizer_config.json"]
    for name in required:
        if not (model / name).is_file():
            raise ValueError(f"Missing HF metadata: {model / name}")
    if not any((model / name).is_file() for name in ("processor_config.json", "preprocessor_config.json")):
        raise ValueError(f"Missing image processor metadata: {model}")
    weights = sorted(model.glob("*.safetensors"))
    if not weights or any(p.stat().st_size == 0 for p in weights):
        raise ValueError(f"Missing or empty exported weights: {model}")
    index = model / "model.safetensors.index.json"
    if index.exists():
        referenced = set(json.loads(index.read_text())["weight_map"].values())
        if not referenced <= {p.name for p in weights}:
            raise ValueError("Checkpoint weight index references missing shards")
    metadata = {p.name: file_sha(p) for p in sorted(model.iterdir())
                if p.is_file() and p.suffix in {".json", ".jinja", ".txt", ".model"}}
    return {"path": str(model.resolve()), "metadata_sha256": metadata,
            "weight_files": [{"name": p.name, "size": p.stat().st_size,
                              "mtime_ns": p.stat().st_mtime_ns} for p in weights],
            "weight_identity_policy": "file names, byte sizes and modification nanoseconds; no claim of full weight-content hashing"}


def validate_model_cpu(model, row):
    from transformers import AutoConfig, AutoProcessor
    config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    if getattr(processor, "image_processor", None) is None:
        raise ValueError("Checkpoint does not load an image processor")
    actual, visual = prompt_tokens(processor, row)
    if actual != row["prompt_tokens"]:
        raise ValueError(f"CPU checkpoint prompt tokens differ: {actual} != {row['prompt_tokens']}")
    return {"model_type": config.model_type, "processor_class": type(processor).__name__,
            "validated_row_id": row["id"], "prompt_tokens": actual, "visual_tokens": visual}


def resolve_model(name, explicit_model, meta, rows):
    """Require a scoped CPU registration for an explicitly requested new export."""
    if name in AUTHORIZED_MODELS:
        model = MODEL_PATHS[name].resolve()
        if explicit_model is not None and Path(explicit_model).resolve() != model:
            raise ValueError("Explicit model differs from the historical model path")
        return model
    if name not in REGISTERED_MODEL_NAMES or explicit_model is None:
        raise ValueError("A new model requires an authorized name and explicit --model export")
    registration = meta.get("registered_models", {}).get(name, {})
    if (registration.get("schema_version") != "highres-model-registration-v1"
            or registration.get("validated_ids_sha256") != digest(sorted(row["id"] for row in rows))
            or registration.get("n_validated") != len(rows)
            or registration.get("model_sha256") != digest(meta.get("models", {}).get(name))):
        raise ValueError("New model lacks a matching CPU registration for this exact frozen subset")
    model = Path(explicit_model).resolve()
    if Path(meta["models"][name]["path"]).resolve() != model:
        raise ValueError("Registered export path differs from --model")
    return model


def recover_predictions(path, rows, name):
    """Reject stale cache; recover only an interrupted, unterminated final line."""
    previous = {}
    if not path.exists():
        return previous
    identities = {r["id"]: digest(r) for r in rows}
    token_counts = {r["id"]: r["prompt_tokens"] for r in rows}
    with path.open("r+b") as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError):
                if line.endswith(b"\n") or stream.read(1):
                    raise ValueError(f"Corrupt prediction line at byte {start}") from None
                evidence = path.with_name(path.name + f".interrupted-tail-{time.time_ns()}.bin")
                evidence.write_bytes(line)
                stream.seek(start)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                break
            if (record.get("id") not in identities or record.get("model") != name
                    or record.get("row_sha256") != identities[record["id"]]):
                raise ValueError("Cached prediction belongs to a different row or model")
            if not record.get("error") and "raw_response" not in record:
                raise ValueError("Successful cached prediction lacks raw_response")
            if not record.get("error") and record.get("usage", {}).get("prompt_tokens") != token_counts[record["id"]]:
                raise ValueError("Cached successful prediction lacks the matching frozen prompt token usage")
            previous[record["id"]] = record
            if not line.endswith(b"\n"):
                stream.write(b"\n")
                stream.flush()
                os.fsync(stream.fileno())
                break
    return previous


def image_content(root, row):
    expected = row["image"]["sha256"]
    with sqlite3.connect((root / "images.sqlite").resolve().as_uri() + "?mode=ro", uri=True) as db:
        result = db.execute("SELECT data FROM images WHERE sha256 = ?", (expected,)).fetchone()
    if result is None:
        raise ValueError(f"Missing image blob for {row['id']}")
    blob = result[0]
    if hashlib.sha256(blob).hexdigest() != expected:
        raise ValueError(f"Image blob SHA256 mismatch for {row['id']}")
    mime = row["image"].get("mime") or row["image"].get("mime_type") or "image/jpeg"
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{base64.b64encode(blob).decode()}"}}


def infer(row, args, base_url, served_name):
    from scoring import score_prediction
    started = time.monotonic()
    record = {"id": row["id"], "benchmark": row["benchmark"], "model": args.name,
              "row_sha256": digest(row), "error": None}
    try:
        # Bound temporary SQLite/base64 copies, not the HTTP request lifetime.
        with _IMAGE_ENCODERS:
            image = image_content(args.root, row)
        payload = {"model": served_name,
            "messages": [{"role": "user", "content": content_for(row["prompt"], image)}],
            "temperature": PROTOCOL["temperature"], "top_p": PROTOCOL["top_p"],
            "max_tokens": PROTOCOL["max_tokens"], "seed": PROTOCOL["seed"],
            "chat_template_kwargs": {"enable_thinking": PROTOCOL["enable_thinking"]},
            "mm_processor_kwargs": {"size": PROTOCOL["image_size"]}}
        for attempt in range(args.retries):
            record["attempts"] = attempt + 1
            try:
                result = request_json(base_url + "/chat/completions", payload, timeout=args.timeout)
                choice = result["choices"][0]
                message = choice["message"]
                record.update(raw_response=message.get("content") or "",
                    reasoning=message.get("reasoning_content") or message.get("reasoning"),
                    finish_reason=choice.get("finish_reason"), usage=result.get("usage") or {}, error=None)
                actual = record["usage"].get("prompt_tokens")
                if actual != row["prompt_tokens"] or actual > PROTOCOL["max_prompt_tokens"]:
                    raise ValueError(f"Prompt-token mismatch: server={actual}, frozen={row['prompt_tokens']}; protocol unchanged")
                record["elapsed_seconds"] = time.monotonic() - started
                return score_prediction(row, record)
            except urllib.error.HTTPError as error:
                detail = error.read(4096).decode(errors="replace")
                record["error"] = f"HTTP {error.code}: {detail[:500]}"
                if 400 <= error.code < 500 and error.code != 429:
                    break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, KeyError, json.JSONDecodeError) as error:
                record["error"] = f"{type(error).__name__}: {str(error)[:500]}"
            if attempt + 1 < args.retries:
                time.sleep(min(2 ** attempt, 10))
    except Exception as error:
        record.update(error=f"{type(error).__name__}: {str(error)[:1000]}", fatal_error=True)
    record["elapsed_seconds"] = time.monotonic() - started
    return record


def stop_server(server):
    if server is None:
        return
    try:
        os.killpg(server.pid, signal.SIGTERM)
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait(timeout=10)
    except ProcessLookupError:
        pass


def execute(args):
    if args.name not in AUTHORIZED_MODELS + REGISTERED_MODEL_NAMES:
        raise ValueError("Unknown high-resolution evaluation model")
    adaptive = getattr(args, "adaptive_concurrency", False)
    high_throughput = getattr(args, "throughput_mode", False)
    profile_index = getattr(args, "execution_profile", 0)
    if adaptive and high_throughput:
        raise ValueError("Select one scheduling mode")
    scheduling_policy = (throughput.configuration() if high_throughput else scheduling_configuration() if adaptive else
                         {"version": "fixed-concurrency-v1", "max_inflight": args.concurrency})
    if adaptive and args.concurrency != scheduling_policy["max_inflight"]:
        raise ValueError("Adaptive scheduling requires --concurrency 16")
    if high_throughput and args.concurrency != scheduling_policy["max_inflight"]:
        raise ValueError("Throughput mode requires --concurrency 64")
    if not high_throughput and profile_index:
        raise ValueError("Execution fallback profiles apply only to throughput mode")
    server_profile = (scheduling_policy["profiles"][profile_index] if high_throughput else
                      {"max_num_seqs": args.concurrency, "max_num_batched_tokens": 8192,
                       "gpu_memory_utilization": 0.88})
    selection = selection_snapshot(args.root)
    rows, meta = load_suite(args.root)
    if (meta.get("selection_sha256") != selection["sha256"]
            or dict(Counter(r["benchmark"] for r in rows)) != selection["expected_counts"]):
        raise ValueError("Frozen suite does not match the complete perception selection manifest")
    benchmarks = getattr(args, "benchmarks", None)
    if benchmarks is not None:
        benchmarks = sorted(set(benchmarks))
        if not benchmarks or not set(benchmarks) <= BENCHMARKS:
            raise ValueError("Unknown or empty benchmark subset")
        rows = [row for row in rows if row["benchmark"] in benchmarks]
        selection = {**selection, "source_expected_counts": selection["expected_counts"],
                     "expected_counts": dict(Counter(r["benchmark"] for r in rows)),
                     "n_requested": len(rows), "benchmarks": benchmarks}
    subset_info = None
    subset_path = getattr(args, "ids", None)
    if subset_path is not None:
        rows, selection, subset_info = apply_subset(rows, selection, subset_path)
        if benchmarks is not None:
            selection["benchmarks"] = benchmarks
    if any(not isinstance(r.get("prompt_tokens"), int) or not 0 < r["prompt_tokens"] <= PROTOCOL["max_prompt_tokens"] for r in rows):
        raise ValueError("Suite must have frozen, CPU-validated prompt token counts")
    if adaptive:
        for row in rows:
            row_cost(row)
    if high_throughput:
        for row in rows:
            throughput.validate_row(row)
    model = resolve_model(args.name, getattr(args, "model", None), meta, rows)
    provenance = model_provenance(model)
    frozen_model = meta["models"][args.name]
    if (Path(frozen_model["path"]).resolve() != model
            or frozen_model["weight_files"] != provenance["weight_files"]
            or frozen_model["metadata_sha256"] != {
                k: v for k, v in provenance["metadata_sha256"].items()
                if Path(k).suffix in {".json", ".jinja"}}):
        raise ValueError("Checkpoint metadata/weights changed after CPU suite freezing")
    for filename, key in (("source_manifest.json", "source_manifest_sha256"),
                          ("dataset_manifest.json", "dataset_manifest_sha256"),
                          ("questions.jsonl", "questions_sha256")):
        if file_sha(args.root / filename) != meta[key]:
            raise ValueError(f"Frozen source artifact changed: {filename}")
    out = args.output_root / "full" / args.name
    out.mkdir(parents=True, exist_ok=True)
    identity = {"name": args.name, "protocol": PROTOCOL,
                "protocol_sha256": digest(PROTOCOL), "suite_sha256": meta["suite_sha256"],
                "source_manifest_sha256": meta.get("source_manifest_sha256"),
                "dataset_manifest_sha256": meta["dataset_manifest_sha256"],
                "selection_sha256": selection["sha256"],
                "expected_counts": selection["expected_counts"],
                "frozen_model_sha256": digest(frozen_model),
                "n_requested": len(rows), "smoke": False,
                "scheduling_policy": scheduling_policy,
                "scheduling_policy_sha256": digest(scheduling_policy)}
    if benchmarks is not None:
        identity["benchmarks"] = benchmarks
    if subset_info is not None:
        identity["subset"] = subset_info
    lock = local_lock_path(out, "inference").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for filename, value in (("checkpoint.json", provenance), ("run.json", identity)):
        path = out / filename
        if path.exists() and json.loads(path.read_text()) != value:
            raise ValueError(f"{path} does not match this checkpoint/suite/protocol; refusing cache reuse")
        atomic_json(path, value)
    path = out / "predictions.jsonl"
    previous = recover_predictions(path, rows, args.name)
    todo = [row for row in rows if row["id"] not in previous or previous[row["id"]].get("error")]
    state = {"pid": os.getpid(), "started_at": now(), "name": args.name, "stage": "cpu_validation",
             "n_requested": len(rows), "completed": len(rows) - len(todo), "server_pid": None,
             "config": {"gpu": args.gpu, "concurrency": args.concurrency, **server_profile,
                        "adaptive_concurrency": adaptive, "scheduling_policy": scheduling_policy,
                        "throughput_mode": high_throughput, "execution_profile": profile_index,
                        "enforce_eager": getattr(args, "enforce_eager", False),
                        "root": str(args.root.resolve()), "output_root": str(args.output_root.resolve()),
                        "python": sys.executable, "timeout": args.timeout, "request_attempts": args.retries}}
    def update(**values):
        state.update(values, updated_at=now())
        atomic_json(out / "status.json", state)
    update()
    if not todo:
        update(stage="complete", completed_at=now(), cache_verified=True)
        print(f"{args.name}: all {len(rows)} frozen predictions already complete", flush=True)
        return 0
    os.environ.update({key: value for key, value in offline_environment().items()
                       if key.startswith(("HF_", "TRANSFORMERS_"))})
    if args.name in REGISTERED_MODEL_NAMES:
        update(metadata={"cpu_preflight": "registered_subset_checked",
                         "registration": meta["registered_models"][args.name]}, selection=selection)
    elif meta.get("preflight_policy", {}).get("skipped"):
        update(metadata={"cpu_preflight": "skipped_at_user_request"}, selection=selection)
    else:
        update(metadata=validate_model_cpu(model, todo[0]), selection=selection)
    # Import the scorer before owning CUDA; dependency failures are CPU failures.
    from scoring import score_prediction  # noqa: F401
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    served_name = "sharedexpert-highres-" + args.name
    command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(model), "--tokenizer", str(model), "--served-model-name", served_name,
        "--host", "127.0.0.1", "--port", str(port),
        "--max-model-len", str(PROTOCOL["max_model_len"]),
        "--max-num-seqs", str(server_profile["max_num_seqs"]),
        "--gpu-memory-utilization", str(server_profile["gpu_memory_utilization"]),
        "--max-num-batched-tokens", str(server_profile["max_num_batched_tokens"]), "--enable-chunked-prefill",
        "--dtype", "bfloat16", "--limit-mm-per-prompt", json.dumps({"image": PROTOCOL["max_images"], "video": 0}),
        "--mm-processor-kwargs", json.dumps({"size": PROTOCOL["image_size"]})]
    if getattr(args, "enforce_eager", False):
        command.append("--enforce-eager")
    update(server_command=command)
    if args.dry_run:
        update(stage="dry_run", note="CPU metadata and cache checks only; no GPU query or server")
        print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
        return 0
    gpu_lock = local_lock_path(EVALUATION, f"gpu-{args.gpu}").open("a")
    try:
        fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        update(stage="waiting_gpu", error="Shared evaluation GPU lock is owned by another process")
        return 75
    resources = gpu_status(args.gpu)
    update(resources=resources)
    if resources["busy"]:
        update(stage="waiting_gpu", error="GPU has a compute PID or more than 256 MiB allocated")
        return 75
    if resources["warning"]:
        print(resources["warning"], flush=True)
    server = None
    server_log = out / "vllm.log"
    server_log_offset = server_log.stat().st_size if server_log.exists() else 0
    admission_budget = None
    handlers = {}
    def interrupted(signum, frame):
        # Closing our server also releases blocked HTTP workers before executor shutdown.
        update(stage="interrupted", signal=signum)
        stop_server(server)
        raise SystemExit(128 + signum)
    try:
        handlers = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
        env = dict(offline_environment(), CUDA_VISIBLE_DEVICES=str(args.gpu),
                   VLLM_USE_FLASHINFER_SAMPLER="0", OMP_NUM_THREADS="8")
        update(server_environment={key: env[key] for key in (
            "CUDA_VISIBLE_DEVICES", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE",
            "NO_PROXY", "VLLM_NO_USAGE_STATS", "OMP_NUM_THREADS")})
        with (out / "server_attempts.jsonl").open("a") as attempts:
            attempts.write(json.dumps({"started_at": now(), "profile_index": profile_index,
                "profile": server_profile, "command": command, "server_log_offset": server_log_offset,
                "scheduling_policy_sha256": digest(scheduling_policy)}) + "\n")
            attempts.flush()
            os.fsync(attempts.fileno())
        with server_log.open("a") as log:
            server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(stage="starting_server", server_pid=server.pid, server_log_offset=server_log_offset)
        base_url = f"http://127.0.0.1:{port}/v1"
        deadline = time.monotonic() + 1200
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(f"Owned vLLM exited {server.returncode}; see {out / 'vllm.log'}")
            try:
                models = request_json(base_url + "/models", timeout=5)
                if served_name in {m["id"] for m in models["data"]}:
                    break
            except (OSError, ValueError, KeyError):
                pass
            update(stage="starting_server")
            time.sleep(5)
        else:
            raise TimeoutError("Formal vLLM server did not become ready within 20 minutes")
        if high_throughput:
            capacity = throughput.read_cache_capacity(server_log, server_log_offset)
            admission_budget = throughput.runtime_budget(capacity, profile_index=profile_index)
            # Fail clearly before submitting if this profile cannot serve a row.
            for row in todo:
                if not throughput.can_admit(row, [], admission_budget):
                    raise ValueError("Profiled cache cannot admit a single validated row")
            update(admission_budget=admission_budget)
        update(stage="evaluating", first_formal_row=todo[0]["id"])
        with path.open("a") as stream:
            consecutive_errors = 0
            def persist(record):
                nonlocal consecutive_errors
                consecutive_errors = consecutive_errors + 1 if record.get("error") else 0
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
                previous[record["id"]] = record
                update(completed=sum(not x.get("error") for x in previous.values()),
                       errors=sum(bool(x.get("error")) for x in previous.values()), last_id=record["id"],
                       consecutive_api_errors=consecutive_errors)
            first = infer(todo[0], args, base_url, served_name)
            persist(first)
            if first.get("error"):
                update(stage="failed", error=first["error"])
                return 3 if first.get("fatal_error") else 2
            update(first_formal_prompt_tokens_verified=first["usage"]["prompt_tokens"])
            remaining = deque(todo[1:])
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
                pending = {}
                def fill_pending():
                    # Preserve frozen FIFO order. A large next request waits for
                    # capacity; smaller later rows never leapfrog it indefinitely.
                    while remaining and len(pending) < args.concurrency:
                        row = remaining[0]
                        if adaptive and not can_admit(row, pending.values()):
                            break
                        if high_throughput and not throughput.can_admit(row, pending.values(), admission_budget):
                            break
                        remaining.popleft()
                        pending[executor.submit(infer, row, args, base_url, served_name)] = row
                    if remaining and not pending:
                        raise ValueError("Scheduling policy cannot admit the next validated row")
                    update(inflight=len(pending),
                           inflight_prompt_tokens=sum(row["prompt_tokens"] for row in pending.values()),
                           inflight_visual_tokens=sum(row["visual_tokens"] for row in pending.values()),
                           inflight_reserved_tokens=sum(row["prompt_tokens"] + PROTOCOL["max_tokens"] + 4096 for row in pending.values()) if high_throughput else None,
                           inflight_slots=sum(row_cost(row) for row in pending.values()) if adaptive else len(pending))
                fill_pending()
                stop_error = None
                stop_code = 2
                while pending:
                    ready, _ = concurrent.futures.wait(pending, timeout=30,
                        return_when=concurrent.futures.FIRST_COMPLETED)
                    if not ready:
                        update()
                        if server.poll() is not None:
                            raise RuntimeError("Owned vLLM server exited during evaluation")
                    for future in ready:
                        del pending[future]
                        record = future.result()
                        persist(record)
                        if record.get("fatal_error"):
                            stop_error, stop_code = record["error"], 3
                        elif consecutive_errors >= 5 and stop_error is None:
                            stop_error = "Five consecutive failed API requests; stopping owned server for a bounded launcher retry"
                    if stop_error:
                        for future in pending:
                            future.cancel()
                        stop_server(server)
                        # Preserve responses already in flight, without submitting more rows.
                        for future in pending:
                            if not future.cancelled():
                                persist(future.result())
                        update(stage="failed", error=stop_error)
                        return stop_code
                    fill_pending()
        errors = sum(bool(previous[row["id"]].get("error")) for row in rows)
        update(stage="failed" if errors else "complete", errors=errors,
               completed_at=now(), error=f"{errors} rows have API errors" if errors else None)
        return 2 if errors else 0
    except Exception as error:
        update(stage="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        stop_server(server)
        update(server_pid=None, gpu_oom=throughput.owned_server_oom(server_log, server_log_offset) if server is not None else False)
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", choices=AUTHORIZED_MODELS + REGISTERED_MODEL_NAMES, required=True)
    parser.add_argument("--model", type=Path,
                        help="Explicit export path; required for a privately registered new model")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--root", type=Path, default=DATA)
    parser.add_argument("--output-root", type=Path, default=RESULTS)
    parser.add_argument("--benchmarks", nargs="+", choices=sorted(BENCHMARKS),
                        help="Run only these benchmarks from the existing frozen suite")
    parser.add_argument("--ids", type=Path,
                        help="JSON highres-subset-v1 manifest selecting frozen suite row IDs")
    parser.add_argument("--concurrency", type=int, choices=(1, 2, 4, 8, 16, 32, 64), default=2)
    parser.add_argument("--adaptive-concurrency", action="store_true",
                        help="Use visual-token admission up to 16 requests; requires --concurrency 16")
    parser.add_argument("--throughput-mode", action="store_true",
                        help="Use actual profiled KV capacity, 95 percent GPU budget and up to 64 requests")
    parser.add_argument("--execution-profile", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs for this execution; inference protocol is unchanged")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true", help="Validate CPU metadata/cache and print formal server command; never start CUDA")
    args = parser.parse_args()
    if args.timeout < 1 or args.retries < 1:
        parser.error("timeout and retries must be positive")
    raise SystemExit(execute(args))


if __name__ == "__main__":
    main()
