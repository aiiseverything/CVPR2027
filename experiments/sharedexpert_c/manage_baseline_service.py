#!/usr/bin/env python3
"""Release/restore recorded idle Qwen3.5-4B services on GPU 0/8110 or 2/8112.

GPU 2 is the default and retains the original baseline suspension state path.
GPU 0 has a separate state file. GPU 1 and all training GPUs are excluded.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EVAL = HERE / "evaluation"
STATE = HERE / "runtime/baseline_service_suspension.json"
ORIGINAL = EVAL / "restored_4B2.json"
PORT, GPU = 8112, "2"
SAFE_ENV = {"CUDA_VISIBLE_DEVICES", "CUDA_HOME", "PATH", "LD_LIBRARY_PATH", "HF_HOME",
            "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_USE_V1", "VLLM_WORKER_MULTIPROC_METHOD",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
            "TOKENIZERS_PARALLELISM", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"}
sys.path.insert(0, str(EVAL))
from common import PROTOCOL, SCORING_PROTOCOL, digest, file_sha, load_suite, score_prediction
from manage_training_gpus import cmdline, starttime, matches_process, healthy, save_json


def configure_gpu(gpu):
    """Select one explicitly allowed service; keep default-GPU state compatible."""
    global GPU, PORT, STATE, ORIGINAL
    if str(gpu) not in {"0", "2"}:
        raise ValueError("Only the recorded project services on GPU 0 or GPU 2 are supported")
    GPU = str(gpu)
    PORT = 8110 + int(GPU)
    ORIGINAL = EVAL / f"restored_4B{GPU}.json"
    STATE = HERE / "runtime" / ("baseline_service_suspension.json" if GPU == "2"
                               else "baseline_service_gpu0_suspension.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def process_running(pid):
    try:
        return (Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] not in {"Z", "X"})
    except FileNotFoundError:
        return False


def runtime_env(pid):
    return {key.decode(): value.decode() for item in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
            if b"=" in item for key, value in [item.split(b"=", 1)] if key.decode() in SAFE_ENV}


def original_record():
    source = read_json(ORIGINAL)
    record = {"pid": source["pid"], "starttime": source["proc_starttime"], "argv": source["argv"],
              "cwd": source["cwd"], "environment": source["nonsecret_runtime_environment"],
              "port": PORT, "gpu": GPU}
    validate_record(record)
    return record


def validate_record(record):
    args = record["argv"]
    expected = {"--port": str(PORT), "--served-model-name": "Qwen3.5-4B",
                "--max-model-len": "40960", "--max-num-seqs": "28"}
    if (args[:3] != [str(ROOT / "pre-lab/venv/bin/python3"), str(ROOT / "pre-lab/venv/bin/vllm"), "serve"]
            or args[3] != str(ROOT / "pre-lab/models/Qwen3.5-4B")
            or record["port"] != PORT or record["gpu"] != GPU
            or record["environment"].get("CUDA_VISIBLE_DEVICES") != GPU
            or record["cwd"] != str(ROOT / "pre-lab")
            or set(record["environment"]) - SAFE_ENV):
        raise RuntimeError("Unexpected baseline command, environment, or GPU identity")
    if any(args[args.index(key) + 1] != value for key, value in expected.items()):
        raise RuntimeError("Recorded baseline options changed")
    if json.loads(args[args.index("--limit-mm-per-prompt") + 1]) != {"image": 1 if GPU == "0" else 8}:
        raise RuntimeError("Image limit differs from the selected original 4B service")


def completed_baseline():
    """Read-only full-row and strict-score audit; never rewrites cached results."""
    rows, meta = load_suite(EVAL / "suite.jsonl")
    folder = EVAL / "results/full/baseline"
    run, scoring, summary = [read_json(p) for p in
                            [folder / "run.json", folder / "scoring.json", EVAL / "results/summary.json"]]
    expected = {"suite_sha256": meta["suite_sha256"], "protocol_sha256": meta["protocol_sha256"],
                "protocol": PROTOCOL, "name": "baseline", "served_model": "Qwen3.5-4B",
                "n_requested": len(rows), "smoke": False}
    if any(run.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Baseline run provenance is incompatible with frozen suite")
    predictions, scored = {}, {}
    for filename, target in [("predictions.jsonl", predictions), ("predictions.scored.jsonl", scored)]:
        with (folder / filename).open() as stream:
            for line in stream:
                item = json.loads(line)
                target[item["id"]] = item
    ids = {row["id"] for row in rows}
    if set(predictions) != ids or set(scored) != ids:
        raise RuntimeError("Baseline raw/scored coverage is incomplete")
    for row in rows:
        pred = predictions[row["id"]]
        if pred.get("error") or "raw_response" not in pred or pred.get("row_sha256") != digest(row):
            raise RuntimeError(f"Incomplete or incompatible baseline prediction: {row['id']}")
        rescored = score_prediction(row, pred)
        if any(scored[row["id"]].get(k) != rescored.get(k) for k in ["correct", "parse_success", "finish_reason"]):
            raise RuntimeError(f"Baseline scoring mismatch: {row['id']}")
    status = summary["status"]["baseline"]
    if (status["status"] != "complete" or status["missing"] != 0 or status["api_errors"] != 0
            or status["n_expected"] != len(rows)
            or summary["suite"]["suite_sha256"] != meta["suite_sha256"]
            or scoring["scorer_source_sha256"] != file_sha(EVAL / "common.py")
            or scoring["scoring_protocol_sha256"] != digest(SCORING_PROTOCOL)
            or summary["scoring_protocol_sha256"] != digest(SCORING_PROTOCOL)):
        raise RuntimeError("Baseline completion/scoring evidence is stale")
    return {"verified_at": now(), "n": len(rows), "missing": 0, "api_errors": 0,
            "row_hash_mismatch": 0, "score_recompute_mismatch": 0,
            "suite_sha256": meta["suite_sha256"], "protocol_sha256": meta["protocol_sha256"],
            "scoring_protocol_sha256": digest(SCORING_PROTOCOL),
            "summary": str(EVAL / "results/summary.json"),
            "files_sha256": {str(p): file_sha(p) for p in [folder / "run.json", folder / "scoring.json",
                folder / "predictions.jsonl", folder / "predictions.scored.jsonl", EVAL / "results/summary.json"]}}


def gpu_snapshot():
    gpu_line = subprocess.check_output(["nvidia-smi", "-i", GPU, "--query-gpu=uuid,memory.used",
                                       "--format=csv,noheader,nounits"], text=True).strip()
    uuid, memory = [part.strip() for part in gpu_line.split(",")]
    apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                    "--format=csv,noheader"], text=True)
    pids = [int(line.split(",")[1]) for line in apps.splitlines() if line.split(",")[0].strip() == uuid]
    return {"gpu": GPU, "uuid": uuid, "memory_used_mib": int(memory), "compute_pids": pids}


def memory_snapshot():
    root = Path("/sys/fs/cgroup")
    stats = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
    current = int((root / "memory.current").read_text())
    return {"at": now(), "current_bytes": current, "working_set_bytes": current - int(stats["inactive_file"]),
            "anon_bytes": int(stats["anon"]), "file_bytes": int(stats["file"]),
            "memory_events": dict(line.split() for line in (root / "memory.events").read_text().splitlines())}


def descendants(pid):
    found = {pid}
    all_processes = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            parts = (path / "stat").read_text().rsplit(")", 1)[1].split()
            all_processes[int(path.name)] = int(parts[1])
        except (FileNotFoundError, ProcessLookupError):
            continue
    while True:
        larger = found | {child for child, parent in all_processes.items() if parent in found}
        if larger == found:
            break
        found = larger
    records = []
    for child in sorted(found):
        info = {"pid": child, "starttime": starttime(child), "argv": cmdline(child)}
        rollup = Path(f"/proc/{child}/smaps_rollup")
        info["memory_kib"] = {k: int(v.split()[0]) for line in rollup.read_text().splitlines()
                              for k, _, v in [line.partition(":")] if k in {"Rss", "Pss", "Pss_Anon"}}
        records.append(info)
    return records


def idle_snapshot():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5) as response:
        data = response.read().decode()
    values = {}
    for line in data.splitlines():
        if line.startswith(("vllm:num_requests_running{", "vllm:num_requests_waiting{",
                            "vllm:kv_cache_usage_perc{", "vllm:request_success_total{",
                            "vllm:generation_tokens_total{")):
            key, value = line.rsplit(" ", 1)
            values[key] = float(value)
    for metric in ["num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"]:
        matched = [v for k, v in values.items() if k.startswith(f"vllm:{metric}{{")]
        if not matched or any(matched):
            raise RuntimeError(f"Baseline endpoint has work or missing {metric}; leave service running")
    return {"at": now(), "values": values}


def idle_clients():
    """Reject live clients; LISTEN and fully closed TIME_WAIT sockets are safe."""
    connections = []
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        for line in table.read_text().splitlines()[1:]:
            fields = line.split()
            local_port, remote_port = [int(fields[i].split(":")[1], 16) for i in (1, 2)]
            if PORT not in (local_port, remote_port):
                continue
            item = {"local_port": local_port, "remote_port": remote_port, "state_hex": fields[3]}
            connections.append(item)
            if item["state_hex"] not in {"0A", "06"}:
                raise RuntimeError(f"Port {PORT} has a live/closing client; leave service running")
    return {"at": now(), "connections": connections, "live_clients": 0}


def port_open():
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", PORT)) == 0


def verify_owned(record, children):
    validate_record(record)
    if not matches_process(record) or not process_running(record["pid"]):
        raise RuntimeError("Baseline PID, start ticks, or exact argv no longer matches")
    if runtime_env(record["pid"]).get("CUDA_VISIBLE_DEVICES") != GPU:
        raise RuntimeError("Live baseline process GPU mapping changed")
    if str(Path(f"/proc/{record['pid']}/cwd").resolve()) != record["cwd"]:
        raise RuntimeError("Live service working directory differs from its saved identity")
    gpu = gpu_snapshot()
    owned = {r["pid"] for r in children}
    if not gpu["compute_pids"] or not set(gpu["compute_pids"]).issubset(owned):
        raise RuntimeError(f"GPU {GPU} has no recorded engine or contains another process; refuse release")
    return gpu


def wait_released(state):
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        surviving = [r["pid"] for r in state["owned_processes"]
                     if process_running(r["pid"]) and starttime(r["pid"]) == r["starttime"]]
        gpu = gpu_snapshot()
        if not surviving and not gpu["compute_pids"] and gpu["memory_used_mib"] < 1024 and not port_open():
            state.update(phase="released", released_at=now(), gpu_after=gpu, memory_after=memory_snapshot())
            state["memory_reduction_bytes"] = {k: state["memory_before"][k] - state["memory_after"][k]
                                                for k in ["current_bytes", "working_set_bytes", "anon_bytes"]}
            save_json(STATE, state)
            print(json.dumps({"phase": state["phase"], "state": str(STATE), "gpu": gpu,
                              "memory_reduction_bytes": state["memory_reduction_bytes"]}), flush=True)
            return
        time.sleep(2)
    raise RuntimeError("Recorded server did not fully release within 120s; no additional process was signalled")


def release():
    evidence = completed_baseline()
    if STATE.exists():
        state = read_json(STATE)
        if state["phase"] in {"release_pending", "signalled", "released"}:
            if state["phase"] == "released":
                gpu = gpu_snapshot()
                surviving = [r["pid"] for r in state["owned_processes"]
                             if process_running(r["pid"]) and starttime(r["pid"]) == r["starttime"]]
                if surviving or gpu["compute_pids"] or gpu["memory_used_mib"] >= 1024 or port_open():
                    raise RuntimeError("Released baseline resources are now occupied; no process was changed")
                print(json.dumps({"phase": "released", "already_released": True, "state": str(STATE),
                                  "gpu": gpu, "memory_reduction_bytes": state["memory_reduction_bytes"]}))
                return
            if not process_running(state["original"]["pid"]):
                wait_released(state)
                return
            if state["phase"] == "signalled":
                wait_released(state)
                return
            record = state["original"]
        elif state["phase"] == "restored":
            record = state["restored"]
        else:
            raise RuntimeError("Restore is in progress; release refused")
    else:
        state = None
        record = original_record()
    if not matches_process(record):
        raise RuntimeError("Recorded original/restored baseline server is no longer the same process")
    # Preserve only an explicit non-secret allowlist from the actual process.
    live_environment = runtime_env(record["pid"])
    if any(live_environment.get(key) != value for key, value in record["environment"].items()):
        raise RuntimeError("Saved non-secret service environment differs from the live process")
    record["environment"] = live_environment
    children = descendants(record["pid"])
    gpu = verify_owned(record, children)
    first_clients = idle_clients()
    first = idle_snapshot()
    idle_started = time.monotonic()
    time.sleep(20)
    second = idle_snapshot()
    second_clients = idle_clients()
    if first["values"] != second["values"]:
        raise RuntimeError("Endpoint counters changed between idle checks; leave it running")
    verify_owned(record, children)
    state = {"version": 1, "phase": "release_pending", "created_at": now(),
             "original_record_path": str(ORIGINAL), "original_record_sha256": file_sha(ORIGINAL),
             "original": record, "owned_processes": children, "baseline_evidence": evidence,
             "idle_checks": [first, second], "idle_interval_seconds": time.monotonic() - idle_started,
             "client_checks": [first_clients, second_clients], "gpu_before": gpu, "memory_before": memory_snapshot(),
             "previous_cycle": state}
    STATE.parent.mkdir(parents=True, exist_ok=True)
    save_json(STATE, state)
    if not matches_process(record):
        raise RuntimeError("Identity changed immediately before SIGTERM; no signal sent")
    idle_clients()
    os.kill(record["pid"], signal.SIGTERM)  # Only this API parent; never killpg/pkill/engine kill.
    state.update(phase="signalled", signalled_at=now(), signalled_pid=record["pid"])
    save_json(STATE, state)
    print(json.dumps({"signalled_only_api_parent": record["pid"], "state": str(STATE)}), flush=True)
    wait_released(state)


def wait_restore(state):
    record = state["restored"]
    deadline = time.monotonic() + 1200
    while time.monotonic() < deadline:
        if not matches_process(record) or not process_running(record["pid"]):
            raise RuntimeError("Restored baseline exited; identity and log retained for retry")
        if healthy(record):
            children = descendants(record["pid"])
            gpu = verify_owned(record, children)
            state.update(phase="restored", healthy_at=now(), restored_gpu=gpu, restored_processes=children)
            save_json(STATE, state)
            print(json.dumps({"phase": "restored", "pid": record["pid"], "port": PORT,
                              "gpu": GPU, "healthy": True, "state": str(STATE)}), flush=True)
            return
        time.sleep(2)
    raise RuntimeError("Baseline readiness timed out; partial restore state retained")


def restore():
    if not STATE.exists():
        original = original_record()
        if healthy(original):
            print(json.dumps({"phase": "original_running", "pid": original["pid"], "healthy": True}))
            return
        raise RuntimeError("No saved suspension exists and original server is not healthy")
    state = read_json(STATE)
    record = state.get("restored")
    if record and matches_process(record) and process_running(record["pid"]):
        validate_record(record)
        wait_restore(state)
        return
    if process_running(state["original"]["pid"]):
        raise RuntimeError("Original API parent still exists; no duplicate restore")
    if any(process_running(r["pid"]) and starttime(r["pid"]) == r["starttime"]
           for r in state["owned_processes"]):
        raise RuntimeError("An original baseline child still exists; no duplicate restore")
    gpu = gpu_snapshot()
    if gpu["compute_pids"] or gpu["memory_used_mib"] >= 1024 or port_open():
        raise RuntimeError(f"GPU {GPU} or port {PORT} is busy; no restart performed")
    original = state["original"]
    validate_record(original)
    env = dict(os.environ)
    env.update(original["environment"])
    env["CUDA_VISIBLE_DEVICES"] = GPU
    log_path = STATE.parent / f"restored_baseline_{PORT}.log"
    with log_path.open("ab", buffering=0) as log:
        process = subprocess.Popen(original["argv"], cwd=original["cwd"], env=env, stdin=subprocess.DEVNULL,
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(original, pid=process.pid, starttime=starttime(process.pid))
    state.update(phase="restoring", restore_started_at=now(), restored=record, restore_log=str(log_path))
    save_json(STATE, state)
    print(json.dumps({"phase": "restoring", "pid": process.pid, "log": str(log_path)}), flush=True)
    wait_restore(state)


def status():
    state = read_json(STATE) if STATE.exists() else None
    record = (state.get("restored") or state["original"]) if state else original_record()
    result = {"phase": state["phase"] if state else "not_suspended", "state": str(STATE),
              "recorded_pid": record["pid"], "identity_matches": matches_process(record),
              "healthy": healthy(record), "gpu": gpu_snapshot(), "memory": memory_snapshot()}
    if result["healthy"]:
        try:
            result["idle"] = idle_snapshot()
        except RuntimeError as error:
            result["idle_error"] = str(error)
    if state:
        result["baseline_evidence"] = state["baseline_evidence"]
        result["memory_reduction_bytes"] = state.get("memory_reduction_bytes")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["release", "restore", "status"])
    parser.add_argument("--gpu", choices=["0", "2"], default="2")
    args = parser.parse_args()
    configure_gpu(args.gpu)
    # Host-local locking avoids the stale network-filesystem flock issue.
    identity = "" if GPU == "2" else "-gpu0"
    lock = Path("/tmp") / f"sharedexpert-baseline-service{identity}-{os.getuid()}.lock"
    with lock.open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[args.action]()


if __name__ == "__main__":
    main()
