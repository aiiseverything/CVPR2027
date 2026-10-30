#!/usr/bin/env python3
"""Run the unchanged perception screening core with independent GPU replicas.

The core remains the sole journal writer. This execution backend partitions
whole questions, keeps all eight samples together, and records its provenance
separately from the unchanged model/sampling/scoring fingerprint.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
from functools import partial
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import filter_perception_v3 as core


def _image_grid(width, height, image_processor, min_pixels, max_pixels):
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    patch = image_processor.patch_size
    resized_h, resized_w = smart_resize(height=height, width=width,
                                       factor=patch * image_processor.merge_size,
                                       min_pixels=min_pixels, max_pixels=max_pixels)
    return [1, resized_h // patch, resized_w // patch]


def compatible_training_prompt(record, protocol, processor, *, prepare):
    """Exclude inputs incompatible with the unchanged formal training recipe.

    Frozen HF defaults and legacy vLLM pixel overrides differ for tiny images.
    Such inputs are ineligible; no sampled answer is generated or scored wrong.
    """
    prompt, metadata = prepare(record, protocol, processor)
    if prompt is None:
        return prompt, metadata
    image_processor = processor.image_processor
    size = image_processor.size
    training = [_image_grid(width, height, image_processor,
                            size["shortest_edge"], size["longest_edge"])
                for width, height in metadata["first_resize_wh"]]
    inference = [_image_grid(width, height, image_processor,
                             protocol["min_image_pixels"], protocol["max_image_pixels"])
                 for width, height in metadata["first_resize_wh"]]
    if training != inference:
        evidence = {key: value for key, value in metadata.items() if key != "expected_prompt_token_ids"}
        evidence.update(reason="training_inference_image_grid_mismatch",
                        training_grid_thw=training, inference_grid_thw=inference)
        return None, evidence
    return prompt, metadata


def split_batch(prompts, params, worker_count):
    if worker_count < 1 or len(prompts) != len(params):
        raise ValueError("Invalid parallel batch dimensions")
    return [{"indices": list(range(worker, len(prompts), worker_count)),
             "prompts": [prompts[i] for i in range(worker, len(prompts), worker_count)],
             "sampling_params": [params[i] for i in range(worker, len(prompts), worker_count)]}
            for worker in range(worker_count)]


def assemble_batch(shards, replies, size):
    if len(shards) != len(replies):
        raise ValueError("Missing worker response")
    ordered = [None] * size
    for shard, reply in zip(shards, replies, strict=True):
        if reply.get("indices") != shard["indices"] or len(reply.get("outputs", [])) != len(shard["indices"]):
            raise ValueError("Worker returned a mismatched batch")
        if len(shard["sampling_params"]) != len(shard["indices"]):
            raise ValueError("Invalid shard sampling parameters")
        for index, output, params in zip(shard["indices"], reply["outputs"], shard["sampling_params"], strict=True):
            if not 0 <= index < size or ordered[index] is not None:
                raise ValueError("Duplicate or invalid request index")
            completions = output.get("outputs", [])
            if (output.get("finished") is not True or len(completions) != params.n or
                    sorted(x["index"] for x in completions) != list(range(params.n)) or
                    any(x.get("finish_reason") not in {"stop", "length"} for x in completions)):
                raise ValueError("Incomplete worker generation; no partial batch may be committed")
            ordered[index] = SimpleNamespace(prompt_token_ids=output["prompt_token_ids"],
                                            outputs=[SimpleNamespace(**x) for x in completions])
    if any(output is None for output in ordered):
        raise ValueError("Missing request in parallel result")
    return ordered


def process_identity(pid):
    stat = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return {"pid": pid, "start_ticks": int(stat[19])}


def worker_main(gpu, kwargs, pipe):
    # Spawned children import this module without importing torch or vLLM.
    os.setsid()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.environ["VLLM_LOGGING_PREFIX"] = f"[screen-gpu-{gpu}] "
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        from vllm import LLM
        llm = LLM(**kwargs)
        pipe.send({"kind": "ready", "gpu": gpu, **process_identity(os.getpid())})
        while True:
            job = pipe.recv()
            if job["kind"] == "close":
                return
            if job["kind"] != "generate":
                raise ValueError("Unknown GPU worker request")
            started = time.monotonic()
            result = llm.generate(job["prompts"], job["sampling_params"], use_tqdm=False)
            payload = [{"prompt_token_ids": list(item.prompt_token_ids), "finished": bool(item.finished),
                        "outputs": [{"index": x.index, "token_ids": list(x.token_ids),
                                     "finish_reason": x.finish_reason} for x in item.outputs]}
                       for item in result]
            pipe.send({"kind": "result", "batch_id": job["batch_id"], "indices": job["indices"],
                       "outputs": payload, "gpu": gpu,
                       "metrics": {"generation_seconds": time.monotonic() - started,
                                   "prompts": len(payload),
                                   "responses": sum(len(x["outputs"]) for x in payload),
                                   "output_tokens": sum(len(y["token_ids"]) for x in payload for y in x["outputs"])}})
    except BaseException:
        try:
            pipe.send({"kind": "error", "gpu": gpu, "error": traceback.format_exc()})
        except (OSError, EOFError):
            pass
        raise
    finally:
        pipe.close()


class ParallelLLM:
    def __init__(self, devices, output_dir, batch_prompts, **kwargs):
        self.devices = devices
        self.output_dir = output_dir
        self.workers = []
        self.locks = []
        self.closed = False
        self.batch_id = 0
        self.receipt_path = None
        self.receipt = {}
        if kwargs.get("tensor_parallel_size") != 1 or len(set(devices)) != len(devices):
            raise ValueError("Parallel screening requires distinct single-GPU replicas")
        try:
            # The unchanged core already owns/checks the first GPU lock.
            for gpu in devices[1:]:
                lock = open(f"/tmp/sharedexpert-perception-v3-gpu-{gpu}.lock", "a")
                self.locks.append(lock)
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                uuid = subprocess.check_output(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid",
                                                "--format=csv,noheader"], text=True).strip()
                apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                                                "--format=csv,noheader"], text=True)
                if any(line.startswith(uuid) for line in apps.splitlines()):
                    raise RuntimeError(f"GPU {gpu} already has a compute process")
            self._start_receipt(kwargs, batch_prompts)
            context = mp.get_context("spawn")
            for gpu in devices:
                parent_pipe, child_pipe = context.Pipe()
                process = context.Process(target=worker_main, args=(gpu, kwargs, child_pipe),
                                          name=f"perception-screen-gpu-{gpu}", daemon=False)
                process.start()
                child_pipe.close()
                self.workers.append({"gpu": gpu, "process": process, "pipe": parent_pipe})
            ready = self._collect(range(len(self.workers)), "ready", timeout=1800)
            self.receipt.update(status="running", workers=[ready[i] for i in range(len(self.workers))])
            self._save_receipt()
            print(json.dumps({"stage": "parallel_workers_ready", "devices": devices,
                              "worker_pids": [x["process"].pid for x in self.workers]}), flush=True)
        except BaseException as exc:
            self.close(error=f"{type(exc).__name__}: {exc}")
            raise

    def _start_receipt(self, kwargs, batch_prompts):
        execution_dir = self.output_dir / "executions"
        execution_dir.mkdir(exist_ok=True)
        journal = self.output_dir / "groups.jsonl"
        if not journal.exists():
            journal.touch()
        initial_bytes = journal.read_bytes()
        snapshot = execution_dir / "legacy-before-parallel.groups.jsonl"
        if not snapshot.exists():
            with snapshot.open("xb") as f:
                f.write(initial_bytes)
                f.flush()
                os.fsync(f.fileno())
        elif not initial_bytes.startswith(snapshot.read_bytes()):
            raise ValueError("Preserved legacy journal is not a prefix of the current journal")
        run = json.loads((self.output_dir / "run.json").read_text())
        core_sha = core.file_sha(HERE / "filter_perception_v3.py")
        if run["spec"]["driver_source_sha256"] != core_sha:
            raise ValueError("Screening core differs from the frozen run")
        wrapper_sha = core.file_sha(__file__)
        source_dir = execution_dir / "sources"
        source_dir.mkdir(exist_ok=True)
        source = source_dir / f"parallel-{wrapper_sha}.py"
        if not source.exists():
            with source.open("xb") as f:
                f.write(Path(__file__).read_bytes())
        if core.file_sha(source) != wrapper_sha:
            raise ValueError("Archived execution source changed")
        now = dt.datetime.now(dt.timezone.utc)
        self.receipt_path = execution_dir / f"parallel-{now.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}.json"
        self.receipt = {"schema": 1, "backend": "independent_gpu_replicas", "devices": self.devices,
                        "tensor_parallel_size_per_worker": 1, "core_driver_sha256": core_sha,
                        "parallel_driver_sha256": wrapper_sha,
                        "parallel_driver_source": str(source.relative_to(execution_dir)), "run_sha256": run["run_sha256"],
                        "initial_groups_sha256": core.file_sha(journal), "initial_groups_bytes": len(initial_bytes),
                        "initial_group_count": initial_bytes.count(b'\n'), "batch_prompts": batch_prompts,
                        **process_identity(os.getpid()), "started_at": now.isoformat(),
                        "status": "initializing", "workers": [], "llm_kwargs": kwargs,
                        "input_eligibility": {"version": "training-inference-image-grid-parity-v1",
                                              "failure_policy": "exclude_before_sampling_not_an_incorrect_answer"},
                        "note": "Topology changed; model, per-record seeds, n=8, preprocessing and scoring unchanged."}
        self._save_receipt()

    def _save_receipt(self):
        if self.receipt_path is not None:
            core.atomic_json(self.receipt_path, self.receipt)

    def _collect(self, indices, kind, timeout):
        pending = set(indices)
        replies = {}
        deadline = time.monotonic() + timeout
        while pending:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"GPU workers timed out waiting for {kind}")
            pipes = {self.workers[i]["pipe"]: i for i in pending}
            for pipe in wait(list(pipes), timeout=1):
                i = pipes[pipe]
                try:
                    message = pipe.recv()
                except EOFError as exc:
                    raise RuntimeError(f"GPU {self.workers[i]['gpu']} disconnected") from exc
                if message.get("kind") == "error":
                    raise RuntimeError(f"GPU {self.workers[i]['gpu']} failed:\n{message['error']}")
                if message.get("kind") != kind or message.get("gpu") != self.workers[i]["gpu"]:
                    raise RuntimeError("Unexpected GPU worker response")
                replies[i] = message
                pending.remove(i)
            for i in pending:
                process = self.workers[i]["process"]
                if not process.is_alive():
                    raise RuntimeError(f"GPU {self.workers[i]['gpu']} worker exited {process.exitcode}")
        return replies

    def generate(self, prompts, sampling_params, use_tqdm=False):
        started = time.monotonic()
        shards = split_batch(prompts, sampling_params, len(self.workers))
        self.batch_id += 1
        active = [i for i, shard in enumerate(shards) if shard["indices"]]
        for i in active:
            self.workers[i]["pipe"].send({"kind": "generate", "batch_id": self.batch_id, **shards[i]})
        # Larger queued windows can legitimately take longer than the old 24-question batch.
        received = self._collect(active, "result", timeout=max(1800, 60 * len(prompts)))
        if any(reply.get("batch_id") != self.batch_id for reply in received.values()):
            raise RuntimeError("Stale GPU worker batch")
        replies = [received.get(i, {"indices": [], "outputs": []}) for i in range(len(shards))]
        result = assemble_batch(shards, replies, len(prompts))
        seconds = time.monotonic() - started
        metrics = {"batch_id": self.batch_id, "parent_pid": os.getpid(),
                   "time": dt.datetime.now(dt.timezone.utc).isoformat(), "prompts": len(prompts),
                   "generation_seconds": seconds,
                   "output_tokens": sum(len(c.token_ids) for output in result for c in output.outputs),
                   "workers": [{"gpu": reply["gpu"], **reply.get("metrics", {})}
                               for reply in received.values()]}
        metrics["output_tokens_per_second"] = metrics["output_tokens"] / max(seconds, 1e-9)
        metrics["questions_per_second"] = len(prompts) / max(seconds, 1e-9)
        with (self.output_dir / "throughput.jsonl").open("a") as log:
            log.write(json.dumps(metrics) + "\n")
        print(json.dumps({"stage": "parallel_batch_throughput", **metrics}), flush=True)
        return result

    def close(self, error=None):
        if self.closed:
            return
        self.closed = True
        # Each owned worker starts a separate session, including its vLLM child.
        for worker in self.workers:
            process = worker["process"]
            if process.is_alive():
                try:
                    if os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGTERM)
                    else:
                        process.terminate()
                except ProcessLookupError:
                    pass
        for worker in self.workers:
            process = worker["process"]
            process.join(timeout=10)
            try:
                # Also remove a lingering engine child in this owned session.
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            worker["pipe"].close()
        for lock in self.locks:
            lock.close()
        if self.receipt_path:
            journal = self.output_dir / "groups.jsonl"
            data = journal.read_bytes()
            self.receipt.update(status="failed" if error else "complete", error=error,
                                finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                                final_groups_sha256=core.file_sha(journal), final_groups_bytes=len(data),
                                final_group_count=data.count(b'\n'))
            self._save_receipt()


def main():
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    options, remaining = parser.parse_known_args()
    devices = [int(x) for x in options.gpus.split(',')]
    if len(devices) not in (1, 2) or len(set(devices)) != len(devices):
        raise ValueError("Screening requires one or two distinct GPUs")
    if options.max_num_seqs < 1 or not 0 < options.gpu_memory_utilization <= 0.95:
        raise ValueError("Invalid screening runtime capacity")
    if any(arg == "--gpu" or arg.startswith("--gpu=") for arg in remaining):
        raise ValueError("Use --gpus with the parallel driver")
    if not any(arg == "--batch-prompts" or arg.startswith("--batch-prompts=") for arg in remaining):
        remaining += ["--batch-prompts", "256"]
    remaining += ["--gpu", str(devices[0])]
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--output-dir", type=Path, default=core.HERE / "prepared/perception_v3_filter")
    config_parser.add_argument("--batch-prompts", type=int)
    settings, _ = config_parser.parse_known_args(remaining)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[0])
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    sys.argv = [str(HERE / "filter_perception_v3.py"), *remaining]
    if "--prepare-only" in remaining:
        original_prepare = core.prepare_prompt
        core.prepare_prompt = partial(compatible_training_prompt, prepare=original_prepare)
        try:
            core.main()
        finally:
            core.prepare_prompt = original_prepare
        return
    from vllm import LLM
    import vllm
    backend = None
    error = None
    def factory(**kwargs):
        nonlocal backend
        runtime_kwargs = {**kwargs, "max_num_seqs": options.max_num_seqs,
                          "gpu_memory_utilization": options.gpu_memory_utilization,
                          "disable_log_stats": False}
        backend = ParallelLLM(devices, settings.output_dir, settings.batch_prompts, **runtime_kwargs)
        return backend
    def stop(signum, frame):
        raise InterruptedError(f"Parallel screening received signal {signum}")
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    vllm.LLM = factory
    original_prepare = core.prepare_prompt
    core.prepare_prompt = partial(compatible_training_prompt, prepare=original_prepare)
    try:
        core.main()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        vllm.LLM = LLM
        core.prepare_prompt = original_prepare
        if backend is not None:
            backend.close(error=error)


if __name__ == '__main__':
    main()
