"""Train both independent 2B students after the authorized dataset export completes."""
import argparse
from contextlib import ExitStack, contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from opd_config import HERE, ROOT, REPO, EXPERIMENT, CURATION, PYTHON, ORIGINALS, VIEWS, RUNS, DATASET, SETTINGS, atomic_json, baseline, compose_config, digest, now, prepare_model_views, sha, validate_config
sys.path[:0] = [str(ROOT),str(REPO),str(HERE)]


def planned_identity():
    diff = subprocess.check_output(["git","-C",str(REPO),"diff","HEAD","--"],text=True)
    return {"settings":SETTINGS,
        "queue_source_sha256":{p.name:sha(p) for p in sorted(HERE.glob("*.py"))},
        "agent_config_sha256":sha(HERE / "agent_loops.yaml"),
        "shared_source_sha256":{str(EXPERIMENT / "training" / n):sha(EXPERIMENT / "training" / n)
            for n in ["launch_100.py","agent_loop.py","reward_manager.py","preflight.py","config/common.json"]},
        "repository_head":subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip(),
        "repository_diff_sha256":hashlib.sha256(diff.encode()).hexdigest()}


def verify_model_lineage(origin_files):
    curated = json.loads((CURATION / "preflight.json").read_text())["model_files"]
    previous = json.loads((EXPERIMENT / "analysis/virl63_token_credit_20261029/preflight.json").read_text())["model_files"]
    for name, current in origin_files.items():
        expected = previous["student"] if name == "student" else curated[name]
        a = {x["name"]:(x["size"],x["sha256"]) for x in current}
        b = {x["name"]:(x["size"],x["sha256"]) for x in expected}
        assert a == b, f"Model differs from the authorized initial student / curation teacher: {name}"


def verify_frozen_inputs():
    frozen = json.loads((HERE / "frozen_training.json").read_text())
    assert frozen["settings"] == SETTINGS
    assert sha(DATASET) == frozen["data"]["parquet_sha256"]
    assert sha(HERE / "exposures.jsonl") == frozen["data"]["exposure_manifest_sha256"]
    for name, expected in frozen["queue_source_sha256"].items():
        assert sha(HERE / name) == expected, f"Frozen training source changed: {name}"
    for name, expected in frozen["shared_source_sha256"].items():
        assert sha(name) == expected, f"Shared training source changed: {name}"
    assert sha(HERE / "agent_loops.yaml") == frozen["agent_config_sha256"]
    for name, files in frozen["original_models"].items():
        for item in files:
            file = ORIGINALS[name] / item["name"]
            assert file.stat().st_size == item["size"] and sha(file) == item["sha256"], f"Frozen model changed: {file}"
    for name, expected in frozen["processor_view_sha256"].items():
        assert sha(name) == expected, f"Processor settings changed: {name}"
    head = subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
    diff = subprocess.check_output(["git","-C",str(REPO),"diff","HEAD","--"],text=True)
    assert head == frozen["repository_head"] and hashlib.sha256(diff.encode()).hexdigest() == frozen["repository_diff_sha256"]


def freeze():
    plan = json.loads((HERE / "planned_training.json").read_text())
    assert planned_identity() == plan, "Queued training code/config changed; refusing to silently replace the checked plan"
    import pyarrow.parquet as pq
    from omegaconf import OmegaConf
    from prepare_training import main as prepare_data
    from check_cpu import check
    data = prepare_data()
    prepare_model_views()
    accepted = []
    with (CURATION / "accepted.jsonl").open() as f:
        for line in f:
            accepted.append(json.loads(line))
            if len(accepted) == 3:
                break
    validation = check(accepted)
    rows = pq.read_table(DATASET).to_pylist()
    assert len(rows) == 6300 and len(rows) // 63 == 100
    for i,row in enumerate(rows):
        assert row["extra_info"]["index"] == i
        assert 0 < row["extra_info"]["prompt_tokens"] <= 8192
        assert row["teacher_images"] == [x["path"] for x in row["images"]]
        assert row["prompt"][0]["content"].count("<image>") == 1
    dep_spec = importlib.util.spec_from_file_location("existing_training_cpu_checks", EXPERIMENT / "training/preflight.py")
    dep = importlib.util.module_from_spec(dep_spec)
    dep_spec.loader.exec_module(dep)
    dependencies = dep.check_imports()
    assert not dependencies["errors"], dependencies
    dataset_check = dep.check_dataset(DATASET,6300)
    assert not dataset_check["errors"], dataset_check
    origin_files = {}
    for name, path in ORIGINALS.items():
        origin_files[name] = [{"name":f.name,"size":f.stat().st_size,"mtime_ns":f.stat().st_mtime_ns,"sha256":sha(f)}
                              for f in sorted(path.iterdir()) if f.is_file() and f.suffix in {".json",".safetensors",".jinja",".txt"}]
    verify_model_lineage(origin_files)
    repo_head = subprocess.check_output(["git","-C",str(REPO),"rev-parse","HEAD"],text=True).strip()
    repo_diff = subprocess.check_output(["git","-C",str(REPO),"diff","HEAD","--"],text=True)
    identity = {"settings":SETTINGS,"data":data,"original_models":origin_files,
        "repository_head":repo_head,"repository_diff_sha256":hashlib.sha256(repo_diff.encode()).hexdigest(),
        "queue_source_sha256":{p.name:sha(p) for p in sorted(HERE.glob("*.py"))},
        "agent_config_sha256":sha(HERE / "agent_loops.yaml"),
        "processor_view_sha256":{str(p):sha(p) for view in VIEWS.values() for p in sorted(view.glob("*.json")) if not p.is_symlink()},
        "shared_source_sha256":{str(EXPERIMENT / "training" / n):sha(EXPERIMENT / "training" / n)
                                for n in ["launch_100.py","agent_loop.py","reward_manager.py","preflight.py","config/common.json"]}}
    snapshot = HERE / "frozen_training.json"
    if snapshot.exists():
        assert json.loads(snapshot.read_text()) == identity, "Training inputs changed since first formal run"
    else:
        atomic_json(snapshot,identity)
    (HERE / "repository.patch").write_text(repo_diff)
    for teacher in SETTINGS["teachers"]:
        cfg = compose_config(teacher)
        validate_config(cfg)
        folder = RUNS[teacher]
        folder.mkdir(parents=True,exist_ok=True)
        text = OmegaConf.to_yaml(cfg,resolve=True)
        dest = folder / "resolved_config.yaml"
        if dest.exists():
            assert dest.read_text() == text
        else:
            dest.write_text(text)
        protocol = {"teacher":teacher,"settings":SETTINGS,"data_sha256":sha(DATASET),
                    "frozen_training_sha256":sha(snapshot),"initial_student_files":origin_files["student"],
                    "fixed_teacher_files":origin_files[teacher],"gpu_smoke_performed":False}
        file = folder / "protocol.json"
        if file.exists():
            assert json.loads(file.read_text()) == protocol
        else:
            atomic_json(file,protocol)
    atomic_json(HERE / "training_preflight.json",{"status":"passed","at":now(),"gpu_smoke_performed":False,
        "dependencies":dependencies,"dataset":dataset_check,"cpu_input_and_loss_validation":validation,
        "frozen_training_sha256":sha(snapshot),"cgroup_memory_limit":Path("/sys/fs/cgroup/memory.max").read_text().strip()})
    return identity


def acquire_gpus(stack):
    evaluation = EXPERIMENT / "evaluation"
    folder = Path("/tmp") / f"sharedexpert-evaluation-locks-{os.getuid()}"
    folder.mkdir(mode=0o700,exist_ok=True)
    for gpu in SETTINGS["gpus"]:
        key = hashlib.sha256(f"{evaluation.resolve()}:gpu-{gpu}".encode()).hexdigest()
        handle = stack.enter_context((folder / (key+".lock")).open("a"))
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return baseline().gpu_check([str(g) for g in SETTINGS["gpus"]])


@contextmanager
def reserve_gpus(teacher):
    while True:
        stack = ExitStack()
        try:
            resources = acquire_gpus(stack)
        except (BlockingIOError, RuntimeError) as error:
            stack.close()
            if not isinstance(error, BlockingIOError) and not any(s in str(error) for s in
                ["is busy (", "Selected GPU has an existing compute process"]):
                raise
            atomic_json(HERE / "queue_status.json",{"status":"waiting_for_gpus","at":now(),
                "teacher":teacher,"pid":os.getpid(),"gpus":SETTINGS["gpus"],"reason":str(error),
                "other_processes_untouched":True})
            time.sleep(30)
        except BaseException:
            stack.close()
            raise
        else:
            try:
                yield resources
            finally:
                stack.close()
            return


def launch(teacher):
    folder = RUNS[teacher]
    legacy = baseline()
    latest = folder / "checkpoints/latest_checkpointed_iteration.txt"
    completed = int(latest.read_text().strip()) if latest.exists() else 0
    assert 0 <= completed <= 100
    env = dict(os.environ,PYTHONPATH=str(ROOT)+":"+str(REPO),CUDA_VISIBLE_DEVICES=",".join(map(str,SETTINGS["gpus"])),
        PYTHONUNBUFFERED="1",HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1",HF_DATASETS_OFFLINE="1",
        VLLM_USE_V1="1",VLLM_NO_USAGE_STATS="1",VLLM_USE_FLASHINFER_SAMPLER="0",TOKENIZERS_PARALLELISM="false",
        OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",WANDB_MODE="disabled",
        TENSORBOARD_DIR=str(folder / "tensorboard"),TORCHINDUCTOR_COMPILE_THREADS="1",MAX_JOBS="1",
        MALLOC_ARENA_MAX="2",MALLOC_TRIM_THRESHOLD_="131072")
    cuda = EXPERIMENT / ".venv-train/lib/python3.10/site-packages/nvidia/cu13"
    if (cuda / "bin/nvcc").is_file():
        env.update(CUDA_HOME=str(cuda),PATH=str(cuda / "bin")+os.pathsep+env.get("PATH",""))
    os.environ.update(env)
    with reserve_gpus(teacher) as resources:
        verify_frozen_inputs()
        atomic_json(folder / "gpu_resources.json",{"at":now(),"snapshot":resources})
        process = None
        def interrupted(signum, frame):
            if process is not None and process.poll() is None:
                process.send_signal(signal.SIGTERM)
                try:
                    process.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    process.kill()
            raise SystemExit(128+signum)
        handlers = {s:signal.signal(s,interrupted) for s in [signal.SIGTERM,signal.SIGINT]}
        try:
            with legacy.monitor_cgroup(folder):
                if completed < 100:
                    with (folder / "training.log").open("a",buffering=1) as log:
                        log.write(f"\nFormal single-token OPD launch/resume from step {completed} to 100; no GPU smoke\n")
                        process = subprocess.Popen([str(PYTHON),"-u",str(HERE / "opd_entry.py"),"--config",str(folder / "resolved_config.yaml")],
                            cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                        while process.poll() is None:
                            checkpoint = int(latest.read_text().strip()) if latest.exists() else 0
                            atomic_json(HERE / "queue_status.json",{"status":"training","at":now(),"teacher":teacher,
                                "pid":os.getpid(),"training_pid":process.pid,"latest_checkpoint_step":checkpoint,
                                "target_steps":100,"global_batch_size":63,"run_directory":str(folder)})
                            time.sleep(10)
                        if process.returncode:
                            raise RuntimeError(f"Formal OPD {teacher} exited {process.returncode}; checkpointed progress preserved")
                assert latest.exists() and int(latest.read_text().strip()) == 100
                atomic_json(HERE / "queue_status.json",{"status":"exporting_student","teacher":teacher,"at":now(),"step":100})
                target = legacy.merge_checkpoint(folder,100)
            atomic_json(folder / "complete.json",{"status":"complete","at":now(),"teacher":teacher,"steps":100,
                "global_batch_size":63,"rollouts":6300,"initial_student":str(ORIGINALS["student"]),
                "hf_model":str(target),"dataset_sha256":sha(DATASET)})
            return str(target)
        finally:
            for sig, handler in handlers.items():
                signal.signal(sig,handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute",action="store_true")
    ap.add_argument("--check",action="store_true")
    args = ap.parse_args()
    assert args.execute != args.check, "Choose exactly one of --check / --execute"
    with (HERE / "queue.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            atomic_json(HERE / "queue_status.json",{"status":"cpu_preflight","at":now(),"pid":os.getpid()})
            freeze()
            if args.check:
                return
            exports = {}
            for teacher in SETTINGS["teachers"]:
                exports[teacher] = launch(teacher)
            atomic_json(HERE / "queue_status.json",{"status":"complete","at":now(),"exports":exports,
                "steps_per_run":100,"global_batch_size":63,"same_initial_student":True,"data_sha256":sha(DATASET)})
        except BaseException as error:
            atomic_json(HERE / "queue_status.json",{"status":"failed_or_interrupted","at":now(),"pid":os.getpid(),
                "error":f"{type(error).__name__}: {error}","checkpoints_preserved":True})
            raise


if __name__ == "__main__":
    main()
