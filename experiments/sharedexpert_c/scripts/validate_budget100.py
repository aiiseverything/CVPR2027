#!/usr/bin/env python3
"""CPU-only audit of the authorized 300 -> 100 endpoint transition."""
import ast
import collections
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path[:0] = [str(HERE), str(HERE / "training"), str(ROOT), str(ROOT / "pre-lab/repo")]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(value, prefix=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten(item, prefix + ("." if prefix else "") + key))
        return result
    return {prefix: value}


def main():
    import pyarrow.parquet as pq
    import torch
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    import run_experiment as driver
    import launch
    import launch_100

    driver.verify_preparation()
    base = json.loads((HERE / "training/config/common.json").read_text())
    budget = driver.execution_budget()
    assert budget == dict(base, steps=100)
    old_ast = ast.parse((HERE / "training/launch.py").read_text())
    new_ast = ast.parse((HERE / "training/launch_100.py").read_text())
    for name in ("make_overrides", "gpu_check", "merge_checkpoint", "monitor_cgroup"):
        nodes = [next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
                 for tree in (old_ast, new_ast)]
        assert ast.dump(nodes[0], include_attributes=False) == ast.dump(nodes[1], include_attributes=False)
    differences, data_audit, artifacts = {}, {}, []
    output = HERE / "logs/budget100"
    output.mkdir(exist_ok=True)
    expected_counts = {
        "perception": {"tallyqa": 1200, "clevr": 1200, "refcoco": 600},
        "reasoning": {"geometry3k": 1200, "mmk12": 600, "virl39k": 1200},
        "generalist": {"vqav2": 1200, "docvqa": 150, "chartqa": 150, "geometry3k": 240,
                      "mmk12": 120, "virl39k": 240, "tallyqa": 240, "clevr": 240,
                      "refcoco": 120, "if_format": 300},
    }
    for role in driver.ROLES:
        teacher = json.loads((HERE / f"training/config/{role}.json").read_text())
        with initialize_config_dir(config_dir=str(ROOT / "pre-lab/repo/verl/trainer/config"), version_base=None):
            before = compose(config_name="ppo_trainer", overrides=launch.make_overrides(base, teacher, 4, HERE / "runs" / role, 300, 30))
            after = compose(config_name="ppo_trainer", overrides=launch_100.make_overrides(budget, teacher, 4, HERE / "runs" / role, 100, 30))
        a, b = [flatten(OmegaConf.to_container(cfg, resolve=True)) for cfg in (before, after)]
        differences[role] = {key: {"before": a.get(key), "after": b.get(key)}
                             for key in a.keys() | b.keys() if a.get(key) != b.get(key)}
        assert differences[role] == {"trainer.total_training_steps": {"before": 300, "after": 100}}
        assert after.actor_rollout_ref.actor.optim.lr_scheduler_type == "constant"
        assert after.actor_rollout_ref.actor.optim.lr_warmup_steps == 10
        target = output / f"{role}.yaml"
        target.write_text(OmegaConf.to_yaml(after, resolve=True))
        artifacts.append(target)
        counts, records, seen = collections.Counter(), set(), 0
        file = ROOT / teacher["data_file"]
        for batch in pq.ParquetFile(file).iter_batches(batch_size=64, columns=["extra_info"], use_threads=False):
            for row in batch.to_pylist():
                if seen == 3000:
                    break
                info = row["extra_info"]
                record = info["record_id"]
                counts[record.split(":", 1)[0]] += 1
                records.add(record)
                seen += 1
            if seen == 3000:
                break
        assert dict(counts) == expected_counts[role], (role, counts)
        data_audit[role] = {"rows_consumed": seen, "responses": seen * 8,
                            "source_counts": dict(counts), "unique_records": len(records),
                            "prepared_file_sha256": sha(file)}
    artifacts += [HERE / name for name in (
        "training/launch.py", "training/launch_100.py", "training/config/common.json",
        "training/config/execution_budget.json", "scripts/validate_budget100.py")]
    assert not torch.cuda.is_initialized()
    receipt = {
        "passed": True, "validated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "effective_config_diffs": differences, "constant_lr_with_unchanged_warmup": True,
        "data_audit": data_audit, "gpu_started": False,
        "validated_artifacts": {str(path.relative_to(HERE)): sha(path) for path in artifacts},
    }
    (HERE / "logs/budget100_validation.json").write_text(json.dumps(receipt, indent=2) + "\n")
    driver.verify_execution_budget()

    checks = []
    for failure in (None, "checkpoint_missing", "memory_check_failed"):
        calls, states = [], []
        def command(args, stage, role=None):
            args = list(map(str, args))
            calls.append({"args": args, "stage": stage, "role": role})
            if failure == "memory_check_failed" and "--supervise" in args:
                raise RuntimeError("injected memory-check failure")
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(driver, "RUNTIME", Path(temp)), \
             patch.object(driver, "STATE", Path(temp) / "state.json"), \
             patch.object(driver, "command", command), \
             patch.object(driver, "write_state", lambda *a, **kw: states.append((a, kw))), \
             patch.object(driver, "verify_preparation", lambda: None), \
             patch.object(driver, "ensure_monitor", lambda **kw: None), \
             patch.object(driver, "checkpoint_is_complete", lambda *a: failure != "checkpoint_missing"):
            try:
                driver.run()
            except RuntimeError:
                assert failure is not None
            else:
                assert failure is None
        executions = [item["role"] for item in calls if "--execute" in item["args"]]
        if failure:
            assert not executions and states[-1][0][0] == "failed"
        else:
            assert executions == ["reasoning", "generalist"]
            merge = next(i for i, item in enumerate(calls) if "--merge-only" in item["args"])
            check = next(i for i, item in enumerate(calls) if "--supervise" in item["args"])
            first_train = next(i for i, item in enumerate(calls) if "--execute" in item["args"])
            assert merge < check < first_train
            evaluations = [item for item in calls if item["stage"] == "evaluation"]
            assert len(evaluations) == 3 and all("global_step_100" in item["args"][3] for item in evaluations)
        checks.append({"case": failure or "P100_export_check_R100_G100_eval100", "passed": True})
    report = {"passed": True, "driver_sha256": sha(HERE / "run_experiment.py"),
              "checks": checks, "gpu_started": False}
    (HERE / "logs/budget100_driver_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"budget_validation": True, "data": data_audit, "driver_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
