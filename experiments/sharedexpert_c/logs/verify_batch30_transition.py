"""Audit the user-authorized batch change without rewriting the old smoke."""
import ast
import difflib
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
EXP = Path(__file__).resolve().parent.parent
ROOT = EXP.parents[1]
TRAINING = EXP / "training"
sys.path[:0] = [str(ROOT), str(ROOT / "pre-lab/repo"), str(TRAINING)]

def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()

def sha(path):
    return sha_bytes(Path(path).read_bytes())

def function_ast(text, name):
    node = next(node for node in ast.parse(text).body if isinstance(node, ast.FunctionDef) and node.name == name)
    return ast.dump(node, include_attributes=False)

def main():
    old_run = EXP / "runs/perception_smoke"
    protocol_path = old_run / "protocol.json"
    old = json.loads(protocol_path.read_text())
    common_path = TRAINING / "config/common.json"
    current = json.loads(common_path.read_text())
    expected = dict(old["protocol"], batch_size=30)
    assert current == expected and old["protocol"]["batch_size"] == 40
    launch_text = (TRAINING / "launch.py").read_text()
    old_launch_text = launch_text.replace(
        '    assert (p["steps"], p["batch_size"], p["group_size"]) == (300, 30, 8)\n'
        '    assert batch * p["group_size"] % len(gpus) == 0\n',
        '    assert (p["steps"], p["batch_size"], p["group_size"]) == (300, 40, 8)\n')
    old_launch_text = old_launch_text.replace(
        '    # Preserve the audited 12,000-row parquet. With shuffle=False, the user\'s\n'
        '    # revised 300 x 30 protocol consumes its first 9,000 rows without rewriting it.\n'
        '    report["dataset"] = check_dataset(WORKSPACE / t["data_file"], None if args.smoke else 12000)\n'
        '    if not args.smoke and report["dataset"].get("rows", 0) < steps * batch:\n'
        '        report["dataset"]["errors"].append("Prepared dataset is shorter than the intended training exposures")\n',
        '    report["dataset"] = check_dataset(WORKSPACE / t["data_file"], None if args.smoke else p["steps"] * p["batch_size"])\n')
    old_launch_sha = sha_bytes(old_launch_text.encode())
    assert old_launch_sha == old["training_code_sha256"]["launch.py"], "Reconstructed original launch must match original smoke hash"
    (EXP / "logs/launch_before_batch30.py").write_text(old_launch_text)
    patch = "".join(difflib.unified_diff(old_launch_text.splitlines(True), launch_text.splitlines(True),
                                       fromfile="launch.py (original validated batch40)", tofile="launch.py (authorized batch30)"))
    (EXP / "logs/batch30_launch.diff").write_text(patch)
    assert function_ast(old_launch_text, "make_overrides") == function_ast(launch_text, "make_overrides")
    unchanged = {}
    for name, old_sha in old["training_code_sha256"].items():
        if name == "launch.py":
            continue
        unchanged[name] = sha(TRAINING / name)
        assert unchanged[name] == old_sha
    assert sha(EXP / "rewards.py") == old["reward_sha256"]
    diff = subprocess.check_output(["git", "-C", str(ROOT / "pre-lab/repo"), "diff", "HEAD", "--"], text=True)
    assert sha_bytes(diff.encode()) == old["repository_diff_sha256"]
    for name, expected_sha in old["training_environment_sha256"].items():
        assert sha(EXP / "logs" / name) == expected_sha
    import verl.trainer.main_ppo
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from launch import make_overrides
    old_overrides = make_overrides(old["protocol"], old["teacher"], 4, old_run, 1, 4)
    new_overrides = make_overrides(current, old["teacher"], 4, old_run, 1, 4)
    assert old_overrides == new_overrides
    with initialize_config_dir(config_dir=str(ROOT / "pre-lab/repo/verl/trainer/config"), version_base=None):
        old_cfg = compose(config_name="ppo_trainer", overrides=old_overrides)
        new_cfg = compose(config_name="ppo_trainer", overrides=new_overrides)
        old_resolved = OmegaConf.to_container(old_cfg, resolve=True)
        new_resolved = OmegaConf.to_container(new_cfg, resolve=True)
        saved_resolved = OmegaConf.to_container(OmegaConf.load(old_run / "resolved_config.yaml"), resolve=True)
        assert old_resolved == new_resolved == saved_resolved
        formal = {}
        for teacher in ("perception", "reasoning", "generalist"):
            t = json.loads((TRAINING / f"config/{teacher}.json").read_text())
            cfg = compose(config_name="ppo_trainer", overrides=make_overrides(current, t, 4, EXP / "runs" / teacher, 300, 30))
            assert cfg.data.train_batch_size == cfg.actor_rollout_ref.actor.ppo_mini_batch_size == 30
            assert cfg.actor_rollout_ref.rollout.n == 8 and cfg.trainer.total_training_steps == 300
            assert not cfg.data.shuffle
            (EXP / "logs" / f"resolved_batch30_{teacher}.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
            formal[teacher] = {"steps": 300, "batch_size": 30, "ppo_mini_batch_size": 30,
                "prepared_rows": 12000, "prompts": 9000, "responses": 72000, "responses_per_rank": 60,
                "selection": "first 9000 prepared rows; shuffle=False; stop at step300"}
    cpu = json.loads((EXP / "logs/cpu-validation.json").read_text())
    for teacher in formal:
        assert cpu[teacher]["hydra_compose"] == "passed" and cpu[teacher]["prompts"] == 9000
        assert cpu[teacher]["responses"] == 72000 and cpu[teacher]["responses_per_rank"] == 60
    assert cpu["gpu_smoke"] == "not_performed_by_this_script"
    receipt = {"passed": True, "effective_smoke_overrides_identical": True,
        "effective_smoke_resolved_config_identical": True,
        "old_smoke_protocol_sha256": sha(protocol_path), "common_sha256": sha(common_path),
        "old_launch_sha256": old_launch_sha, "new_launch_sha256": sha(TRAINING / "launch.py"),
        "cpu_validation_log_sha256": sha(EXP / "logs/batch30_cpu_validation.log"),
        "cpu_validation_report_sha256": sha(EXP / "logs/cpu-validation.json"),
        "validation_script_sha256": sha(TRAINING / "validate_cpu.py"),
        "audit_script_sha256": sha(__file__), "launch_diff_sha256": sha(EXP / "logs/batch30_launch.diff"),
        "unchanged_training_code_sha256": unchanged, "reward_sha256": old["reward_sha256"],
        "repository_patch_sha256": old["repository_diff_sha256"],
        "training_environment_sha256": old["training_environment_sha256"],
        "common_change": {"batch_size": {"from": 40, "to": 30}},
        "launch_change_scope": ["Explicit formal protocol guard 40 to30", "Response count divisible by four ranks",
             "Separate retained prepared12000 rows from effective9000 exposures; do not rewrite dataset"],
        "formal_teachers": formal,
        "full_batch30_gpu_step_status": "pending",
        "old_smoke_status": "Preserved original4prompt GPU smoke; no claim it ran batch30",
        "gpu_numeric_code_changed": False}
    target = EXP / "logs/batch30_transition_validation.json"
    target.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(receipt, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
