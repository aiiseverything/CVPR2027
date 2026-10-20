#!/usr/bin/env python3
"""Require actual nonzero RL learning, rollout budgets and checkpoint/export evidence."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()
    protocol = json.loads((run / "protocol.json").read_text())
    config = OmegaConf.load(run / "resolved_config.yaml")
    groups, group_outputs, rows = defaultdict(list), defaultdict(list), []
    for path in sorted((run / "rollouts").glob("*.jsonl")):
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if int(row["step"]) == 1:
                assert row.get("group_uid"), "Missing real group UID"
                groups[row["group_uid"]].append(float(row["score"]))
                assert isinstance(row.get("output"), str), "Missing decoded rollout output"
                group_outputs[row["group_uid"]].append(row["output"])
                rows.append(row)
    assert len(groups) == 4 and len(rows) == 32
    assert all(len(scores) == 8 for scores in groups.values())
    assert all(0 < int(row["response_tokens"]) <= 8192 for row in rows)
    assert all(0 < int(row["prompt_tokens"]) <= 2048 for row in rows)
    nonuniform = sum(max(scores) - min(scores) > 1e-12 for scores in groups.values())
    assert nonuniform > 0, "Smoke has no useful group; repeat with different real train examples"
    events = EventAccumulator(str(run / "tensorboard"), size_guidance={"scalars": 0}).Reload()
    scalars = {tag: [event.value for event in events.Scalars(tag) if event.step == 1]
               for tag in events.Tags()["scalars"]}
    required = ["actor/pg_loss", "actor/grad_norm", "rollout_corr/rollout_is_mean",
                "rollout_corr/rollout_is_min", "rollout_corr/rollout_is_max"]
    for tag in required:
        assert scalars.get(tag) and all(math.isfinite(value) for value in scalars[tag]), tag
    assert max(scalars["actor/grad_norm"]) > 0
    assert not config.actor_rollout_ref.actor.use_kl_loss and not config.algorithm.use_kl_in_reward
    assert config.actor_rollout_ref.actor.entropy_coeff == 0
    assert not config.actor_rollout_ref.actor.calculate_entropy
    assert not config.trainer.reuse_rollout_log_probs
    assert config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu == 1
    assert not config.actor_rollout_ref.actor.use_dynamic_bsz
    assert not config.actor_rollout_ref.rollout.log_prob_use_dynamic_bsz
    assert "actor/entropy" not in scalars
    checkpoint = run / "checkpoints/global_step_1/actor"
    export = run / "hf/global_step_1"
    assert (checkpoint / "fsdp_config.json").is_file()
    assert len(list(checkpoint.glob("model_world_size_4_rank_*.pt"))) == 4
    assert len(list(checkpoint.glob("optim_world_size_4_rank_*.pt"))) == 4
    assert (export / "sharedexpert_export_complete.json").is_file()
    assert list(export.glob("*.safetensors"))
    resources = json.loads((run / "resource_usage.json").read_text())
    assert resources["memory_peak_bytes"] > 0 and len(resources["gpu_peak_used_bytes"]) == 4
    assert resources["working_set_peak_bytes"] > 0
    assert resources["memory_events_delta"]["oom"] == 0
    assert resources["memory_events_delta"]["oom_kill"] == 0
    replica_seeds = protocol["rollout_replica_seeds"]
    assert replica_seeds == {str(rank): protocol["protocol"]["seed"] + rank for rank in range(4)}
    report = {"passed": True, "run_dir": str(run), "step": 1, "prompts": 4, "responses": 32,
              "group_size": 8, "nonuniform_reward_groups": nonuniform,
              "unique_full_outputs_per_group": {uid: len(set(outputs))
                                                for uid, outputs in group_outputs.items()},
              "rewards_per_group": dict(groups),
              "metrics": {tag: scalars[tag] for tag in required},
              "kl_disabled": True, "entropy_disabled": True, "old_logprobs_recomputed": True,
              "independent_single_sequence_microbatches": True,
              "rollout_replica_seeds": replica_seeds,
              "checkpoint": str(checkpoint), "hf_export": str(export), "resources": resources,
              "reward_sha256": protocol["reward_sha256"],
              "training_code_sha256": protocol["training_code_sha256"],
              "repository_patch_sha256": protocol["repository_diff_sha256"],
              "training_environment_sha256": protocol["training_environment_sha256"]}
    target = Path(__file__).resolve().parent.parent / "logs/smoke_validation.json"
    target.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
