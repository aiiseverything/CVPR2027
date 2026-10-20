#!/usr/bin/env python3
"""Summarize real logged rollout budgets and useful/noninformative GRPO groups."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    steps, sources = {}, defaultdict(lambda: defaultdict(float))
    for path in sorted((args.run_dir / "rollouts").glob("*.jsonl")):
        groups = defaultdict(list)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("score") is None:
                continue
            step = int(row["step"])
            uid = row.get("group_uid")
            if not uid:
                raise ValueError(f"Missing group UID in {path}")
            groups[(step, uid)].append(float(row["score"]))
            source = sources[row["audit_source"]]
            source["responses"] += 1
            source["reward_sum"] += float(row["score"])
            for key in ("truncated", "format_valid", "prompt_truncated", "response_tokens", "prompt_tokens"):
                source[key] += float(row.get(key, 0))
        for (step, _uid), values in groups.items():
            if len(values) != 8:
                raise ValueError(f"Step {step} group has {len(values)} responses, expected 8")
            record = steps.setdefault(step, {"groups": 0, "responses": 0, "uniform_reward_groups": 0})
            record["groups"] += 1
            record["responses"] += len(values)
            record["uniform_reward_groups"] += int(max(values) - min(values) < 1e-12)
    report = {"logged_steps": len(steps), "last_logged_step": max(steps, default=0),
              "prompt_exposures": sum(v["groups"] for v in steps.values()),
              "sampled_responses": sum(v["responses"] for v in steps.values()),
              "uniform_reward_groups": sum(v["uniform_reward_groups"] for v in steps.values()),
              "sources": {}}
    for name, values in sources.items():
        n = values["responses"]
        report["sources"][name] = {"responses": int(n), "mean_reward": values["reward_sum"] / n,
                                  **{f"mean_{key}": values[key] / n for key in
                                     ("truncated", "format_valid", "prompt_truncated", "response_tokens", "prompt_tokens")}}
    latest = args.run_dir / "checkpoints/latest_checkpointed_iteration.txt"
    report["last_saved_checkpoint"] = int(latest.read_text().strip()) if latest.exists() else 0
    report["export_complete"] = (args.run_dir / "hf/global_step_300/sharedexpert_export_complete.json").is_file()
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
