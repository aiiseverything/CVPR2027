#!/usr/bin/env python3
"""Stop the existing perception run at completed checkpoint 100, then take over.

No change is injected into the running trainer. Exact process identities, owned
process-tree cleanup, the driver flock, and idle GPUs gate a single takeover.
The detached watcher never restarts a failed or user-stopped experiment.
"""
import argparse
import collections
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import controller_handoff as core
import run_experiment as driver

SELF = Path(__file__).resolve()


class Context(core.Context):
    def __init__(self, here=core.HERE):
        super().__init__(here)
        self.home = self.runtime / "budget100_handoff"
        self.plan = self.home / "plan.json"
        self.state = self.home / "status.json"
        self.events = self.home / "events.jsonl"
        self.lock = self.lock.with_name(self.lock.name.replace("sharedexpert-handoff-", "sharedexpert-budget100-"))

    def hashes(self):
        result = super().hashes()
        for path in (SELF, self.here / "training/launch_100.py",
                     self.here / "training/config/execution_budget.json",
                     self.here / "logs/budget100_validation.json"):
            result[str(path)] = core.sha(path)
        return result


def checkpoint_ready(ctx):
    run = ctx.here / "runs/perception"
    tracker = run / "checkpoints/latest_checkpointed_iteration.txt"
    if not tracker.exists() or tracker.read_text().strip() != "100":
        return False
    checkpoint = run / "checkpoints/global_step_100"
    required = [checkpoint / "data.pt", checkpoint / "actor/fsdp_config.json"]
    required += [checkpoint / "actor" / f"{kind}_world_size_4_rank_{rank}.pt"
                 for kind in ("model", "optim", "extra_state") for rank in range(4)]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        rows = [json.loads(line) for line in (run / "rollouts/100.jsonl").read_text().splitlines()]
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    groups = collections.Counter(row.get("group_uid") for row in rows)
    return len(rows) == 240 and len(groups) == 30 and set(groups.values()) == {8} and None not in groups


def unexpected_progress(ctx):
    evidence = core.reasoning_progress(ctx)
    for role in ("perception", "generalist"):
        limit = 100 if role == "perception" else 0
        run = ctx.here / "runs" / role
        for path in (run / "rollouts").glob("*.jsonl"):
            if path.stem.isdigit() and int(path.stem) > limit:
                evidence.append(str(path))
        tracker = run / "checkpoints/latest_checkpointed_iteration.txt"
        if tracker.exists() and int(tracker.read_text()) > limit:
            evidence.append(str(tracker))
    return evidence


def decision(ctx, plan):
    old = plan["old_driver"]
    if not core.alive(old):
        return "abort", "Original driver exited or its identity changed; never revive automatically"
    state = core.read(ctx.runtime / "experiment_state.json", {})
    if state.get("driver_pid") != old["pid"] or state.get("stage") in ("failed", "interrupted", "complete"):
        return "abort", "Experiment stopped or another driver took over"
    if unexpected_progress(ctx):
        return "abort", "A teacher progressed beyond the eligible 100-step boundary"
    if state.get("stage") != "training" or state.get("teacher") != "perception":
        return "abort", "Original driver is no longer training perception"
    if not checkpoint_ready(ctx):
        return "wait", "Waiting for checkpoint 100: four model/optimizer/RNG shards, data.pt and 240 rollout rows"
    return "ready", "Perception checkpoint and rollout 100 complete; preserve them and export step 100"


def perception_children(ctx, plan, cleaned=()):
    children = [p for p in core.processes().values() if p["ppid"] == plan["old_driver"]["pid"]]
    expected = [str(ctx.launcher), "perception", "--gpus", "3,4,5,6", "--execute"]
    for child in children:
        if (child["cwd"] != str(ctx.root) or child["pgrp"] != child["pid"]
                or child["session"] != child["pid"] or child["argv"][1:] != expected):
            raise RuntimeError("Unexpected child at the perception boundary; no signal sent")
    if len(children) > 1:
        raise RuntimeError("Multiple perception launchers; ownership is ambiguous")
    state = core.read(ctx.runtime / "experiment_state.json", {})
    recorded = state.get("child_pid")
    known = {p["pid"] for p in children} | {p["pid"] for p in cleaned if not core.alive(p)}
    if recorded is not None and recorded not in known:
        raise RuntimeError("Saved perception child disagrees with actual process ancestry")
    return children


def launch(ctx):
    # The running watcher holds this lock for its lifetime. Check its exact
    # identity first so repeated launch requests return the existing instance.
    current = status(ctx)
    if current["watcher_running"]:
        return {"already_running": True, "pid": current["pid"]}
    with core.lock_file(ctx.lock):
        plan = core.read(ctx.plan)
        if not plan:
            raise RuntimeError("Prepare the 100-step handoff first")
        core.verify_hashes(ctx, plan)
        state = core.read(ctx.state, {})
        if state.get("stage") in core.TERMINAL or state.get("irreversible_started") or state.get("launch_requested"):
            raise RuntimeError("One-shot transition is terminal; do not repeat automatically")
        if state.get("pid"):
            p = core.identity(state["pid"])
            if p and p["start_ticks"] == state.get("start_ticks") and str(SELF) in p["argv"]:
                return {"already_running": True, "pid": p["pid"]}
        with (ctx.home / "watcher.log").open("ab") as log:
            child = subprocess.Popen([sys.executable, str(SELF), "run", "--plan", str(ctx.plan)], cwd=ctx.root,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED="1",
                    OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
        identity = core.identity(child.pid)
        if not identity:
            raise RuntimeError("100-step watcher exited during launch")
        state.update(stage="launching", pid=child.pid, start_ticks=identity["start_ticks"],
                     heartbeat_epoch=time.time(), at=core.now())
        core.atomic(ctx.state, state)
        return {"pid": child.pid, "log": str(ctx.home / "watcher.log")}


def status(ctx):
    state = core.read(ctx.state, {"stage": "not_prepared"})
    p = core.identity(state.get("pid", -1))
    state["watcher_running"] = bool(p and p["state"] not in ("Z", "X")
        and p["start_ticks"] == state.get("start_ticks") and str(SELF) in p["argv"])
    state["heartbeat_age_seconds"] = max(0, time.time() - state.get("heartbeat_epoch", 0))
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "launch", "run", "status", "cancel"))
    parser.add_argument("--old-pid", type=int)
    parser.add_argument("--old-start-ticks", type=int)
    parser.add_argument("--plan", type=Path, help="Internal detached-watcher plan path")
    args = parser.parse_args()
    if args.plan:
        plan = core.read(args.plan)
        ctx = Context(plan["here"])
        if ctx.plan != args.plan.resolve():
            raise RuntimeError("Plan path does not match the bound workspace")
    else:
        ctx = Context()
    if args.action == "prepare":
        if args.old_pid is None or args.old_start_ticks is None:
            parser.error("prepare requires the original PID and start_ticks")
        driver.verify_preparation()
        driver.verify_execution_budget()
        result = core.prepare(ctx, args.old_pid, args.old_start_ticks)
        outcome, reason = decision(ctx, result)
        if outcome == "abort":
            raise RuntimeError(reason)
    elif args.action == "launch":
        result = launch(ctx)
    elif args.action == "run":
        core.run(ctx, decision_fn=decision, children_fn=perception_children,
                 progress_fn=unexpected_progress)
        return
    elif args.action == "status":
        result = status(ctx)
    else:
        state = status(ctx)
        if state["watcher_running"]:
            p = core.identity(state["pid"])
            if not p or p["start_ticks"] != state["start_ticks"] or str(SELF) not in p["argv"]:
                raise RuntimeError("Watcher identity changed")
            core.send_exact(p, signal.SIGTERM)
            result = {"cancel_requested": True, "pid": p["pid"]}
        else:
            result = {"cancelled": False, "reason": "No matching live watcher"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
