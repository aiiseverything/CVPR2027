#!/usr/bin/env python3
"""Dedicated subprocess entry point for an already composed, audited config."""

import argparse
import os
import signal
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(WORKSPACE), str(WORKSPACE / "pre-lab/repo")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    from omegaconf import OmegaConf
    from verl.trainer.main_ppo import run_ppo
    import ray

    cfg = OmegaConf.load(args.config)
    assert cfg.algorithm.adv_estimator == "grpo"
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "vanilla"
    assert cfg.actor_rollout_ref.rollout.n == 8
    assert cfg.data.max_prompt_length == 2048 and cfg.data.max_response_length == 8192
    assert not cfg.algorithm.use_kl_in_reward
    assert not cfg.actor_rollout_ref.actor.use_kl_loss
    assert cfg.actor_rollout_ref.actor.entropy_coeff == 0
    assert not cfg.actor_rollout_ref.actor.calculate_entropy
    os.chdir(WORKSPACE)
    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        run_ppo(cfg)
    finally:
        # Shut down only this process's isolated Ray instance.
        ray.shutdown()


if __name__ == "__main__":
    main()
