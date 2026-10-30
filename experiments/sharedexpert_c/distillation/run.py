#!/usr/bin/env python3
"""Worker-process entry point for the pinned VERL domain-OPD plugin."""
import argparse
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
    import ray
    from experiments.sharedexpert_c.distillation import verl_plugin
    from verl.trainer.main_ppo import run_ppo
    cfg = OmegaConf.load(args.config)
    assert cfg.algorithm.adv_estimator == "mopd_targets"
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode == "mopd_reverse_kl"
    assert cfg.actor_rollout_ref.actor.ppo_epochs == 1
    assert not cfg.algorithm.use_kl_in_reward and not cfg.actor_rollout_ref.actor.use_kl_loss

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        run_ppo(cfg)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
