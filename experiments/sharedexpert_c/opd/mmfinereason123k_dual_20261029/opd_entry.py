"""Execute one frozen formal OPD config; no preliminary GPU training invocation."""
import argparse
import os
import signal
import sys
from pathlib import Path
from opd_config import ROOT, REPO, validate_config

sys.path[:0] = [str(ROOT), str(REPO)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    args = ap.parse_args()
    from omegaconf import OmegaConf
    import ray
    from verl.trainer.main_ppo import run_ppo
    cfg = OmegaConf.load(args.config)
    validate_config(cfg)
    os.chdir(ROOT)
    def interrupted(signum, frame):
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        run_ppo(cfg)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
