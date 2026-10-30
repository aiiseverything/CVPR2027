#!/usr/bin/env python3
"""Run mixed-domain GRPO using the existing isolated VERL teacher entry point."""
import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(WORKSPACE), str(WORKSPACE / "pre-lab/repo")]

# The established runner asserts GRPO/n=8/2048/8192/no-KL/no-entropy and shuts
# down only its own isolated Ray instance on exit. It does not select a teacher.
from experiments.sharedexpert_c.training.run import main

if __name__ == "__main__":
    main()
