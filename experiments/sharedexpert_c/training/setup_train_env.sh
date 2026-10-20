#!/usr/bin/env bash
# Only modifies this experiment's dedicated virtual environment.
set -euo pipefail
TRAINING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT_DIR="$(dirname "$TRAINING_DIR")"
WORKSPACE_DIR="$(cd "$TRAINING_DIR/../../.." && pwd)"
TRAIN_VENV="${SHAREDEXPERT_TRAIN_VENV:-$EXPERIMENT_DIR/.venv-train}"
mkdir -p "$EXPERIMENT_DIR/logs"
if [[ "$TRAIN_VENV" == "$WORKSPACE_DIR/pre-lab/venv" ]]; then
  echo 'Refusing to modify the existing inference environment.' >&2
  exit 2
fi
if [[ ! -x "$TRAIN_VENV/bin/python" ]]; then
  python3 -m venv "$TRAIN_VENV"
fi
# PyTorch's extension builder discovers ninja on PATH, even when invoked using
# an absolute Python path. Keep all build tools in this dedicated environment.
export PATH="$TRAIN_VENV/bin:$PATH"
"$TRAIN_VENV/bin/python" -m pip install --upgrade 'pip>=25'
# The repository's tested freeze uses transformers 5.5 despite vLLM 0.18's
# stale <5 package metadata. Install its declared dependency graph first, then
# deliberately reproduce the repository's transformer version. Runtime import
# and GPU smoke checks remain mandatory; record rather than hide the mismatch.
grep -v '^transformers==' "$TRAINING_DIR/requirements-core.txt" > "$EXPERIMENT_DIR/logs/requirements-stage1.txt"
"$TRAIN_VENV/bin/python" -m pip install -r "$EXPERIMENT_DIR/logs/requirements-stage1.txt"
"$TRAIN_VENV/bin/python" -m pip install 'transformers==5.5.0'
# Build FlashAttention against the pinned torch/CUDA ABI, not against another
# isolated build environment's torch. MAX_JOBS bounds host compilation load.
# FA 2.8.3 explicitly reads FLASH_ATTN_CUDA_ARCHS (not TORCH_CUDA_ARCH_LIST).
# L20 is sm89 and can execute the sm80 cubins. Avoid unused sm90/100/120 builds.
# Official 2.8.3 has no torch2.10/cp310 Linux wheel, so compile against this ABI.
"$TRAIN_VENV/bin/python" "$TRAINING_DIR/build_flash_attention.py" --qwen-restricted --qwen-heads-only
"$TRAIN_VENV/bin/python" -m pip install --no-deps -e "$WORKSPACE_DIR/pre-lab/repo"
"$TRAIN_VENV/bin/python" "$TRAINING_DIR/check_dependency_metadata.py"
"$TRAIN_VENV/bin/python" -m pip freeze > "$EXPERIMENT_DIR/logs/training-environment.freeze.txt"
"$TRAIN_VENV/bin/python" "$TRAINING_DIR/preflight.py" --imports-only
