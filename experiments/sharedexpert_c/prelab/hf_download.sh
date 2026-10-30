#!/usr/bin/env bash
# 从 hf-mirror 拉基座模型到 pre-lab/models/
set -uo pipefail
export PATH=/root/miniconda3/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/.cache/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_DISABLE_XET=1
ROOT=/root/autodl-fs/CVPR2027/pre-lab
mkdir -p "$ROOT/models" "$ROOT/logs"
for repo in Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B; do
  name=${repo##*/}
  echo "[$(date '+%F %T')] START $repo"
  if hf download "$repo" --local-dir "$ROOT/models/$name"; then
    echo "[$(date '+%F %T')] OK $repo size=$(du -sh "$ROOT/models/$name" 2>/dev/null | cut -f1)"
  else
    echo "[$(date '+%F %T')] FAIL $repo exit=$?"
  fi
done
echo "[$(date '+%F %T')] ALL DONE"
