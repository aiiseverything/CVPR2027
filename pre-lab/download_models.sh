#!/usr/bin/env bash
# 下载 Qwen3.5-4B / Qwen3.5-9B (instruct, 原生多模态) 到 pre-lab/models/
set -uo pipefail

export HF_HOME=/home/ml-user/workdir/cache/hf
export HF_XET_HIGH_PERFORMANCE=1

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
mkdir -p "$ROOT/models" "$ROOT/logs"

fail=0
for repo in Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B; do
    name="${repo##*/}"
    echo "=============================================="
    echo "[$(date '+%F %T')] 开始下载 $repo -> models/$name"
    echo "=============================================="
    if hf download "$repo" --local-dir "$ROOT/models/$name"; then
        echo "[$(date '+%F %T')] ✓ $name 完成"
    else
        echo "[$(date '+%F %T')] ✗ $name 失败 (exit=$?)"
        fail=1
    fi
done

echo "[$(date '+%F %T')] 全部结束, fail=$fail"
exit $fail
