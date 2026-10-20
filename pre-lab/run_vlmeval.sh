#!/usr/bin/env bash
# 运行 VLMEvalKit。
#
#   ./run_vlmeval.sh infer  4B MathVision_MINI 0     # 冒烟：某 benchmark 单卡
#   ./run_vlmeval.sh infer  9B "MME,MmmStar" 1       # 多 benchmark
#   ./run_vlmeval.sh eval  4B MathVision_MINI        # 判分（需要 judge API）
#
# 参数: 模式( infer | eval | all ) 模型(4B|9B) 数据(逗号分隔) GPU编号
set -uo pipefail

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
cd "$ROOT/vlmevalkit"
source "$ROOT/venv/bin/activate"

MODE="${1:-infer}"
MODEL_KEY="${2:-4B}"
DATA="${3:-MathVision_MINI}"
GPU="${4:-0}"

case "$MODEL_KEY" in
  4B) MODEL="Qwen3.5-4B" ;;
  9B) MODEL="Qwen3.5-9B" ;;
  *)  MODEL="$MODEL_KEY" ;;
esac

# --- 本机环境修复（与 probe 实验一致）-----------------------------------
export CUDA_HOME="$ROOT/venv/lib/python3.10/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=8
# 数据用我们已下好并校验过的目录（绕过该站过期的 https 证书）
export LMUData="$ROOT/lmudata"
export CUDA_VISIBLE_DEVICES="$GPU"
export QWEN35_PROJ="$ROOT"
# 不加这个的话 stdout 被缓冲，看不到 tqdm 进度，无法判断吞吐
export PYTHONUNBUFFERED=1

DATA_ARGS=(); IFS=',' read -ra ARR <<< "$DATA"; for d in "${ARR[@]}"; do DATA_ARGS+=("$d"); done

echo "[$(date '+%T')] mode=$MODE model=$MODEL gpu=$GPU data=${DATA_ARGS[*]}"
python run.py \
    --model "$MODEL" \
    --data "${DATA_ARGS[@]}" \
    --work-dir "$ROOT/vlmeval_out" \
    --mode "$MODE" \
    --verbose
rc=$?
echo "[$(date '+%T')] 结束 exit=$rc"
exit $rc
