#!/usr/bin/env bash
# 6 组推理（2 模型 × 3 条件）在 6 张 GPU 上并行。
#
#   ./probe/run_all.sh             # 全量（各 1000 条）
#   ./probe/run_all.sh --smoke 20  # 冒烟：各 20 条
#
# 每组独立进程 + 独立 GPU；共享 vLLM 编译缓存，避免重复编译。
set -uo pipefail

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
cd "$ROOT"
source "$ROOT/venv/bin/activate" || { echo "✗ venv 不存在"; exit 1; }

# --- 环境修复（本机特有）-----------------------------------------------------
# 1) 系统 nvcc 是 CUDA 12.2，不支持 flashinfer JIT 需要的 --compress-mode；
#    且与 torch 2.13+cu130 的头文件不兼容。用 venv 自带的 CUDA 13.4 toolkit。
export CUDA_HOME="$ROOT/venv/lib/python3.10/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
# 2) 索性禁用 flashinfer sampler —— 它正是触发那段 JIT 编译的地方，且我们不需要它。
export VLLM_USE_FLASHINFER_SAMPLER=0
# 3) 编译缓存必须**每进程独立**。共用同一个 VLLM_CACHE_ROOT 时，
#    6 个进程会争抢 triton cubin / torch_compile_cache，日志里出现
#    FileNotFoundError 并触发反复重编译（实测明显拖慢）。
export OMP_NUM_THREADS=8

LIMIT_ARGS=()
if [ "${1:-}" = "--smoke" ]; then
    LIMIT_ARGS=(--limit "${2:-20}")
    echo ">>> 冒烟模式: 每条件 ${2:-20} 条"
fi

mkdir -p probe/logs probe/out

declare -a PIDS=() NAMES=()
GPU=0
for mk in 4B 9B; do
    for cond in no_image original bbox; do
        name="${mk}__${cond}"
        echo "[$(date '+%T')] 启动 $name -> GPU $GPU"
        CUDA_VISIBLE_DEVICES=$GPU \
        VLLM_CACHE_ROOT="$ROOT/probe/.vllm_cache/$name" \
        nohup python probe/run_infer.py \
            --model-key "$mk" --condition "$cond" "${LIMIT_ARGS[@]}" \
            > "probe/logs/$name.log" 2>&1 &
        PIDS+=($!)
        NAMES+=("$name")
        GPU=$((GPU + 1))
    done
done

echo
echo "[$(date '+%T')] 全部启动，等待完成 ..."
fail=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "  ✓ ${NAMES[$i]}"
    else
        echo "  ✗ ${NAMES[$i]} (见 probe/logs/${NAMES[$i]}.log)"
        fail=1
    fi
done

echo
echo "[$(date '+%T')] 汇总"
python probe/score.py
exit $fail
