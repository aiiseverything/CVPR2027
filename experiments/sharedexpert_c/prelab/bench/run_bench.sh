#!/usr/bin/env bash
# 7 个 benchmark × 2 模型的全量评测编排。
#
#   推理: VLMEvalKit API 模式 -> 本地 vLLM OpenAI 服务端（实测比它的本地 vLLM 路径快约 400 倍）
#   判分: 用用户提供的 DeepSeek-V4.1-Flash（reasoning 已关，直接出分）
#
#   ./bench/run_bench.sh start     # 起服务端并跑全部作业（后台）
#   ./bench/run_bench.sh status
#   ./bench/run_bench.sh stop
#   ./bench/run_bench.sh judge     # 推理完成后判分
#
# GPU 分配（按实测吞吐加权：9B 慢，给更多卡）
#   GPU 0,1,2 -> 4B  端口 8110,8111,8112
#   GPU 3,4,5,6 -> 9B 端口 8113,8114,8115,8116
set -uo pipefail

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
VK="$ROOT/vlmevalkit"
OUT="$ROOT/vlmeval_out"
LOGS="$ROOT/logs/bench"
PIDDIR="$ROOT/logs/bench/pids"
mkdir -p "$LOGS" "$PIDDIR" "$OUT"

BENCHMARKS=(MME MathVista_MINI MathVerse_MINI MMStar MMBench_DEV_EN_V11 MathVision_MINI MMMU_DEV_VAL)

declare -A SRV_PORT=( [4B0]=8110 [4B1]=8111 [4B2]=8112 [9B0]=8113 [9B1]=8114 [9B2]=8115 [9B3]=8116 )
declare -A SRV_GPU=(  [4B0]=0 [4B1]=1 [4B2]=2 [9B0]=3 [9B1]=4 [9B2]=5 [9B3]=6 )
declare -A SRV_MDL=(  [4B0]=Qwen3.5-4B [4B1]=Qwen3.5-4B [4B2]=Qwen3.5-4B \
                      [9B0]=Qwen3.5-9B [9B1]=Qwen3.5-9B [9B2]=Qwen3.5-9B [9B3]=Qwen3.5-9B )

source "$ROOT/venv/bin/activate"
export CUDA_HOME="$ROOT/venv/lib/python3.10/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export OMP_NUM_THREADS=8
export LMUData="$ROOT/lmudata"
export QWEN35_PROJ="$ROOT"
export PYTHONUNBUFFERED=1

judge_key() { grep '^PARATERA_API_KEY=' "$VK/.env" | cut -d= -f2; }

start_servers() {
    for key in "${!SRV_PORT[@]}"; do
        port="${SRV_PORT[$key]}"; gpu="${SRV_GPU[$key]}"; mdl="${SRV_MDL[$key]}"
        if curl -sS -m 3 "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            echo "  端口 $port 已在运行，跳过"; continue
        fi
        echo "  启动 $mdl -> GPU $gpu, 端口 $port"
        CUDA_VISIBLE_DEVICES="$gpu" setsid nohup vllm serve "$ROOT/models/$mdl" \
            --served-model-name "$mdl" --port "$port" \
            --max-model-len 40960 --gpu-memory-utilization 0.90 --max-num-seqs 28 \
            --limit-mm-per-prompt '{"image": 8}' --trust-remote-code \
            > "$LOGS/vllm_$key.log" 2>&1 < /dev/null &
        echo $! > "$PIDDIR/vllm_$key.pid"
    done
    echo "等待服务端就绪（最多 20 分钟）..."
    for i in $(seq 1 80); do
        ready=0
        for port in "${SRV_PORT[@]}"; do
            curl -sS -m 3 "http://localhost:$port/v1/models" >/dev/null 2>&1 && ready=$((ready+1))
        done
        [ "$ready" -eq "${#SRV_PORT[@]}" ] && { echo "  全部 $ready 个服务端就绪 ($((i*15))s)"; return 0; }
        sleep 15
    done
    echo "  ✗ 超时，仅 $ready/${#SRV_PORT[@]} 就绪"
    return 1
}

# 把 7 个 benchmark 轮流分给某模型的若干服务端副本。
# 返回的是**副本序号**（0..nrep-1），因为 SRV_PORT 的键是 4B0/4B1/.../9B0/.../9B3。
job_server() {
    local idx="$1" nrep="$2"
    echo $(( idx % nrep ))
}

run_jobs() {
    local mdl="$1" nrep="$2" tag="$3"
    for i in "${!BENCHMARKS[@]}"; do
        bench="${BENCHMARKS[$i]}"
        srv="$(job_server "$i" "$nrep")"
        port="${SRV_PORT[${tag}${srv}]}"
        log="$LOGS/${tag}_${bench}.log"
        echo "  [$mdl] $bench -> 端口 $port"
        (
          cd "$VK"
          python run.py \
            --model "$mdl" --model-class GPT4V \
            --base-url "http://localhost:$port/v1" --key EMPTY \
            --data "$bench" --work-dir "$OUT" \
            --mode infer --api-nproc 28 --retry 3 --timeout 1800 \
            --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 20 \
            --reuse
        ) > "$log" 2>&1 &
        echo $! > "$PIDDIR/infer_${tag}_${bench}.pid"
    done
}

case "${1:-status}" in
  start|dispatch)
    # 只在服务端尚未运行时清空输出；否则视为续跑，保留已有结果
    any_up=0
    for port in "${SRV_PORT[@]}"; do
        curl -sS -m 3 "http://localhost:$port/v1/models" >/dev/null 2>&1 && any_up=1
    done
    if [ "$any_up" -eq 0 ]; then
        rm -rf "$OUT"
        echo "[$(date '+%T')] 冷启动，已清空 $OUT"
    else
        echo "[$(date '+%T')] 检测到服务端已在运行 —— 续跑模式，保留已有结果"
    fi
    rm -f "$PIDDIR"/*.pid
    echo "[$(date '+%T')] 启动服务端"
    start_servers || exit 1
    echo "[$(date '+%T')] 下发推理作业"
    run_jobs "Qwen3.5-4B" 3 "4B"
    run_jobs "Qwen3.5-9B" 4 "9B"
    echo "[$(date '+%T')] 全部作业已下发（14 个）"
    ;;
  judge)
    KEY="$(judge_key)"
    for i in "${!BENCHMARKS[@]}"; do
        bench="${BENCHMARKS[$i]}"
        for mdl in Qwen3.5-4B Qwen3.5-9B; do
            echo "  判分 $mdl / $bench"
            ( cd "$VK" && python run.py \
                --model "$mdl" --model-class GPT4V --base-url http://localhost:8110/v1 --key EMPTY \
                --data "$bench" --work-dir "$OUT" --mode eval --reuse \
                --judge DeepSeek-V4.1-Flash \
                --judge-base-url https://llmapi.paratera.com/v1 \
                --judge-key "$KEY" \
                --judge-args '{"reasoning_effort":"none"}' \
              ) > "$LOGS/judge_${mdl}_${bench}.log" 2>&1
        done
    done
    ;;
  status)
    echo "=== 服务端 ==="
    for key in "${!SRV_PORT[@]}"; do
        port="${SRV_PORT[$key]}"
        if curl -sS -m 3 "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            n=$(grep -c "Running:" "$LOGS/vllm_$key.log" 2>/dev/null || echo 0)
            thr=$(grep -oE "Avg generation throughput: [0-9.]+" "$LOGS/vllm_$key.log" 2>/dev/null | tail -1)
            echo "  $key (端口 $port, GPU ${SRV_GPU[$key]}): 运行中  $thr"
        else echo "  $key (端口 $port): 未运行"; fi
    done
    echo "=== 作业 ==="
    for i in "${!BENCHMARKS[@]}"; do
        bench="${BENCHMARKS[$i]}"
        for tag in 4B 9B; do
            f="$LOGS/${tag}_${bench}.log"
            [ -e "$f" ] || continue
            prog=$(grep -oE "[0-9]+/[0-9]+ \[" "$f" 2>/dev/null | tail -1 | tr -d ' [')
            done_=$(grep -c "Run Summary" "$f" 2>/dev/null || echo 0)
            printf "  %-5s %-22s %s\n" "$tag" "$bench" "${prog:-未开始}$([ "$done_" -gt 0 ] && echo '  完成')"
        done
    done
    ;;
  stop)
    for f in "$PIDDIR"/*.pid; do
        [ -e "$f" ] || continue
        p=$(cat "$f"); kill "$p" 2>/dev/null
    done
    sleep 5
    echo "已下发停止信号（vLLM 子进程可能需手动清理：pkill -f 'vllm serve'）"
    ;;
  *) echo "用法: $0 {start|status|judge|stop}"; exit 1;;
esac
