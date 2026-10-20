#!/usr/bin/env bash
# 端到端流水线：等推理跑完 -> 自动补跑失败条目 -> 判分 -> 出汇总。
# 独立会话运行，关终端/断 SSH/关电脑都不受影响。
#
#   ./bench/pipeline.sh start     # 挂上（setsid）
#   ./bench/pipeline.sh status
#   ./bench/pipeline.sh stop
#
# 日志: logs/bench/pipeline.log
set -uo pipefail

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
VK="$ROOT/vlmevalkit"
LOGS="$ROOT/logs/bench"
OUT="$ROOT/vlmeval_out"
RES="$ROOT/bench/results"
PIPELOG="$LOGS/pipeline.log"
mkdir -p "$LOGS" "$RES"

BENCHMARKS=(MME MathVista_MINI MathVerse_MINI MMStar MMBench_DEV_EN_V11 MathVision_MINI MMMU_DEV_VAL)
MODELS=(Qwen3.5-4B Qwen3.5-9B)
# 推理端口（每个 benchmark 轮转分配；MathVerse 走 8114）
declare -A PORT4B=( [MME]=8110 [MathVista_MINI]=8110 [MathVerse_MINI]=8112 [MMStar]=8111 \
                    [MMBench_DEV_EN_V11]=8110 [MathVision_MINI]=8111 [MMMU_DEV_VAL]=8112 )
declare -A PORT9B=( [MME]=8113 [MathVista_MINI]=8113 [MathVerse_MINI]=8114 [MMStar]=8114 \
                    [MMBench_DEV_EN_V11]=8116 [MathVision_MINI]=8116 [MMMU_DEV_VAL]=8115 )

export CUDA_HOME="$ROOT/venv/lib/python3.10/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export LMUData="$ROOT/lmudata" QWEN35_PROJ="$ROOT" PYTHONUNBUFFERED=1 ROOT
PY="$ROOT/venv/bin/python"

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$PIPELOG"; }
judge_key(){ grep '^PARATERA_API_KEY=' "$VK/.env" | cut -d= -f2; }

# MathVerse 由看门人按 index 1000 叫停，没有正常结束的 xlsx，
# 不参与「自动补跑」（会和正在跑的作业打架）；判分前用 finalize_mathverse.py 转换。
SKIP_RETRY=("MathVerse_MINI")
is_skip_retry(){ local s; for s in "${SKIP_RETRY[@]}"; do [ "$1" = "$s" ] && return 0; done; return 1; }

# 某 (model, bench) 的失败条数；无预测文件返回 -1
fails(){
    "$PY" - "$1" "$2" "$OUT" <<'PYEOF' 2>/dev/null
import sys, glob, os, pandas as pd
mdl, b, out = sys.argv[1], sys.argv[2], sys.argv[3]
# 既要看扁平的软链路径（judge/finalize 写这里），也要看时间戳子目录
fs=[f for f in [f'{out}/{mdl}/{mdl}_{b}.xlsx'] + glob.glob(f'{out}/{mdl}/*/{mdl}_{b}.xlsx')
    if os.path.exists(f) and 'auxmatch' not in f]
if not fs: print(-1); raise SystemExit
f=max(fs, key=os.path.getmtime)
try:
    d=pd.read_excel(f)
    print(int((d['prediction'].astype(str)=='Failed to obtain answer via API.').sum()))
except Exception: print(-1)
PYEOF
}

infer_running(){ pgrep -f "vlmevalkit/run.py .*--mode infer" >/dev/null 2>&1; }
# 该 (模型, benchmark) 是否已有作业在跑 —— 避免重复下发同一个
job_running_for(){ pgrep -f "vlmevalkit/run.py --model $1 .*--data $2 " >/dev/null 2>&1; }

launch_infer(){  # model port bench logname [reuse]
    local mdl="$1" port="$2" b="$3" logn="$4" reuse="${5:---reuse}"
    setsid nohup "$PY" "$VK/run.py" --model "$mdl" --model-class GPT4V \
        --base-url "http://localhost:$port/v1" --key EMPTY --data "$b" \
        --work-dir "$OUT" --mode infer --api-nproc 28 --retry 3 --timeout 1800 \
        --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 20 $reuse \
        > "$LOGS/$logn.log" 2>&1 < /dev/null &
    log "  下发 $mdl / $b -> 端口 $port  ($logn)"
}

do_judge(){
    local KEY JARGS
    KEY="$(judge_key)"
    JARGS='{"reasoning_effort":"none"}'
    log "===== 判分前：把 MathVerse checkpoint 转成预测 xlsx ====="
    ROOT="$ROOT" "$PY" "$ROOT/bench/finalize_mathverse.py" 2>&1 | sed 's/^/    /' | tee -a "$PIPELOG"
    log "===== 开始判分 ====="
    for mdl in "${MODELS[@]}"; do
        for b in "${BENCHMARKS[@]}"; do
            [ "$(fails "$mdl" "$b")" = "-1" ] && { log "  跳过 $mdl/$b（无预测文件）"; continue; }
            log "  判分 $mdl / $b"
            ( cd "$VK" && "$PY" run.py --model "$mdl" --model-class GPT4V \
                --base-url http://localhost:8110/v1 --key EMPTY \
                --data "$b" --work-dir "$OUT" --mode eval --reuse \
                --judge DeepSeek-V4.1-Flash \
                --judge-base-url https://llmapi.paratera.com/v1 \
                --judge-key "$KEY" --judge-args "$JARGS" \
              ) > "$LOGS/judge_${mdl}_${b}.log" 2>&1
            log "    完成 $mdl/$b (exit=$?)"
        done
    done
    "$PY" "$ROOT/bench/summarize.py" > "$RES/summary.md" 2>"$LOGS/summarize.err" \
        && log "✓ 汇总已生成 $RES/summary.md" || log "✗ 汇总失败，见 $LOGS/summarize.err"
}

main_loop(){
    log "===== 流水线启动 ====="
    local round=0
    while true; do
        round=$((round+1))
        # --- 1) 若有失败但无推理在跑 -> 补跑 ---
        pending=0
        for mdl in "${MODELS[@]}"; do
            for b in "${BENCHMARKS[@]}"; do
                # 跳过列表里的（MathVerse）永不参与补跑，也**不能计入 pending**，
                # 否则 pending==0 永远不成立，流水线永远不会进入判分。
                is_skip_retry "$b" && continue
                f=$(fails "$mdl" "$b"); f=${f:-0}
                [ "$f" = "-1" ] && continue
                [ "$f" -gt 0 ] && pending=$((pending+f))
            done
        done
        log "第 $round 轮：待修复失败条目 $pending 条，推理进程 $(pgrep -cf 'vlmevalkit/run.py .*--mode infer')"

        # 逐个 (模型, benchmark) 判断：有失败 且 该组没有作业在跑 -> 立刻补跑。
        # 不能等「全部推理停下」才补 —— 那样只要有一组长作业在跑，
        # 其余组的失败就永远排不上队（曾导致 4B 的 MathVista/MMMU 一直没补）。
        launched=0
        for mdl in "${MODELS[@]}"; do
            for b in "${BENCHMARKS[@]}"; do
                is_skip_retry "$b" && continue
                f=$(fails "$mdl" "$b"); f=${f:-0}
                [ "$f" -gt 0 ] || continue
                job_running_for "$mdl" "$b" && continue
                if [ "$mdl" = "Qwen3.5-4B" ]; then port="${PORT4B[$b]}"; else port="${PORT9B[$b]}"; fi
                launch_infer "$mdl" "$port" "$b" "retry_${mdl}_${b}" --reuse
                launched=$((launched+1)); sleep 15
            done
        done
        if [ "$launched" -gt 0 ]; then
            log "本轮补跑了下发 $launched 组"
            sleep 300; continue
        fi

        # --- 2) 全部干净 -> 判分 ---
        if [ "$pending" -eq 0 ] && ! infer_running; then
            log "所有 benchmark 无失败且无推理在跑 -> 判分"
            do_judge
            log "===== 流水线完成 ====="
            return 0
        fi
        sleep 300
    done
}

case "${1:-status}" in
  start)
    setsid nohup "$ROOT/bench/pipeline.sh" __run >/dev/null 2>&1 </dev/null &
    sleep 5
    p=$(pgrep -f "bench/pipeline.sh __run" | head -1)
    echo "$p" > "$LOGS/pipeline.pid"
    echo "流水线 PID=$p (PPID=$(ps -o ppid= -p $p 2>/dev/null | tr -d ' '))"
    echo "日志: $PIPELOG"
    ;;
  __run) main_loop ;;
  status)
    p=$(cat "$LOGS/pipeline.pid" 2>/dev/null)
    [ -n "${p:-}" ] && kill -0 "$p" 2>/dev/null && echo "流水线运行中 (PID $p)" || echo "流水线未运行"
    echo "--- 最近日志 ---"; tail -12 "$PIPELOG" 2>/dev/null | sed 's/^/  /'
    ;;
  once) do_judge ;;
  stop)
    p=$(cat "$LOGS/pipeline.pid" 2>/dev/null); [ -n "${p:-}" ] && kill "$p" 2>/dev/null
    echo "已停";;
  *) echo "用法: $0 {start|status|once|stop}"; exit 1;;
esac
