#!/usr/bin/env bash
# 评测全量运行的监控守护。
#
#   1. 服务端死了 -> 重启
#   2. 作业死了但未完成 -> 重启（--reuse 断点续跑）
#   3. 作业 N 分钟无进展 -> 判卡死并重启
#   4. 服务端出现新 500 / EngineDead -> 告警
#   5. 全部 14 个作业完成 -> 打印汇总后退出
#
# 日志: logs/bench/monitor.log
set -uo pipefail

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
LOGS="$ROOT/logs/bench"
PIDDIR="$LOGS/pids"
MONLOG="$LOGS/monitor.log"
INTERVAL=180; STALL_MIN=45; MAX_RESTART=12

BENCHMARKS=(MME MathVista_MINI MathVerse_MINI MMStar MMBench_DEV_EN_V11 MathVision_MINI MMMU_DEV_VAL)

# 故意不监控的组合：MathVerse 已被 stop_mathverse_at_1k.sh 按 index 1000 截止，
# 若监控把它当成「进程死了但没完成」重启，就会越过 1000 条继续跑，白费算力。
declare -A SKIP_JOBS=( ["4B/MathVerse_MINI"]=1 )
declare -A SRV_PORT=( [4B0]=8110 [4B1]=8111 [4B2]=8112 [9B0]=8113 [9B1]=8114 [9B2]=8115 [9B3]=8116 )
declare -A SRV_GPU=(  [4B0]=0 [4B1]=1 [4B2]=2 [9B0]=3 [9B1]=4 [9B2]=5 [9B3]=6 )
declare -A SRV_MDL=(  [4B0]=Qwen3.5-4B [4B1]=Qwen3.5-4B [4B2]=Qwen3.5-4B \
                      [9B0]=Qwen3.5-9B [9B1]=Qwen3.5-9B [9B2]=Qwen3.5-9B [9B3]=Qwen3.5-9B )

export CUDA_HOME="$ROOT/venv/lib/python3.10/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export LMUData="$ROOT/lmudata"
export QWEN35_PROJ="$ROOT"
export ROOT
PY="$ROOT/venv/bin/python"

log(){ echo "[$(date '+%F %T')] [$1] $2" | tee -a "$MONLOG"; }

alive_port(){ curl -sS -m 3 "http://localhost:$1/v1/models" >/dev/null 2>&1; }
completed(){ grep -q "Run Summary" "$1" 2>/dev/null; }

# 已完成条数 —— 从 checkpoint/xlsx 读，**单调递增**。
# 不能用日志里的 tqdm 计数器：作业重启后它会归零，导致监控误判为
# 「一直没进展」而反复重启（MME/MathVision 曾因此陷入重启死循环）。
prog_count(){
    "$PY" - "$1" "$2" <<'PYEOF' 2>/dev/null
import sys, glob, pickle, os
mdl, bench = sys.argv[1], sys.argv[2]
root = os.environ.get("ROOT", ".")
xls = glob.glob(f"{root}/vlmeval_out/{mdl}/*/{mdl}_{bench}.xlsx")
if xls:
    try:
        import pandas as pd
        print(len(pd.read_excel(xls[0])))
    except Exception:
        print(-1)   # 完成但读不出行数，用 -1 表示「已完成」
    raise SystemExit
n = 0
for p in glob.glob(f"{root}/vlmeval_out/{mdl}/*/{mdl}_{bench}_checkpoint.pkl"):
    try:
        with open(p, "rb") as f:
            n = max(n, len(pickle.load(f)))
    except Exception:
        pass
print(n)
PYEOF
}

job_port(){  # tag(4B|9B) benchmark_index -> port
    local tag="$1" i="$2" nrep
    [ "$tag" = "4B" ] && nrep=3 || nrep=4
    echo "${SRV_PORT[${tag}$(( i % nrep ))]}"
}

start_job(){
    local tag="$1" bench="$2" mdl port
    [ "$tag" = "4B" ] && mdl=Qwen3.5-4B || mdl=Qwen3.5-9B
    local i=0; for b in "${BENCHMARKS[@]}"; do [ "$b" = "$bench" ] && break; i=$((i+1)); done
    port=$(job_port "$tag" "$i")
    setsid nohup "$PY" "$ROOT/vlmevalkit/run.py" --model "$mdl" --model-class GPT4V \
        --base-url "http://localhost:$port/v1" --key EMPTY --data "$bench" \
        --work-dir "$ROOT/vlmeval_out" --mode infer --api-nproc 28 --retry 3 --timeout 1800 \
        --max-tokens 32768 --temperature 1.0 --top-p 0.95 --top-k 20 --reuse \
        > "$LOGS/${tag}_${bench}.log" 2>&1 < /dev/null &
    echo $! > "$PIDDIR/infer_${tag}_${bench}.pid"
}

log INFO "监控启动 (间隔 ${INTERVAL}s, 卡死阈值 ${STALL_MIN}min)"
declare -A last_prog last_change restarts
declare -A LASTERR     # 必须声明为关联数组，否则 LASTERR[err_4B0] 会被当成算术下标而在 set -u 下报 unbound
for tag in 4B 9B; do for b in "${BENCHMARKS[@]}"; do
    last_prog[$tag/$b]=-1; last_change[$tag/$b]=$(date +%s); restarts[$tag/$b]=0
done; done
for k in "${!SRV_PORT[@]}"; do LASTERR[err_$k]=0; done

while true; do
    sleep $INTERVAL
    now=$(date +%s); all_done=1

    # --- 服务端健康 ---
    for k in "${!SRV_PORT[@]}"; do
        if ! alive_port "${SRV_PORT[$k]}"; then
            log WARN "服务端 $k (端口 ${SRV_PORT[$k]}) 已死，重启"
            # 必须用 venv 里 vllm 的绝对路径：监控脚本没有 source venv，
            # PATH 里找不到 vllm，否则会一直 "failed to run command 'vllm'"
            CUDA_VISIBLE_DEVICES="${SRV_GPU[$k]}" setsid nohup "$ROOT/venv/bin/vllm" serve "$ROOT/models/${SRV_MDL[$k]}" \
                --served-model-name "${SRV_MDL[$k]}" --port "${SRV_PORT[$k]}" \
                --max-model-len 40960 --gpu-memory-utilization 0.90 --max-num-seqs 28 \
                --limit-mm-per-prompt '{"image": 8}' --trust-remote-code \
                >> "$LOGS/vllm_$k.log" 2>&1 < /dev/null &
            echo $! > "$PIDDIR/vllm_$k.pid"
        fi
    done

    # --- 作业健康 ---
    for tag in 4B 9B; do
        for b in "${BENCHMARKS[@]}"; do
            f="$LOGS/${tag}_${b}.log"
            key="$tag/$b"
            if completed "$f"; then last_prog[$key]=999999; continue; fi
            if [ -n "${SKIP_JOBS[$key]:-}" ]; then continue; fi   # 故意不监控
            all_done=0
            mdl_j=Qwen3.5-4B; [ "$tag" = "9B" ] && mdl_j=Qwen3.5-9B

            # 进程已死且未完成 -> 立即重启，不必等 STALL_MIN 分钟
            if ! pgrep -f "run.py --model $mdl_j .*--data $b " >/dev/null 2>&1; then
                local_i=0; for bb in "${BENCHMARKS[@]}"; do [ "$bb" = "$b" ] && break; local_i=$((local_i+1)); done
                jp=$(job_port "$tag" "$local_i")
                if alive_port "$jp"; then
                    log WARN "$key 进程已死且未完成，立即重启"
                    start_job "$tag" "$b"
                    last_change[$key]=$now
                    last_prog[$key]=-1
                else
                    log WARN "$key 进程已死但服务端 $jp 未就绪，等服务器恢复"
                    last_change[$key]=$now
                fi
                continue
            fi

            cur=$(prog_count "$mdl_j" "$b"); cur=${cur:-0}
            if [ "$cur" -gt "${last_prog[$key]}" ]; then
                last_prog[$key]=$cur; last_change[$key]=$now; continue
            fi
            stalled=$(( (now - ${last_change[$key]}) / 60 ))
            if [ "$stalled" -ge "$STALL_MIN" ]; then
                # 若该作业对应的服务端还没起来，重启作业毫无意义（只会空转）。
                # 先把时间戳推后，等服务端恢复后自然会重新推进。
                local_i=0; for bb in "${BENCHMARKS[@]}"; do [ "$bb" = "$b" ] && break; local_i=$((local_i+1)); done
                jp=$(job_port "$tag" "$local_i")
                if ! alive_port "$jp"; then
                    log WARN "$key 无进展但服务端 $jp 未就绪，等待服务端恢复，暂不重启作业"
                    last_change[$key]=$now
                    continue
                fi
                if [ "${restarts[$key]}" -lt "$MAX_RESTART" ]; then
                    log WARN "$key 已 ${stalled} 分钟无进展 (${cur})，重启"
                    pkill -f "run.py --model .* --data $b" 2>/dev/null
                    sleep 3
                    restarts[$key]=$(( ${restarts[$key]} + 1 ))
                    last_change[$key]=$now
                    start_job "$tag" "$b"
                else
                    log ERROR "$key 卡死且达重启上限，需人工介入"
                fi
            fi
        done
    done

    # --- 服务端错误 ---
    for k in "${!SRV_PORT[@]}"; do
        # grep -c 在计数为 0 时退出码是 1，不能用 `|| echo 0`（会多输出一个 0）
        n=$(grep -c "500 Internal Server Error" "$LOGS/vllm_$k.log" 2>/dev/null); n=${n:-0}
        key="err_$k"
        prev=${LASTERR[$key]:-0}
        [ "$n" -gt "$prev" ] && log WARN "服务端 $k 新增 $((n-prev)) 条 500 错误 (累计 $n)"
        LASTERR[$key]=$n
    done

    # --- 汇总 ---
    tot=0; for tag in 4B 9B; do for b in "${BENCHMARKS[@]}"; do
        mj=Qwen3.5-4B; [ "$tag" = "9B" ] && mj=Qwen3.5-9B
        c=$(prog_count "$mj" "$b"); c=${c:-0}; [ "$c" -lt 0 ] && c=0
        tot=$(( tot + c ))
    done; done
    log INFO "心跳: 已完成 $tot/30088 ($(awk "BEGIN{printf \"%.1f\", $tot/30088*100}")%)"

    if [ "$all_done" -eq 1 ]; then
        log INFO "全部 14 个作业完成，退出监控"
        exit 0
    fi
done
