#!/usr/bin/env bash
# 看门人：MathVerse 作业跑到 index 1000 就停，不在后 2940 条上浪费算力。
#
# 背景：用户把 MathVerse 统一为「测量顺序的前 1000 条」（= TSV 前 1000 行 = index 1–1000），
# 命名为 mathversemini，单独切出以便下次复用（bench/datasets/MathVerse_MINI_1K.tsv）。
# 正在跑的作业是按完整 3940 条下发的，这里让它们做完 1000 就收工，不重跑。
#
# VLMEvalKit 按 TSV 行序处理，所以「已完成条数 >= 1000」等价于「前 1000 条已测完」。
set -uo pipefail

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
LOGS="$ROOT/logs/bench"
LOG="$LOGS/stop_mathverse.log"
PY="$ROOT/venv/bin/python"
LIMIT=1000

export ROOT
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

count_done(){  # $1 = Qwen3.5-4B | Qwen3.5-9B
    "$PY" - "$1" <<'PYEOF' 2>/dev/null
import sys, glob, pickle, os
mdl = sys.argv[1]; root = os.environ.get("ROOT", ".")
n = 0
for p in glob.glob(f"{root}/vlmeval_out/{mdl}/*/{mdl}_MathVerse_MINI_checkpoint.pkl"):
    try:
        with open(p, "rb") as f:
            d = pickle.load(f)
        ks = sorted(int(k) for k in d.keys() if str(k).isdigit())
        if ks:
            n = max(n, sum(1 for k in ks if k <= 1000))
    except Exception:
        pass
print(n)
PYEOF
}

log "看门人启动：MathVerse 到 $LIMIT 条即停"
while true; do
    done4=$(count_done Qwen3.5-4B); done4=${done4:-0}
    done9=$(count_done Qwen3.5-9B); done9=${done9:-0}

    for pair in "4B:Qwen3.5-4B:$done4" "9B:Qwen3.5-9B:$done9"; do
        tag="${pair%%:*}"; rest="${pair#*:}"; mdl="${rest%%:*}"; n="${rest##*:}"
        if [ "$n" -ge "$LIMIT" ]; then
            if pgrep -f "run.py --model $mdl .*--data MathVerse_MINI " >/dev/null 2>&1; then
                log "$mdl 已达 $n 条（>= $LIMIT），停止该作业"
                # 杀之前先备份 checkpoint —— 上次直接 pkill 恰好撞上 VLMEvalKit
                # 正在写 checkpoint，把文件截断成 0 字节，1000 条结果全丢了。
                for ck in $ROOT/vlmeval_out/$mdl/*/${mdl}_MathVerse_MINI_checkpoint.pkl; do
                    [ -s "$ck" ] && cp -f "$ck" "$ROOT/bench/backup/${mdl}_MathVerse_MINI_$(date +%s).pkl" \
                        && log "  已备份 $ck"
                done
                sleep 5   # 让可能的写入落盘
                pkill -f "run.py --model $mdl .*--data MathVerse_MINI " 2>/dev/null
            fi
        fi
    done

    if [ "$done4" -ge "$LIMIT" ] && [ "$done9" -ge "$LIMIT" ]; then
        log "两个模型的 MathVerse 前 $LIMIT 条均已测完，看门人退出"
        exit 0
    fi
    log "进度: 4B=$done4/$LIMIT  9B=$done9/$LIMIT"
    sleep 300
done
