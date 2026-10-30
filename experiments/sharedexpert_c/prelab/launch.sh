#!/usr/bin/env bash
# 在**独立会话**中启动全量推理，脱离终端 / SSH / Claude 会话。
# 会话结束、终端关闭、SSH 断开都不会影响它。
#
#   ./launch.sh              # 启动全量（1000 条 × 6 组）
#   ./launch.sh --smoke 20   # 冒烟
#   ./launch.sh --status     # 看进度
#   ./launch.sh --stop       # 停止
#
# 日志: probe/logs/_full_run.log   各组合: probe/logs/{model}__{cond}.log
set -uo pipefail

ROOT=/home/ml-user/workdir/CVPR2027/pre-lab
cd "$ROOT"
PIDFILE="$ROOT/probe/logs/full_run.pid"
LOGFILE="$ROOT/probe/logs/_full_run.log"
mkdir -p probe/logs

case "${1:-}" in
  --status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
        echo "运行中 (PID $(cat $PIDFILE), 会话 $(ps -o sid= -p $(cat $PIDFILE) | tr -d ' '))"
    else
        echo "未在运行"
    fi
    echo
    for f in probe/logs/4B__*.log probe/logs/9B__*.log; do
        [ -e "$f" ] || continue
        n=$(basename "$f" .log)
        l=$(grep -oE "[0-9]+/[0-9]+  " "$f" 2>/dev/null | tail -1)
        printf "  %-16s %s\n" "$n" "${l:-初始化中}"
    done
    echo
    echo "已产出结果 (/1000):"
    for f in probe/out/*.jsonl; do
        [ -e "$f" ] || continue
        printf "  %-24s %s 条\n" "$(basename $f)" "$(wc -l < $f)"
    done
    echo
    if [ -f "$ROOT/probe/logs/monitor.pid" ] && kill -0 "$(cat $ROOT/probe/logs/monitor.pid)" 2>/dev/null; then
        echo "监控守护: 运行中 (PID $(cat $ROOT/probe/logs/monitor.pid))"
    else
        echo "监控守护: 未运行"
    fi
    echo "--- monitor.log 最近 8 条 ---"
    tail -8 "$ROOT/probe/logs/monitor.log" 2>/dev/null | sed 's/^/  /' || echo "  (暂无)"
    echo
    echo "--- 告警/WARN/ERROR 统计 ---"
    # grep -c 在计数为 0 时退出码是 1，所以不能用 `|| echo 0`（会多打一个 0）
    NW=$(grep -c '\[WARN\]'  "$ROOT/probe/logs/monitor.log" 2>/dev/null); NW=${NW:-0}
    NE=$(grep -c '\[ERROR\]' "$ROOT/probe/logs/monitor.log" 2>/dev/null); NE=${NE:-0}
    [ -f "$ROOT/probe/logs/monitor.log" ] || { NW=0; NE=0; }
    printf "  WARN: %s   ERROR: %s\n" "$NW" "$NE"
    if [ "${NE:-0}" -gt 0 ]; then
        echo "  --- 最近的 ERROR ---"
        grep '\[ERROR\]' "$ROOT/probe/logs/monitor.log" | tail -5 | sed 's/^/  /'
    fi
    exit 0
    ;;
  --stop)
    # 先停监控，否则它会把我们刚杀掉的进程当异常重启
    if [ -f "$ROOT/probe/logs/monitor.pid" ]; then
        MP=$(cat "$ROOT/probe/logs/monitor.pid")
        MS=$(ps -o sid= -p "$MP" 2>/dev/null | tr -d ' ')
        [ -n "$MS" ] && kill -TERM -"$MS" 2>/dev/null
        kill "$MP" 2>/dev/null
        rm -f "$ROOT/probe/logs/monitor.pid"
        echo "监控已停"
    fi
    if [ -f "$PIDFILE" ]; then
        PID=$(cat "$PIDFILE")
        SID=$(ps -o sid= -p "$PID" 2>/dev/null | tr -d ' ')
        # 杀掉整个会话，连同 vLLM 的 EngineCore 子进程
        [ -n "$SID" ] && kill -TERM -"$SID" 2>/dev/null
        sleep 5
        for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null; done
        rm -f "$PIDFILE"
        echo "推理已停"
    else
        echo "没有 PID 文件"
    fi
    exit 0
    ;;
esac

if [ -f "$PIDFILE" ] && kill -0 "$(cat $PIDFILE)" 2>/dev/null; then
    echo "已经在跑 (PID $(cat $PIDFILE))。先 ./launch.sh --stop，或用 ./launch.sh --status 看进度"
    exit 1
fi

# setsid: 新建会话，完全脱离当前终端/会话
setsid nohup ./probe/run_all.sh "$@" > "$LOGFILE" 2>&1 < /dev/null &
sleep 3

# setsid 可能 fork，$! 未必是真正的 run_all.sh，按命令行精确定位
# 真实命令行形如 "bash ./probe/run_all.sh"，且 PPID=1（已脱离）
PID=$(pgrep -f "^bash \./probe/run_all\.sh$" | head -1)
if [ -z "$PID" ]; then
    echo "✗ 启动失败，见 $LOGFILE"
    tail -20 "$LOGFILE"
    exit 1
fi
echo "$PID" > "$PIDFILE"

# 监控守护进程，同样放进独立会话
MONPIDFILE="$ROOT/probe/logs/monitor.pid"
setsid nohup "$ROOT/venv/bin/python" "$ROOT/probe/monitor.py" \
    > /dev/null 2>&1 < /dev/null &
sleep 2
MONPID=$(pgrep -f "^\./venv/bin/python probe/monitor\.py$" | head -1)
[ -n "$MONPID" ] && echo "$MONPID" > "$MONPIDFILE"

SID=$(ps -o sid= -p "$PID" 2>/dev/null | tr -d ' ')
echo "已启动"
echo "  推理主进程 : PID $PID   会话 $SID   (独立会话 — 关终端/断 SSH 都不受影响)"
[ -n "$MONPID" ] && echo "  监控守护   : PID $MONPID  会话 $(ps -o sid= -p $MONPID 2>/dev/null | tr -d ' ')"
echo "  日志       : $LOGFILE"
echo
echo "查看进度:  ./launch.sh --status"
echo "监控日志:  tail -f probe/logs/monitor.log"
echo "实时日志:  tail -f $LOGFILE"
echo "停止:      ./launch.sh --stop"
echo "浏览例子:  python3 probe/view.py --n 3"
