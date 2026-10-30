#!/usr/bin/env python3
"""监控全量推理任务：发现异常就写告警并尝试自动恢复；全部跑完自动出报告。

异常判定：
  1. 某个 run_infer 进程死了但该组结果没跑完        -> 告警 + 自动重启（断点续跑）
  2. 某组进度 STALL_MIN 分钟没增长（卡死）          -> 告警 + 自动重启
  3. run_all.sh 主进程死了但结果不完整              -> 告警
  4. 日志里出现新的致命错误（OOM / CUDA error）     -> 告警
  5. 孤儿 VLLM::EngineCore 占着显存没人管           -> 告警（自动清理）
  6. 磁盘剩余 < 100GB                              -> 告警

日志: probe/logs/monitor.log
启动: setsid nohup python3 probe/monitor.py > /dev/null 2>&1 &
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

ROOT = common.ROOT
LOG_DIR = os.path.join(common.PROBE_DIR, "logs")
MON_LOG = os.path.join(LOG_DIR, "monitor.log")
PIDFILE = os.path.join(LOG_DIR, "full_run.pid")
TOTAL = 1000

INTERVAL = 60          # 巡检间隔（秒）
STALL_MIN = 15         # 多久没进展算卡死（分钟）
MAX_RESTART = 3        # 每组最多自动重启次数

FATAL_PAT = re.compile(
    r"CUDA out of memory|torch\.OutOfMemoryError|RuntimeError: CUDA error|"
    r"Engine core initialization failed|EngineCore failed to start|"
    r"illegal memory access|device-side assert",
    re.I,
)


def log(msg, level="INFO"):
    line = f"[{datetime.now():%F %T}] [{level}] {msg}"
    print(line, flush=True)
    with open(MON_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return f""


def progress(mk, cond):
    """该组已完成的样本数。"""
    p = common.out_path(mk, cond)
    if not os.path.exists(p):
        return 0
    with open(p, "rb") as f:
        return sum(1 for _ in f)


def procs_alive(pattern):
    out = sh(f"ps -eo pid,args | grep '[{pattern[0]}]{pattern[1:]}'")
    return [l.split(None, 1)[0] for l in out.strip().splitlines() if l.strip()]


def run_infer_pid(mk, cond):
    out = sh(f"ps -eo pid,args | grep '[r]un_infer.py --model-key {mk} --condition {cond}'")
    for l in out.strip().splitlines():
        if l.strip():
            return l.split(None, 1)[0]
    return None


def start_one(mk, cond):
    env = os.environ.copy()
    env["CUDA_HOME"] = f"{ROOT}/venv/lib/python3.10/site-packages/nvidia/cu13"
    env["PATH"] = env["CUDA_HOME"] + "/bin:" + env.get("PATH", "")
    env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    env["VLLM_CACHE_ROOT"] = f"{ROOT}/probe/.vllm_cache/{mk}__{cond}"
    env["OMP_NUM_THREADS"] = "8"
    gpu = list(common.MODELS).index(mk) * 3 + common.CONDITIONS.index(cond)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(os.path.join(LOG_DIR, f"{mk}__{cond}.restart.log"), "a")
    subprocess.Popen(
        [f"{ROOT}/venv/bin/python", os.path.join(common.PROBE_DIR, "run_infer.py"),
         "--model-key", mk, "--condition", cond],
        cwd=ROOT, env=env, stdout=logf, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    log(f"已重启 {mk}/{cond} (GPU {gpu})", "WARN")


def check_gpu_orphans():
    out = sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader")
    orphans = []
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        pid = line.split(",")[0].strip()
        args = sh(f"ps -p {pid} -o args= 2>/dev/null").strip()
        ppid = sh(f"ps -p {pid} -o ppid= 2>/dev/null").strip()
        if "EngineCore" in args and ppid == "1":
            orphans.append((pid, line.split(",")[1].strip()))
    return orphans


def check_disk():
    """返回挂载点剩余 GB。df -BG 的字段: 文件系统 容量 已用 可用 使用% 挂载点"""
    parts = sh(f"df -BG {ROOT} | tail -1").split()
    if len(parts) < 4:
        return 9999
    try:
        return int(parts[3].rstrip("G"))
    except ValueError:
        return 9999


def scan_new_errors():
    """扫描各组日志的致命错误（只报没报过的）。"""
    seen_path = os.path.join(LOG_DIR, ".seen_errors")
    seen = set(open(seen_path).read().split("\n")) if os.path.exists(seen_path) else set()
    hits = []
    for mk in common.MODELS:
        for cond in common.CONDITIONS:
            p = os.path.join(LOG_DIR, f"{mk}__{cond}.log")
            if not os.path.exists(p):
                continue
            try:
                with open(p, errors="ignore") as f:
                    for i, line in enumerate(f):
                        if FATAL_PAT.search(line):
                            key = f"{mk}__{cond}:{i}:{line[:80]}"
                            if key not in seen:
                                seen.add(key)
                                hits.append(f"{mk}/{cond} -> {line.strip()[:160]}")
            except Exception:
                pass
    with open(seen_path, "w") as f:
        f.write("\n".join(seen))
    return hits


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    log("=" * 70)
    log(f"监控启动  (间隔 {INTERVAL}s, 卡死阈值 {STALL_MIN}min, 每组最多重启 {MAX_RESTART} 次)")

    last_prog = {(mk, c): -1 for mk in common.MODELS for c in common.CONDITIONS}
    last_change = {(mk, c): time.time() for mk in common.MODELS for c in common.CONDITIONS}
    restarts = {(mk, c): 0 for mk in common.MODELS for c in common.CONDITIONS}
    reported_done = False

    while True:
        time.sleep(INTERVAL)
        now = time.time()

        # ---- 完成判定 ----
        done = {(mk, c): progress(mk, c) for mk in common.MODELS for c in common.CONDITIONS}
        if all(v >= TOTAL for v in done.values()):
            if not reported_done:
                log("全部 6 组完成，生成汇总与报告 ...")
                for script in ("score.py", "report.py"):
                    r = subprocess.run([sys.executable, os.path.join(common.PROBE_DIR, script)],
                                       cwd=ROOT, capture_output=True, text=True)
                    log(f"{script} exit={r.returncode}")
                    if r.returncode != 0:
                        log(f"{script} stderr: {r.stderr[-500:]}", "ERROR")
                reported_done = True
                log("报告已生成: probe/results/report.md — 监控退出")
            return

        # ---- 逐组检查 ----
        for mk in common.MODELS:
            for cond in common.CONDITIONS:
                key = (mk, cond)
                cur = done[key]
                if cur > last_prog[key]:
                    last_prog[key], last_change[key] = cur, now
                    continue
                if cur >= TOTAL:
                    continue

                pid = run_infer_pid(mk, cond)
                stalled = (now - last_change[key]) / 60

                if not pid:
                    if restarts[key] < MAX_RESTART:
                        log(f"{mk}/{cond} 进程不存在且未完成 ({cur}/{TOTAL})，自动重启 "
                            f"(第 {restarts[key]+1}/{MAX_RESTART} 次)", "WARN")
                        restarts[key] += 1
                        last_change[key] = now
                        start_one(mk, cond)
                    else:
                        log(f"{mk}/{cond} 反复失败，已达重启上限 ({cur}/{TOTAL})，需人工介入", "ERROR")
                elif stalled > STALL_MIN:
                    if restarts[key] < MAX_RESTART:
                        log(f"{mk}/{cond} 已 {stalled:.0f} 分钟无进展 ({cur}/{TOTAL})，"
                            f"判定卡死，重启 (第 {restarts[key]+1}/{MAX_RESTART} 次)", "WARN")
                        sh(f"kill {pid}")
                        time.sleep(5)
                        restarts[key] += 1
                        last_change[key] = now
                        start_one(mk, cond)
                    else:
                        log(f"{mk}/{cond} 卡死且达重启上限 ({cur}/{TOTAL})，需人工介入", "ERROR")

        # ---- 致命错误 ----
        for h in scan_new_errors():
            log(f"日志出现致命错误: {h}", "ERROR")

        # ---- 孤儿 EngineCore ----
        for pid, mem in check_gpu_orphans():
            log(f"孤儿 VLLM::EngineCore PID={pid} 占用 {mem}，清理", "WARN")
            sh(f"kill -9 {pid}")

        # ---- 磁盘 ----
        free = check_disk()
        if free < 100:
            log(f"磁盘剩余仅 {free}GB", "ERROR")

        # ---- 心跳 ----
        tot = sum(done.values())
        alive = len(procs_alive("run_infer.py"))
        log(f"心跳: 总进度 {tot}/{TOTAL*6} ({tot/(TOTAL*6)*100:.1f}%) | "
            f"活跃进程 {alive}/6 | 磁盘 {free}GB")


if __name__ == "__main__":
    main()
