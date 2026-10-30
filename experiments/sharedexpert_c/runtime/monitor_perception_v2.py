#!/usr/bin/env python3
"""Read-only monitoring for the separately queued perception-v2 teacher."""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
from zoneinfo import ZoneInfo

RUNTIME = Path(__file__).resolve().parent
EXPERIMENT = RUNTIME.parent
sys.path.insert(0, str(EXPERIMENT))
import monitor_training as shared

QUEUE = RUNTIME / "perception_v2_queue.json"
OUTPUT = EXPERIMENT / "monitoring_perception_v2"
RUN = EXPERIMENT / "runs/perception_v2"


def queue_health(state):
    """Do not report training alive based only on a stale queue JSON file."""
    stage = state.get("stage", "unknown")
    error = state.get("error")
    if stage not in {"complete", "failed", "interrupted", "unknown"}:
        pid, ticks = state.get("pid"), state.get("start_ticks")
        try:
            proc = Path("/proc") / str(int(pid))
            fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
            argv = (proc / "cmdline").read_bytes().split(b"\0")
            valid = (ticks is not None and int(fields[19]) == int(ticks)
                     and fields[0] not in {"Z", "X"}
                     and any(Path(a.decode(errors="replace")).name == "queue_perception_v2.py"
                             for a in argv if a))
        except (OSError, ValueError, TypeError, IndexError):
            valid = False
        if not valid:
            stage, error = "failed", "Queue process is absent or its PID/start time changed; inspect the queue log."
    return stage, error


class PerceptionMonitor(shared.Monitor):
    def observe_experiment(self, driver, sample):
        state = shared.read_json(QUEUE, {})
        stage, error = queue_health(state)
        display = ("COMPLETE" if stage == "complete" else "STOPPED" if stage in {"failed", "interrupted"}
                   else "WAITING" if stage == "waiting" else "RUNNING" if stage in {"preflight", "smoke", "training"}
                   else "UNKNOWN")
        signature = [stage, error]
        sample.update(experiment_stage=stage, experiment_teacher="perception_v2", experiment_error=error,
                      display_status=display, stage_changed=signature != self.state.get("experiment_signature"),
                      queue_pid=state.get("pid"), status_reason=state.get("status_reason"))
        for progress in sample.get("progress", {}).values():
            progress["exported"] = (RUN / "hf/global_step_100/sharedexpert_export_complete.json").is_file()
        self.state.update(experiment_stage=stage, experiment_teacher="perception_v2", experiment_error=error,
                          display_status=display, experiment_signature=signature,
                          status_reason=state.get("status_reason"), queue_pid=state.get("pid"))
        if stage in {"failed", "interrupted"}:
            self.alert("perception_v2_queue_stopped", json.dumps(signature), source=str(QUEUE), reason=error)
            self.stop_requested = True

    def write_status(self):
        state = self.state
        progress = state.get("progress", {}).get("perception_v2", {})
        stamp = dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        lines = ["# 第二版感知教师训练状态", "", f"更新时间（北京时间）：{stamp}", "",
                 f"状态：**{state.get('display_status', 'UNKNOWN')}**；阶段：`{state.get('experiment_stage', 'unknown')}`。", "",
                 f"说明：{state.get('experiment_error') or state.get('status_reason') or '等待队列状态。'}", "",
                 f"最新训练指标步数：{state.get('latest_metric_steps', {}).get('perception_v2', '尚无指标')} / 100。", "",
                 f"已保存 checkpoint：{progress.get('checkpoint_step', 0)}；最终导出完成：{progress.get('exported', False)}。", "",
                 "训练配置：100 steps × batch 30 × group 8，thinking 开启，GRPO，LR 1e-6。", "",
                 "六类比例：20:20:20:15:15:10；正式训练 3,000 条问题、24,000 次回答采样。", "",
                 "[训练曲线](training_metrics.png) · [指标 CSV](metrics.csv) · [队列状态](../runtime/perception_v2_queue.json)", "",
                 "这里只绘制 perception_v2 正式训练；不混入 smoke 或其他教师的指标。", ""]
        shared.atomic_text(self.output / "status.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run",), default="run")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(OUTPUT / ".matplotlib")
    with shared.lock_path(OUTPUT).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"already_running": True, "output": str(OUTPUT)}))
            return
        monitor = PerceptionMonitor(OUTPUT, {"perception_v2": RUN}, include_driver=False, test_mode=False)
        def stop(signum, frame):
            monitor.stop_requested = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        monitor.run(interval=5.0, render_interval=60.0, once=args.once)


if __name__ == "__main__":
    main()
