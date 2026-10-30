#!/usr/bin/env python3
"""Read-only monitoring for perception-v3 filtering and formal training."""
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
import monitor_training as shared  # noqa: E402

QUEUE = RUNTIME / "perception_v3_pipeline.json"
OUTPUT = EXPERIMENT / "monitoring_perception_v3"
RUN = EXPERIMENT / "runs/perception_v3"
FILTER = EXPERIMENT / "data_pipeline/prepared/perception_v3_filter"
ACTIVE_STAGES = {"prepping", "screening", "expanding", "preflight", "waiting_gpu", "waiting_network", "training"}
WEIGHTS = {"object_attribute": 20, "grounding": 20, "ocr_document": 20,
           "spatial_detail": 15, "existence_confusion": 15, "general_vqa_replay": 10}


def process_identity_matches(identity, script, proc_root, *, require_argv=False):
    """Reject dead processes and recycled PIDs, including unrelated launchers."""
    try:
        proc = proc_root / str(int(identity["pid"]))
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        argv = [arg.decode(errors="replace") for arg in
                (proc / "cmdline").read_bytes().split(b"\0") if arg]
        return (int(fields[19]) == int(identity["start_ticks"])
                and fields[0] not in {"Z", "X"}
                and any(Path(arg).name == script for arg in argv)
                and (not require_argv or argv == identity.get("argv")))
    except (OSError, KeyError, ValueError, TypeError, IndexError):
        return False


def fast_start_complete(run_dir):
    """Require this launch's completion receipt and its actual final HF export."""
    try:
        run_dir = Path(run_dir)
        started = shared.read_json(run_dir / "fast_start.json", {})
        completed = shared.read_json(run_dir / "fast_start_complete.json", {})
        export = run_dir / "hf/global_step_100"
        marker = shared.read_json(export / "sharedexpert_export_complete.json", {})
        if not (completed.get("status") == "complete" and completed.get("step") == 100
                and completed.get("config_sha256")
                and completed["config_sha256"] == started.get("config_sha256")
                and dt.datetime.fromisoformat(completed["finished_at"]) >=
                    dt.datetime.fromisoformat(started["started_at"])
                and Path(completed["hf_model"]).resolve() == export.resolve()
                and marker.get("step") == 100
                and (export / "config.json").is_file()):
            return False
        shards = list(export.glob("*.safetensors"))
        if not shards or not all(path.stat().st_size > 0 for path in shards):
            return False
        index = export / "model.safetensors.index.json"
        return not index.exists() or set(shared.read_json(index, {})["weight_map"].values()) == {
            path.name for path in shards}
    except (OSError, KeyError, ValueError, TypeError):
        return False


def queue_health(state, proc_root=Path("/proc"), run_dir=RUN):
    """A queue JSON alone is insufficient evidence that the pipeline is alive."""
    stage = state.get("stage", "unknown")
    error = state.get("error")
    if stage == "training" and state.get("active_training"):
        if fast_start_complete(run_dir):
            return "complete", None
        if process_identity_matches(state["active_training"], "fast_start_perception_v3.py",
                                    proc_root, require_argv=True):
            return stage, None
    if stage in ACTIVE_STAGES:
        valid = process_identity_matches(state, "run_perception_v3_pipeline.py", proc_root)
        if not valid:
            stage = "failed"
            error = ("Pipeline and training launcher are absent or their process identity changed; "
                     "inspect perception_v3_pipeline.log and runs/perception_v3/training.log.")
    return stage, error


class PerceptionMonitor(shared.Monitor):
    def __init__(self, output, run_dir, queue_path=QUEUE, filter_dir=FILTER):
        self.queue_path = Path(queue_path)
        self.filter_dir = Path(filter_dir)
        self.run_dir = Path(run_dir)
        super().__init__(output, {"perception_v3": self.run_dir}, include_driver=False, test_mode=False)

    def observe_experiment(self, driver, sample):
        state = shared.read_json(self.queue_path, {})
        stage, error = queue_health(state, run_dir=self.run_dir)
        screening = shared.read_json(self.filter_dir / "status.json", {})
        display = ("COMPLETE" if stage == "complete" else "STOPPED" if stage in {"failed", "interrupted"}
                   else "WAITING" if stage in {"waiting_gpu", "waiting_network"} else "RUNNING" if stage in ACTIVE_STAGES
                   else "UNKNOWN")
        signature = [stage, error, state.get("status_reason")]
        sample.update(
            experiment_stage=stage, experiment_teacher="perception_v3", experiment_error=error,
            display_status=display, stage_changed=signature != self.state.get("experiment_signature"),
            queue_pid=state.get("pid"), status_reason=state.get("status_reason"), screening=screening,
        )
        for progress in sample.get("progress", {}).values():
            progress["exported"] = (self.run_dir / "hf/global_step_100/sharedexpert_export_complete.json").is_file()
        self.state.update(
            experiment_stage=stage, experiment_teacher="perception_v3", experiment_error=error,
            display_status=display, experiment_signature=signature,
            status_reason=state.get("status_reason"), queue_pid=state.get("pid"), screening=screening,
        )
        self.scan_log(RUNTIME / "perception_v3_pipeline.log", driver_log=True)
        if stage in {"failed", "interrupted"}:
            self.alert("perception_v3_pipeline_stopped", json.dumps(signature), source=str(self.queue_path), reason=error)
            self.stop_requested = True

    def write_status(self):
        state = self.state
        progress = state.get("progress", {}).get("perception_v3", {})
        screening = state.get("screening", {})
        stamp = dt.datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")
        counts = screening.get("accepted_counts", {})
        quotas = screening.get("quotas", {})
        lines = [
            "# 第三版感知数据筛选与正式训练状态", "",
            f"更新时间（北京时间）：{stamp}", "",
            f"状态：**{state.get('display_status', 'UNKNOWN')}**；阶段：`{state.get('experiment_stage', 'unknown')}`。", "",
            f"说明：{state.get('experiment_error') or state.get('status_reason') or '等待流程状态。'}", "",
            f"筛选阶段：`{screening.get('status', '尚未开始')}`；已检查候选题：{screening.get('screened', 0)}；满足条件：{screening.get('accepted', 0)}。", "",
            "入选条件：使用冻结的训练起点模型，每题采样 8 次，答对次数必须为 2–6 次。", "",
            "| 类别 | 原配比 | 已通过筛选 | 目标数量 |", "|---|---:|---:|---:|",
        ]
        for category, weight in WEIGHTS.items():
            lines.append(f"| {category} | {weight}% | {counts.get(category, 0)} | {quotas.get(category, 3000 * weight // 100)} |")
        lines.extend([
            "", f"最新训练指标步数：{state.get('latest_metric_steps', {}).get('perception_v3', '尚无指标')} / 100。", "",
            f"已保存 checkpoint：{progress.get('checkpoint_step', 0)}；最终导出完成：{progress.get('exported', False)}。", "",
            "正式训练配置：100 steps × batch 30 × group 8，thinking 开启，GRPO，LR 1e-6。", "",
            "训练输入目标：3,000 条已通过筛选的问题；六类比例 20:20:20:15:15:10。", "",
            "[训练曲线](training_metrics.png) · [指标 CSV](metrics.csv) · [流程状态](../runtime/perception_v3_pipeline.json)", "",
            "筛选结果来自实际推理记录；训练曲线只读取 perception_v3 正式训练指标。", "",
        ])
        shared.atomic_text(self.output / "status.md", "\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run",), default="run")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--queue-state", type=Path, default=QUEUE)
    parser.add_argument("--filter-dir", type=Path, default=FILTER)
    parser.add_argument("--run-dir", type=Path, default=RUN)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(args.output_dir / ".matplotlib")
    with shared.lock_path(args.output_dir).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"already_running": True, "output": str(args.output_dir)}))
            return
        monitor = PerceptionMonitor(args.output_dir, args.run_dir, args.queue_state, args.filter_dir)
        def stop(signum, frame):
            monitor.stop_requested = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        monitor.run(interval=5.0, render_interval=60.0, once=args.once)


if __name__ == "__main__":
    main()
