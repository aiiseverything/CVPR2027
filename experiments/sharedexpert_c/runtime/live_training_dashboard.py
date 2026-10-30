#!/usr/bin/env python3
"""Live GPU/host-memory chart plus periodic training_metrics.png refresh.

Read-only with respect to the training job. Samples /sys/fs/cgroup and
nvidia-smi every 10 s, redraws monitoring_perception_v3/memory_live.png every
30 s, writes an auto-refreshing memory_live.html with a latest-values table,
and refreshes monitoring_perception_v3/training_metrics.png every 60 s from the
existing TensorBoard-backed PerceptionMonitor.

Chart form: two stacked time-series panels sharing one time axis (GPU memory;
host memory components). Series use the validated three-slot categorical
palette; thresholds are muted/warning reference lines with visible labels.
"""
from __future__ import annotations

import csv
import datetime as dt
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from zoneinfo import ZoneInfo

E = Path("/autodl-fs/data/CVPR2027/experiments/sharedexpert_c")
MON = E / "monitoring_perception_v3"
RUN = E / "runs/perception_v3"
FILTER = E / "data_pipeline/prepared/perception_v3_filter"
LOG = RUN / "training.log"
SAMPLES = MON / "memory_samples.jsonl"
PNG = MON / "memory_live.png"
HTML = MON / "memory_live.html"
STATE = MON / "state.json"
STATUS_MD = MON / "status.md"
LOCK = MON / ".live_dashboard.lock"
ERRORS = E / "runtime/live_training_dashboard_errors.log"
BJ = ZoneInfo("Asia/Shanghai")
GIB = 1024 ** 3
SAMPLE_SECONDS = 10
PLOT_SECONDS = 30
METRICS_SECONDS = 60
HOST_WATCHDOG_CAP = 92 * GIB
GPU_WATCHDOG = 76 * GIB
MAX_POINTS = 21600  # six hours at 1 sample/s equivalent; we plot the tail

C = {
    "surface": "#fcfcfb", "page": "#f9f9f7", "ink": "#0b0b0b",
    "secondary": "#52514e", "muted": "#898781", "grid": "#e1e0d9",
    "baseline": "#c3c2b7", "blue": "#2a78d6", "orange": "#eb6834",
    "aqua": "#1baf7a", "warning": "#fab219", "warning_ink": "#8a5a00",
}


def read_text(path):
    try:
        return Path(path).read_text()
    except OSError:
        return ""


def parse_stats():
    stats = {}
    for line in read_text("/sys/fs/cgroup/memory.stat").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit():
            stats[parts[0]] = int(parts[1])
    return stats


def host_watchdog(limit):
    return min(limit - 8 * GIB, HOST_WATCHDOG_CAP) if limit is not None else HOST_WATCHDOG_CAP


def launch_info():
    try:
        return json.loads((RUN / "fast_start.json").read_text())
    except (OSError, ValueError):
        return {}


def session_started_epoch():
    try:
        return dt.datetime.fromisoformat(launch_info()["started_at"]).timestamp()
    except (KeyError, ValueError, TypeError):
        return None


def resume_step():
    value = launch_info().get("resume_from_checkpoint")
    if value is None:
        try:
            state = json.loads((E / "runtime/perception_v3_pipeline.json").read_text())
            value = state["training"]["fast_start"]["resume_from_checkpoint"]
        except (OSError, ValueError, KeyError, TypeError):
            value = checkpoint_step()
    return int(value)


def training_status():
    import monitor_perception_v3 as mp
    state = mp.shared.read_json(E / "runtime/perception_v3_pipeline.json", {})
    stage, _ = mp.queue_health(state, run_dir=RUN)
    return "COMPLETE" if stage == "complete" else "RUNNING" if stage == "training" else "STOPPED"


def gpu_memory():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            text=True, timeout=5).strip().splitlines()
        name, used, total = [x.strip() for x in out[0].split(",")]
        return int(used) * 1024 ** 2, int(total) * 1024 ** 2, name
    except Exception:
        return None, None, "GPU 0"


def progress_step():
    text = ""
    try:
        started = session_started_epoch()
        if started is not None and LOG.stat().st_mtime < started:
            return checkpoint_step()
        with LOG.open("rb") as stream:
            stream.seek(max(0, LOG.stat().st_size - (256 << 10)))
            text = stream.read().decode(errors="replace")
    except OSError:
        pass
    matches = re.findall(r"Training Progress:\s*(\d+)%", text)
    step = int(matches[-1]) if matches else checkpoint_step()
    return max(0, min(100, step))


def checkpoint_step():
    path = RUN / "checkpoints/latest_checkpointed_iteration.txt"
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return 0


def sample():
    stats = parse_stats()
    current = int(read_text("/sys/fs/cgroup/memory.current").strip() or 0)
    limit_text = read_text("/sys/fs/cgroup/memory.max").strip()
    limit = None if limit_text in {"", "max"} else int(limit_text)
    used, total, gpu_name = gpu_memory()
    status = training_status()
    return {
        "time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "step": 100 if status == "COMPLETE" else progress_step(),
        "session_started_at": launch_info().get("started_at"),
        "resume_from_checkpoint": resume_step(),
        "host_watchdog": host_watchdog(limit),
        "gpu_name": gpu_name,
        "status": status,
        "checkpoint_step": checkpoint_step(),
        "alive": status == "RUNNING",
        "anon": int(stats.get("anon", 0)),
        "shmem": int(stats.get("shmem", 0)),
        "file": int(stats.get("file", 0)),
        "inactive_file": int(stats.get("inactive_file", 0)),
        "current": current,
        "limit": limit,
        "gpu_used": used,
        "gpu_total": total,
    }


def append_sample(sample_value):
    with SAMPLES.open("a") as stream:
        stream.write(json.dumps(sample_value) + "\n")


def load_samples():
    rows = []
    try:
        with SAMPLES.open() as stream:
            for line in stream:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        pass
    return rows[-MAX_POINTS:]


def bj_time(value=None):
    value = value or dt.datetime.now(dt.timezone.utc)
    return value.astimezone(BJ).strftime("%Y-%m-%d %H:%M:%S")


def gib(value):
    return None if value is None else value / GIB


def plot_memory(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    if not rows:
        return
    times = [dt.datetime.fromisoformat(row["time"]).astimezone(BJ) for row in rows]
    anon = [gib(row["anon"]) for row in rows]
    shmem = [gib(row["shmem"]) for row in rows]
    page = [gib(row["file"]) for row in rows]
    gpu = [gib(row["gpu_used"]) if row["gpu_used"] is not None else float("nan") for row in rows]
    total = next((gib(row["gpu_total"]) for row in reversed(rows) if row["gpu_total"]), 79.18)
    limit = next((gib(row["limit"]) for row in reversed(rows) if row["limit"]), 100.0)
    latest = rows[-1]
    latest_gpu = gib(latest["gpu_used"]) or 0.0
    latest_anon = gib(latest["anon"]) or 0.0
    latest_shmem = gib(latest["shmem"]) or 0.0
    latest_page = gib(latest["file"]) or 0.0
    status = latest.get("status", "RUNNING" if latest.get("alive") else "STOPPED")
    step = latest.get("step", 0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.0, 1.25]})
    fig.patch.set_facecolor(C["surface"])
    for ax in (ax1, ax2):
        ax.set_facecolor(C["surface"])
        ax.grid(True, color=C["grid"], linewidth=0.6, alpha=0.9)
        ax.tick_params(colors=C["muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_axisbelow(True)

    # GPU: one series, direct label at the left end; two labelled reference lines.
    ax1.plot(times, gpu, color=C["blue"], linewidth=2.0, solid_capstyle="round")
    ax1.annotate(f"GPU 0 used · {latest_gpu:.1f} GiB",
                 xy=(0.015, latest_gpu), xycoords=ax1.get_yaxis_transform(),
                 xytext=(8, 7), textcoords="offset points", ha="left", va="bottom",
                 fontsize=8.5, color=C["blue"], fontweight="semibold")
    ax1.axhline(total, color=C["baseline"], linewidth=1.0, linestyle="--")
    ax1.annotate(f"total {total:.1f} GiB", xy=(1.0, total), xycoords=ax1.get_yaxis_transform(),
                 xytext=(-6, 3), textcoords="offset points", ha="right", va="bottom",
                 fontsize=8, color=C["muted"],
                 bbox=dict(facecolor=C["surface"], edgecolor="none", pad=1.0, alpha=0.92))
    ax1.axhline(76, color=C["warning"], linewidth=1.2, linestyle="--")
    ax1.annotate("watchdog 76 GiB", xy=(1.0, 76), xycoords=ax1.get_yaxis_transform(),
                 xytext=(-6, -3), textcoords="offset points", ha="right", va="top",
                 fontsize=8, color=C["warning_ink"],
                 bbox=dict(facecolor=C["surface"], edgecolor="none", pad=1.0, alpha=0.92))
    ax1.set_ylim(0, max(total + 4, latest_gpu + 4))
    ax1.set_ylabel("GPU memory (GiB)", color=C["secondary"], fontsize=9)
    ax1.set_title("GPU 0", loc="left", fontsize=10, color=C["secondary"], pad=6)

    # Host: anon + pinned/offload stacked (the OOM-relevant budget); page cache is a
    # separate reclaimable line with a surface halo so it stays visible over the stack.
    ax2.stackplot(times, anon, shmem, colors=[C["blue"], C["orange"]],
                  labels=[f"anonymous · {latest_anon:.1f} GiB",
                          f"pinned / offload · {latest_shmem:.1f} GiB"],
                  edgecolor=C["surface"], linewidth=1.2)
    ax2.plot(times, page, color=C["surface"], linewidth=4.6, zorder=4, solid_capstyle="round")
    ax2.plot(times, page, color=C["aqua"], linewidth=2.0, zorder=5, solid_capstyle="round",
             label=f"page cache (reclaimable) · {latest_page:.1f} GiB")
    ax2.axhline(limit, color=C["baseline"], linewidth=1.0, linestyle="--")
    ax2.annotate(f"cgroup limit {limit:.0f} GiB", xy=(1.0, limit), xycoords=ax2.get_yaxis_transform(),
                 xytext=(-6, 3), textcoords="offset points", ha="right", va="bottom",
                 fontsize=8, color=C["muted"],
                 bbox=dict(facecolor=C["surface"], edgecolor="none", pad=1.0, alpha=0.92))
    threshold = gib(latest.get("host_watchdog", host_watchdog(latest.get("limit"))))
    ax2.axhline(threshold, color=C["warning"], linewidth=1.2, linestyle="--")
    ax2.annotate(f"anon+pinned watchdog {threshold:.0f} GiB", xy=(1.0, threshold), xycoords=ax2.get_yaxis_transform(),
                 xytext=(-6, -3), textcoords="offset points", ha="right", va="top",
                 fontsize=8, color=C["warning_ink"],
                 bbox=dict(facecolor=C["surface"], edgecolor="none", pad=1.0, alpha=0.92))
    ax2.set_ylim(0, max(limit + 8, max(anon[-1] + shmem[-1], page[-1]) + 8))
    ax2.set_ylabel("Host memory (GiB)", color=C["secondary"], fontsize=9)
    ax2.set_xlabel("Beijing time", color=C["secondary"], fontsize=9)
    ax2.set_title("Host: anonymous + pinned offload, and reclaimable page cache", loc="left",
                  fontsize=10, color=C["secondary"], pad=6)
    if len(times) >= 2:
        indices = sorted({0, len(times) // 4, len(times) // 2, (3 * len(times)) // 4, len(times) - 1})
        ax2.set_xticks([times[i] for i in indices])
        ax2.set_xticklabels([times[i].strftime("%H:%M:%S") for i in indices])
        ax2.tick_params(axis="x", labelsize=8, colors=C["muted"])

    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.055, 0.905),
               ncol=3, frameon=False, fontsize=8.5, labelcolor=C["secondary"],
               handlelength=1.8, columnspacing=2.0)

    fig.suptitle("perception_v3 training · live GPU and host memory", x=0.055, y=0.985,
                 ha="left", fontsize=15, color=C["ink"], weight="semibold")
    fig.text(0.055, 0.94,
             f"step {step}/100 · {status} · {latest.get('gpu_name', 'GPU 0')} · samples {SAMPLE_SECONDS}s · "
             f"chart updated {bj_time()} (Asia/Shanghai)",
             fontsize=9, color=C["secondary"], ha="left")
    fig.text(0.055, 0.015,
             "Source: /sys/fs/cgroup/memory.* and nvidia-smi. Page cache is reclaimable; "
             "the host watchdog counts anonymous + pinned/offload memory only.",
             fontsize=7.5, color=C["muted"], ha="left")
    fig.subplots_adjust(top=0.845, bottom=0.10, left=0.075, right=0.965, hspace=0.30)
    temporary = MON / ".memory_live.png.tmp"
    fig.savefig(temporary, format="png", dpi=150, facecolor=C["surface"], bbox_inches="tight")
    plt.close(fig)
    temporary.replace(PNG)


def write_html(rows):
    if not rows:
        return
    latest = rows[-1]
    used = gib(latest["gpu_used"]) or 0.0
    total = gib(latest["gpu_total"]) or 79.18
    anon = gib(latest["anon"]) or 0.0
    shmem = gib(latest["shmem"]) or 0.0
    page = gib(latest["file"]) or 0.0
    current = gib(latest["current"]) or 0.0
    limit = gib(latest["limit"]) or 100.0
    threshold = gib(latest.get("host_watchdog", host_watchdog(latest.get("limit"))))
    watchdog_headroom = threshold - (anon + shmem)
    status = latest.get("status", "RUNNING" if latest.get("alive") else "STOPPED")
    table = [
        ("GPU 0 used / total", f"{used:.1f} / {total:.1f} GiB"),
        ("Host anonymous", f"{anon:.1f} GiB"),
        ("Pinned / offload (shmem)", f"{shmem:.1f} GiB"),
        ("Anon + pinned (watchdog metric)", f"{anon + shmem:.1f} GiB"),
        (f"Host watchdog headroom ({threshold:.0f} GiB)", f"{watchdog_headroom:.1f} GiB"),
        ("Page cache (reclaimable)", f"{page:.1f} GiB"),
        ("Cgroup current / limit", f"{current:.1f} / {limit:.1f} GiB"),
        ("Training step", f"{latest.get('step', 0)} / 100"),
        ("Status", status),
    ]
    body = "\n".join(f"<tr><th>{name}</th><td>{value}</td></tr>" for name, value in table)
    metrics_png = MON / "training_metrics.png"
    metrics_version = metrics_png.stat().st_mtime_ns if metrics_png.exists() else 0
    HTML.write_text(f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta http-equiv="refresh" content="15">
<title>perception_v3 训练曲线与实时资源</title>
<style>
:root {{ --surface:{C['surface']}; --page:{C['page']}; --ink:{C['ink']};
  --secondary:{C['secondary']}; --muted:{C['muted']}; --border:{C['grid']};
  --blue:{C['blue']}; --orange:{C['orange']}; --aqua:{C['aqua']}; }}
body {{ margin:0; padding:28px; background:var(--page); color:var(--ink);
  font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
main {{ max-width:1180px; margin:0 auto; background:var(--surface);
  border:1px solid var(--border); border-radius:12px; padding:24px 28px; }}
h1 {{ font-size:20px; margin:0 0 4px; font-weight:600; }}
p.sub {{ color:var(--secondary); font-size:13px; margin:0 0 18px; }}
img {{ width:100%; height:auto; border:1px solid var(--border); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; margin-top:18px; font-size:13px;
  font-variant-numeric: tabular-nums; }}
th, td {{ text-align:left; padding:7px 10px; border-bottom:1px solid var(--border); }}
th {{ color:var(--secondary); font-weight:500; width:46%; }}
td {{ color:var(--ink); }}
.links {{ margin-top:16px; font-size:13px; color:var(--secondary); }}
a {{ color:var(--blue); text-decoration:none; }}
.dot {{ display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; }}
.note {{ color:var(--muted); font-size:12px; margin-top:14px; }}
</style></head>
<body><main>
<h1><span class="dot" style="background:{C['blue']}"></span>perception_v3 训练曲线 · {latest.get('step', 0)}/100</h1>
<p class="sub">北京时间 {bj_time()} · 训练曲线每 60 秒更新，本页每 15 秒刷新 · 状态 {status}</p>
<img src="training_metrics.png?v={metrics_version}" alt="训练指标曲线：有效历史及本次续训">
<p class="note">曲线保留 checkpoint {latest.get('resume_from_checkpoint', 50)} 及以前的历史；之后仅显示本次续训指标。新点在每个训练步完成后出现。</p>
<h2>实时资源</h2>
<img src="memory_live.png?ts={int(time.time())}" alt="GPU 与 host 内存实时曲线">
<table>{body}</table>
<p class="links">[<a href="memory_live.png">memory_live.png</a>] ·
[<a href="training_metrics.png?v={metrics_version}">training_metrics.png</a>] ·
[<a href="status.md">status.md</a>] ·
[<a href="../runtime/perception_v3_pipeline.json">pipeline state</a>]</p>
<p class="note">采集：/sys/fs/cgroup/memory.* 与 nvidia-smi。page cache 可回收；host watchdog 只统计 anon + pinned/offload。</p>
</main></body></html>
""")


def latest_metrics(max_step=None):
    values = {}
    started = session_started_epoch()
    try:
        with (MON / "metrics.csv").open() as stream:
            for row in csv.DictReader(stream):
                try:
                    step = int(row["step"])
                    value = float(row["value"])
                    if started is not None and float(row["wall_time"]) < started:
                        continue
                except (ValueError, KeyError):
                    continue
                if not (value == value) or value in (float("inf"), float("-inf")):
                    continue
                if max_step is not None and step > max_step:
                    continue
                key = row["tag"]
                if key not in values or step >= values[key][0]:
                    values[key] = (step, value)
    except OSError:
        pass
    return values


def render_metrics():
    sys.path.insert(0, str(E))
    import monitor_perception_v3 as mp
    monitor = mp.PerceptionMonitor(MON, RUN)
    status = training_status()
    started = session_started_epoch()
    restored = resume_step()
    monitor.state.update(display_status=status, experiment_stage="complete" if status == "COMPLETE" else "training",
                         experiment_teacher=f"perception_v3 | resumed from checkpoint {restored}, current step {progress_step()}",
                         experiment_error=None,
                         status_reason="Checkpoint history plus current-session metrics; earlier unsaved steps excluded.")
    if started is None:
        raise RuntimeError("Missing launch timestamp; cannot distinguish resumed metrics from discarded steps")
    monitor.render(reason="live", resume_step=restored, resumed_at=started)


def write_dashboard(rows):
    if not rows:
        return
    latest = rows[-1]
    current_state = {}
    try:
        current_state = json.loads(STATE.read_text())
    except (OSError, ValueError):
        pass
    step = latest.get("step", 0)
    metrics = latest_metrics(step)
    reward = metrics.get("critic/rewards/mean") or metrics.get("critic/score/mean")
    loss = metrics.get("actor/pg_loss")
    grad = metrics.get("actor/grad_norm")
    response = metrics.get("response_length/mean")
    step_time = metrics.get("timing_s/step") or metrics.get("perf/time_per_step")
    status = latest.get("status", "RUNNING" if latest.get("alive") else "STOPPED")
    stage = "complete" if status == "COMPLETE" else "training" if latest.get("alive") else "stopped"
    reason = ("Single-GPU bf16-master fast-start with manual param/optimizer offload (no CPUOffloadPolicy), "
              "OMP=8, rollout KV 0.82, enforce_eager=True (CUDA graphs off; step-69 hang mitigation), "
              f"resumed from checkpoint {latest.get('resume_from_checkpoint', 50)}. "
              f"Current session started {latest.get('session_started_at')}; displayed scalar values are from this session only.")
    current_state.update({
        "heartbeat_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "last_sample_at": latest["time"],
        "experiment_stage": stage,
        "latest_metric_steps": {"perception_v3": step if metrics else None},
        "display_status": status,
        "experiment_error": None,
        "experiment_teacher": "perception_v3",
        "status_reason": reason,
        "experiment_signature": [stage, None, reason],
        "plot_banner": f"{status} | stage={stage} | teacher=perception_v3 | step {step}/100",
        "screening": json.loads((FILTER / "status.json").read_text()),
        "training_live": {
            "step": step, "checkpoint_step": latest.get("checkpoint_step", 0),
            "gpu_used_gib": round(gib(latest["gpu_used"]) or 0, 2),
            "gpu_total_gib": round(gib(latest["gpu_total"]) or 0, 2),
            "anonymous_gib": round(gib(latest["anon"]) or 0, 2),
            "pinned_offload_gib": round(gib(latest["shmem"]) or 0, 2),
            "page_cache_gib": round(gib(latest["file"]) or 0, 2),
            "cgroup_current_gib": round(gib(latest["current"]) or 0, 2),
            "cgroup_limit_gib": round(gib(latest["limit"]) or 0, 2),
            "host_watchdog_gib": gib(latest.get("host_watchdog", host_watchdog(latest.get("limit")))), "gpu_watchdog_gib": 76,
            "session_started_at": latest.get("session_started_at"),
            "resume_from_checkpoint": latest.get("resume_from_checkpoint"),
            "reward": reward[1] if reward else None,
            "pg_loss": loss[1] if loss else None,
            "grad_norm": grad[1] if grad else None,
            "response_length": response[1] if response else None,
            "step_time_s": step_time[1] if step_time else None,
        },
    })
    current_state.setdefault("progress", {}).setdefault("perception_v3", {})["checkpoint_step"] = latest.get("checkpoint_step", 0)
    current_state["progress"]["perception_v3"]["exported"] = (RUN / "hf/global_step_100/sharedexpert_export_complete.json").is_file()
    temporary = STATE.with_name(STATE.name + ".live_writing")
    temporary.write_text(json.dumps(current_state, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(STATE)

    def fmt(value, digits=4):
        return "—" if value is None else f"{value:.{digits}f}"

    live = current_state["training_live"]
    md = [
        "# perception_v3 单 GPU 训练实时状态", "",
        f"更新时间（北京时间）：{bj_time()}", "",
        f"状态：**{status}**；阶段：`{stage}`；step **{step}/100**；checkpoint {live['checkpoint_step']}。", "",
        f"说明：{reason}", "",
        "## 实时资源", "",
        "| 指标 | 当前值 | 阈值 |", "|---|---:|---:|",
        f"| GPU 0 显存 | {live['gpu_used_gib']:.1f} / {live['gpu_total_gib']:.1f} GiB | watchdog 76 GiB |",
        f"| Host anonymous | {live['anonymous_gib']:.1f} GiB | — |",
        f"| Pinned / offload (shmem) | {live['pinned_offload_gib']:.1f} GiB | — |",
        f"| anon + pinned | {live['anonymous_gib'] + live['pinned_offload_gib']:.1f} GiB | watchdog {live['host_watchdog_gib']:.0f} GiB |",
        f"| Page cache（可回收） | {live['page_cache_gib']:.1f} GiB | — |",
        f"| cgroup current / limit | {live['cgroup_current_gib']:.1f} / {live['cgroup_limit_gib']:.1f} GiB | — |", "",
        "## 本次续训指标（TensorBoard 实际 scalar；尚无新步数时显示 —）", "",
        f"- reward：{fmt(live['reward'])}",
        f"- pg_loss：{fmt(live['pg_loss'])}",
        f"- grad_norm：{fmt(live['grad_norm'])}",
        f"- response_length：{fmt(live['response_length'], 1)}",
        f"- step_time：{fmt(live['step_time_s'], 2)} s", "",
        "## 图与链接", "",
        "- [实时显存/内存图](memory_live.png)",
        "- [实时训练曲线与资源（每 15 秒刷新）](memory_live.html)",
        "- [训练指标图](training_metrics.png)",
        "- [流程状态](../runtime/perception_v3_pipeline.json)",
        "- [筛选日志](../data_pipeline/prepared/perception_v3_filter/screening.log)", "",
        "图每 30 s 重绘，训练指标图每 60 s 从 TensorBoard 重绘；资源每 10 s 采样。", "",
        f"训练曲线保留 checkpoint {latest.get('resume_from_checkpoint', 50)} 及以前的历史，其后只显示本次续训指标；新点在每个完整训练步结束后出现。", "",
        "单 GPU 恢复运行：batch 30、group 8、100 steps、LR 1e-6、thinking 不变；仅拓扑与内存/并发参数适配。", "",
    ]
    temporary = STATUS_MD.with_name(STATUS_MD.name + ".live_writing")
    temporary.write_text("\n".join(md))
    temporary.replace(STATUS_MD)


def main():
    MON.mkdir(parents=True, exist_ok=True)
    lock = LOCK.open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    last_plot = 0.0
    last_metrics = 0.0
    while True:
        try:
            value = sample()
            append_sample(value)
            rows = load_samples()
            now = time.monotonic()
            if now - last_metrics >= METRICS_SECONDS:
                try:
                    render_metrics()
                except Exception as exc:
                    with ERRORS.open("a") as stream:
                        stream.write(json.dumps({"time": dt.datetime.now(dt.timezone.utc).isoformat(),
                                                 "component": "metrics",
                                                 "error": f"{type(exc).__name__}: {exc}"}) + "\n")
                last_metrics = now
            if now - last_plot >= PLOT_SECONDS:
                plot_memory(rows)
                write_html(rows)
                last_plot = now
            write_dashboard(rows)
        except Exception as exc:
            with ERRORS.open("a") as stream:
                stream.write(json.dumps({"time": dt.datetime.now(dt.timezone.utc).isoformat(),
                                         "component": "loop",
                                         "error": f"{type(exc).__name__}: {exc}"}) + "\n")
        time.sleep(SAMPLE_SECONDS)


if __name__ == "__main__":
    main()
