#!/usr/bin/env python3
"""Read-only, durable training monitor. No process termination or GPU allocation.

Use the system /opt/venv/main/bin/python3; the frozen training venv is untouched.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

HERE = Path(__file__).resolve().parent
ROLES = ("perception", "reasoning", "generalist")
DEFAULT_OUTPUT = HERE / "monitoring"
HISTORICAL_EVENTS = {"oom": 19, "oom_kill": 12}
ACTIVE_STAGES = {"initializing_monitor", "waiting_for_smoke", "validating_smoke", "validating_preparation",
                 "preflight", "training", "evaluation", "finishing_baseline", "summarizing", "restoring_previous_services",
                 "restoring_baseline_service"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
GROUPS = {
    "reward": ("Reward / score (actual scalar tags)", lambda t: t in {"critic/rewards/mean", "critic/score/mean"}),
    "accuracy": ("Accuracy (only if logged)", lambda t: bool(re.search(r"(^|/)accuracy($|/)", t))),
    "response_length": ("Response length (tokens)", lambda t: t in {"response_length/mean", "response_length/max"}),
    "pg_loss": ("Policy gradient loss", lambda t: t == "actor/pg_loss"),
    "grad_norm": ("Gradient norm", lambda t: t == "actor/grad_norm"),
    "kl_diagnostic": ("KL diagnostics (regularization remains off)", lambda t: t in {"actor/ppo_kl", "actor/kl_loss", "rollout_corr/kl", "rollout_corr/k3_kl"}),
    "importance_sampling": ("Importance sampling diagnostics", lambda t: t in {"rollout_corr/rollout_is_mean", "rollout_corr/rollout_is_max", "rollout_corr/rollout_is_eff_sample_size"}),
    "step_time": ("Step time (seconds)", lambda t: t in {"timing_s/step", "perf/time_per_step"}),
}


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def atomic_json(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".monitor-", delete=False) as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_text(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=".monitor-", delete=False) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    temporary.replace(path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def counters(path):
    return {key: int(value) for key, value in (line.split() for line in Path(path).read_text().splitlines())}


def cgroup_sample(root=Path("/sys/fs/cgroup")):
    current = int((root / "memory.current").read_text())
    limit = (root / "memory.max").read_text().strip()
    stats = counters(root / "memory.stat")
    inactive = stats.get("inactive_file", 0)
    return {"scope": "whole shared container; not teacher attribution",
            "limit_bytes": None if limit == "max" else int(limit), "current_bytes": current,
            "inactive_file_bytes": inactive, "working_set_bytes": max(0, current - inactive),
             "memory_stat": stats,
             "anon_bytes": stats.get("anon"), "file_bytes": stats.get("file"),
             "active_file_bytes": stats.get("active_file"), "shmem_bytes": stats.get("shmem"),
             "slab_unreclaimable_bytes": stats.get("slab_unreclaimable"),
             "memory_pressure": (root / "memory.pressure").read_text().strip(),
            "events": counters(root / "memory.events"), "cpu_max": (root / "cpu.max").read_text().strip()}


def counter_delta(current, previous):
    # Reset is recorded separately, never interpreted as negative OOM events.
    return {key: max(0, value - previous.get(key, value)) for key, value in current.items()}


def gpu_sample():
    fields = ("index", "uuid", "memory.used", "memory.total", "utilization.gpu", "utilization.memory")
    result = subprocess.run(["nvidia-smi", "--query-gpu=" + ",".join(fields), "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, timeout=4, check=True)
    rows = []
    for values in csv.reader(io.StringIO(result.stdout)):
        raw = dict(zip(fields, (x.strip() for x in values)))
        def number(key):
            try:
                return int(raw[key])
            except (KeyError, ValueError):
                return None
        rows.append({"index": number("index"), "uuid": raw.get("uuid"),
                     "memory_used_mib": number("memory.used"), "memory_total_mib": number("memory.total"),
                     "gpu_utilization_percent": number("utilization.gpu"), "memory_utilization_percent": number("utilization.memory")})
    return rows


def start_ticks(pid):
    try:
        fields = (Path("/proc") / str(int(pid)) / "stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, ValueError, TypeError, IndexError):
        return None


def process_alive(pid, expected_ticks=None):
    try:
        cmdline = (Path("/proc") / str(int(pid)) / "cmdline").read_bytes()
        return (str(Path(__file__).resolve()).encode() in cmdline and b"run" in cmdline.split(b"\0")
                and (expected_ticks is None or start_ticks(pid) == expected_ticks))
    except (OSError, TypeError, ValueError):
        return False


def experiment_processes(driver, run_dirs):
    all_processes = {}
    roots = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes().split(b"\0")
            args = [x.decode(errors="replace") for x in raw if x]
            if not args or int(entry.name) == os.getpid():
                continue
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
            item = {"pid": int(entry.name), "ppid": int(fields[1]), "state": fields[0], "start_ticks": int(fields[19]), "argv": args}
            all_processes[item["pid"]] = item
            command = " ".join(args)
            if (str(HERE / "training/run.py") in args and any(str(path) + "/" in command for path in run_dirs.values())):
                roots.add(item["pid"])
            if str(HERE / "training/launch.py") in args and "--execute" in args:
                roots.add(item["pid"])
        except (OSError, ValueError, IndexError):
            continue
    selected = set(roots)
    while True:
        children = {pid for pid, item in all_processes.items() if item["ppid"] in selected}
        if children <= selected:
            break
        selected |= children
    return {"trainer_roots": sorted(roots), "trainer_processes": [all_processes[p] for p in sorted(selected)],
            "driver": all_processes.get(driver.get("driver_pid"))}


def classify_oom(line, driver_log=False):
    if driver_log and re.search(r"(94%.*memory|working set exceeded|working.set.*94%|Cgroup working set)", line, re.I):
        return "preventive_memory_guard"
    if re.search(r"CUDA out of memory|torch\.cuda\.OutOfMemoryError|CUDA error:\s*out of memory", line, re.I):
        return "cuda_oom_log"
    if re.search(r"ray\.exceptions\.OutOfMemoryError|OutOfMemoryError:|node was low on memory|killed due to memory pressure|workers?.*killed.*memory|std::bad_alloc|Cannot allocate memory", line, re.I):
        return "host_or_ray_oom_log"
    if re.search(r"\bMemoryError\b", line):
        return "host_memory_error_log"
    return None


class Monitor:
    def __init__(self, output, run_dirs, include_driver=True, test_mode=False):
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        for name in ("samples.jsonl", "alerts.jsonl", "errors.jsonl"):
            (self.output / name).touch(exist_ok=True)
        self.run_dirs = run_dirs
        self.include_driver = include_driver
        self.test_mode = test_mode
        self.state_path = self.output / "state.json"
        self.state = read_json(self.state_path, {})
        self.state.setdefault("created_at", timestamp())
        self.state.setdefault("historical_shared_container_reference", HISTORICAL_EVENTS)
        self.state.setdefault("log_cursors", {})
        self.state.setdefault("alert_count", 0)
        self.state.setdefault("error_count", 0)
        # Load IDs as well as persisted offsets, so a crash between alert append
        # and state write cannot emit the same byte-offset alert twice.
        self.alert_ids = set()
        try:
            for line in (self.output / "alerts.jsonl").open():
                try:
                    self.alert_ids.add(json.loads(line)["id"])
                except (ValueError, KeyError):
                    pass
        except FileNotFoundError:
            pass
        self.error_seen = {}
        self.state.update(pid=os.getpid(), start_ticks=start_ticks(os.getpid()), status="running", instance_started_at=timestamp(),
                          python_executable=sys.executable, test_mode=test_mode,
                          run_dirs={k: str(v) for k, v in run_dirs.items()})
        self.state["environment"] = {name: importlib.metadata.version(name) for name in ("tensorboard", "matplotlib", "numpy")}
        self.stop_requested = False

    def save(self):
        self.state.update(heartbeat_at=timestamp(), heartbeat_epoch=time.time())
        try:
            self.write_status()
        except Exception as exc:
            self.error("status_markdown", exc)
        atomic_json(self.state_path, self.state)

    def write_status(self):
        state = self.state
        lines = ["# SharedExpert 训练监控状态", "", f"更新时间（UTC）：{state.get('heartbeat_at')}", "",
                 f"状态：**{state.get('display_status', 'UNKNOWN')}**；阶段：`{state.get('experiment_stage', 'unknown')}`；"
                 f"教师：`{state.get('experiment_teacher') or '—'}`。", "",
                 "原因：" + (state.get("experiment_error") or "未记录失败原因。"), "",
                 "| 教师 | 最新 TensorBoard step | 已保存 checkpoint step |", "|---|---:|---:|"]
        for role in self.run_dirs:
            step = state.get("latest_metric_steps", {}).get(role)
            checkpoint = state.get("progress", {}).get(role, {}).get("checkpoint_step", 0)
            lines.append(f"| {role} | {step if step is not None else '尚无 scalar'} | {checkpoint} |")
        cg = state.get("cgroup", {})
        delta = cg.get("delta_since_monitor_first_start", {})
        lines += ["", "共享容器 OOM 增量（相对本实验首次监控基线；不直接归因教师）："
                  + json.dumps({k: delta.get(k) for k in ("oom", "oom_kill")}, ensure_ascii=False), "",
                  "首次基线：" + json.dumps(state.get("baseline_events", {}), ensure_ascii=False), ""]
        for label, key in (("最近告警", "last_alert"), ("最近监控错误", "last_error")):
            lines += [label + "：", "```json", json.dumps(state.get(key), ensure_ascii=False, indent=2), "```", ""]
        lines += ["曲线：[training_metrics.png](training_metrics.png)。历史采样：[samples.jsonl](samples.jsonl)。",
                  "监控存活不代表训练正在运行；checkpoint 与 scalar step 分开记录。", ""]
        atomic_text(self.output / "status.md", "\n".join(lines))

    def observe_experiment(self, driver, sample):
        stage = sample["experiment_stage"]
        teacher = sample.get("experiment_teacher")
        observed = sample.get("driver")
        expected = self.state.get("driver_identity", {})
        pid = driver.get("driver_pid")
        ticks = driver.get("driver_start_ticks")
        if ticks is None and expected.get("pid") == pid:
            ticks = expected.get("start_ticks")
        # A matching PID alone is insufficient: reject zombies and PID reuse.
        alive = bool(observed and observed.get("state") not in {"Z", "X"}
                     and str(HERE / "run_experiment.py") in observed.get("argv", [])
                     and "run" in observed.get("argv", [])
                     and (ticks is None or observed.get("start_ticks") == ticks))
        if alive:
            self.state["driver_identity"] = {"pid": pid, "start_ticks": observed["start_ticks"]}
            ticks = observed["start_ticks"]
        health = "alive" if alive else "not_running" if "driver" in sample else "unknown"
        error = driver.get("error")
        display = "RUNNING" if stage in ACTIVE_STAGES else "COMPLETE" if stage == "complete" else "UNKNOWN"
        kind = None
        if stage in {"failed", "interrupted"}:
            display, kind = "STOPPED", "training_stopped"
        elif stage in ACTIVE_STAGES and health == "not_running" and self.include_driver:
            display, kind = "STOPPED", "driver_unexpected_exit"
            error = f"Driver PID {pid} is absent or its command/start_ticks does not match; last recorded stage={stage}."
        elif stage == "teachers_evaluated_baseline_pending":
            display = "INCOMPLETE"
        if kind:
            identity = json.dumps([pid, ticks, stage, driver.get("updated_at"), error], sort_keys=True)
            self.alert(kind, identity, source=str(HERE / "runtime/experiment_state.json"),
                       experiment_stage=stage, teacher=teacher, driver_pid=pid, driver_start_ticks=ticks,
                       state_updated_at=driver.get("updated_at"), reason=error,
                       attribution="driver state/process identity evidence; does not imply OOM")
        signature = [stage, teacher, display, error]
        sample.update(driver_health=health, experiment_error=error, display_status=display,
                      stage_changed=signature != self.state.get("experiment_signature"))
        self.state.update(experiment_stage=stage, experiment_teacher=teacher, experiment_error=error,
                          driver_health=health, display_status=display, experiment_signature=signature)

    def error(self, component, exc):
        text = f"{type(exc).__name__}: {exc}"
        self.state["last_error"] = {"time": timestamp(), "component": component, "message": text}
        key = (component, text)
        # Repeated environmental errors remain visible without 5-second spam.
        if time.time() - self.error_seen.get(key, 0) >= 60:
            append_json(self.output / "errors.jsonl", self.state["last_error"])
            self.state["error_count"] += 1
            self.error_seen[key] = time.time()

    def alert(self, kind, identity, **fields):
        alert_id = hashlib.sha256((kind + "\0" + identity).encode()).hexdigest()
        if alert_id in self.alert_ids:
            return
        record = {"id": alert_id, "observed_at": timestamp(), "kind": kind, "test_mode": self.test_mode, **fields}
        append_json(self.output / "alerts.jsonl", record)
        self.alert_ids.add(alert_id)
        self.state["alert_count"] = len(self.alert_ids)
        self.state["last_alert"] = record

    def scan_log(self, path, driver_log=False):
        path = Path(path)
        if not path.exists():
            return
        stat = path.stat()
        key = str(path.resolve())
        cursor = self.state["log_cursors"].get(key)
        identity = [stat.st_dev, stat.st_ino]
        if cursor is None:
            # Existing driver history includes smoke/probes. Do not alarm on
            # those. Formal training logs are separately read from their start.
            cursor = {"identity": identity, "offset": stat.st_size if driver_log else 0,
                      "generation": 0, "context": []}
        elif cursor["identity"] != identity or stat.st_size < cursor["offset"]:
            cursor = {"identity": identity, "offset": 0, "generation": cursor.get("generation", 0) + 1, "context": []}
        with path.open("rb") as stream:
            stream.seek(cursor["offset"])
            # Bound IO per tick; remaining content is processed next sample.
            for _ in range(5000):
                offset = stream.tell()
                raw = stream.readline(65536)
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    if len(raw) < 65536:
                        stream.seek(offset)
                        break
                line = ANSI.sub("", raw.decode(errors="replace")).rstrip()
                kind = classify_oom(line, driver_log)
                if kind:
                    self.alert(kind, f"{key}:{identity}:{cursor['generation']}:{offset}", source=key,
                               byte_offset=offset, context=cursor["context"][-3:] + [line[-2000:]],
                               attribution="explicit log evidence; observed timestamp, not reconstructed event time")
                cursor["context"] = (cursor["context"] + [line[-2000:]])[-3:]
            cursor["offset"] = stream.tell()
        self.state["log_cursors"][key] = cursor

    def sample(self):
        now = timestamp()
        driver = read_json(HERE / "runtime/experiment_state.json", {}) if self.include_driver else {}
        sample = {"time": now, "epoch": time.time(), "pid": os.getpid(), "test_mode": self.test_mode,
                  "experiment_stage": driver.get("stage", "unknown"), "experiment_teacher": driver.get("teacher")}
        try:
            cg = cgroup_sample()
            events = cg["events"]
            self.state.setdefault("baseline_events", dict(events))
            self.state.setdefault("baseline_observed_at", now)
            self.state.setdefault("last_events", dict(events))
            self.state.setdefault("instance_baseline_events", dict(events))
            previous = self.state["last_events"]
            cg["delta_since_previous_sample"] = counter_delta(events, previous)
            cg["delta_since_monitor_first_start"] = counter_delta(events, self.state["baseline_events"])
            cg["counter_reset_keys"] = [k for k, v in events.items() if v < previous.get(k, v)]
            if any(cg["delta_since_previous_sample"].get(k, 0) for k in ("oom", "oom_kill", "oom_group_kill")):
                self.alert("cgroup_oom_counter_increment", json.dumps([previous, events], sort_keys=True),
                           previous_events=previous, current_events=events,
                           delta=cg["delta_since_previous_sample"],
                           interval_start=self.state.get("last_sample_at"), interval_end=now,
                           attribution="shared container only; this does not identify a teacher or process")
            sample["cgroup"] = cg
            self.state["last_events"] = events
            self.state["cgroup"] = cg
        except Exception as exc:
            self.error("cgroup", exc)
        try:
            sample["gpus"] = gpu_sample()
        except Exception as exc:
            self.error("nvidia_smi", exc)
        try:
            sample.update(experiment_processes(driver, self.run_dirs))
        except Exception as exc:
            self.error("processes", exc)
        progress = {}
        for role, directory in self.run_dirs.items():
            try:
                checkpoint = directory / "checkpoints/latest_checkpointed_iteration.txt"
                progress[role] = {"checkpoint_step": int(checkpoint.read_text()) if checkpoint.exists() else 0,
                                  "exported": (directory / "hf/global_step_300/sharedexpert_export_complete.json").exists()}
                self.scan_log(directory / "training.log")
            except Exception as exc:
                self.error("training_log/" + role, exc)
        if self.include_driver:
            try:
                self.scan_log(HERE / "runtime/experiment.log", driver_log=True)
            except Exception as exc:
                self.error("driver_log", exc)
        sample["progress"] = progress
        self.observe_experiment(driver, sample)
        self.state.update(progress=progress, last_sample_at=now)
        append_json(self.output / "samples.jsonl", sample)
        self.save()
        return sample

    def metrics(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        rows = []
        inventories = {}
        for role, directory in self.run_dirs.items():
            tb = directory / "tensorboard"
            if not tb.exists() or not any(tb.glob("events.out.tfevents.*")):
                inventories[role] = {"status": "waiting_for_tensorboard", "tags": [], "latest_step": None}
                continue
            try:
                accumulator = EventAccumulator(str(tb), size_guidance={"scalars": 0, "tensors": 0}, purge_orphaned_data=True)
                accumulator.Reload()
                tags = accumulator.Tags().get("scalars", [])
                latest = None
                for tag in sorted(tags):
                    # A resumed writer may repeat a step; retain the latest
                    # actual event for that (role,tag,step), never average runs.
                    events = {}
                    for event in accumulator.Scalars(tag):
                        if event.step not in events or event.wall_time >= events[event.step].wall_time:
                            events[event.step] = event
                    for step, event in sorted(events.items()):
                        latest = step if latest is None else max(latest, step)
                        if not math.isfinite(float(event.value)):
                            self.alert("nonfinite_scalar", f"{role}:{tag}:{step}:{event.wall_time}",
                                       source=str(tb), run=role, tag=tag, step=step,
                                       value=repr(float(event.value)), wall_time=event.wall_time,
                                       attribution="actual TensorBoard event; retained in CSV, excluded from finite plotting")
                        rows.append({"run": role, "step": step, "tag": tag, "value": float(event.value),
                                     "wall_time": event.wall_time, "wall_time_utc": dt.datetime.fromtimestamp(event.wall_time, dt.timezone.utc).isoformat(),
                                     "tensorboard_dir": str(tb), "test_mode": self.test_mode})
                inventories[role] = {"status": "observed" if tags else "waiting_for_scalars", "tags": tags,
                                     "tensor_tags_not_used_as_scalar": accumulator.Tags().get("tensors", []), "latest_step": latest}
            except Exception as exc:
                inventories[role] = {"status": "read_error", "message": str(exc), "latest_step": None}
                self.error("tensorboard/" + role, exc)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=self.output, prefix=".metrics-", delete=False) as stream:
            writer = csv.DictWriter(stream, fieldnames=["run", "step", "tag", "value", "wall_time", "wall_time_utc", "tensorboard_dir", "test_mode"])
            writer.writeheader()
            writer.writerows(rows)
            temporary = Path(stream.name)
        temporary.replace(self.output / "metrics.csv")
        atomic_json(self.output / "scalar_inventory.json", inventories)
        return rows, inventories

    def render(self, reason="periodic"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            rows, inventories = self.metrics()
            series = {}
            for row in rows:
                if math.isfinite(row["value"]):
                    series.setdefault((row["run"], row["tag"]), []).append((row["step"], row["value"]))
            fig, axes = plt.subplots(4, 2, figsize=(15, 15), constrained_layout=True)
            colors = {role: color for role, color in zip(self.run_dirs, ("#1976d2", "#d84315", "#388e3c"))}
            groups_present = {}
            display = self.state.get("display_status", "UNKNOWN")
            empty_label = "STOPPED" if display == "STOPPED" else "WAITING" if display == "RUNNING" else "NO DATA"
            for ax, (group, (title, predicate)) in zip(axes.flat, GROUPS.items()):
                matching = [(key, values) for key, values in sorted(series.items()) if predicate(key[1])]
                groups_present[group] = [f"{run}:{tag}" for (run, tag), _ in matching]
                for number, ((run, tag), values) in enumerate(matching):
                    points = sorted(values)
                    ax.plot([p[0] for p in points], [p[1] for p in points], marker="o", markersize=3,
                            linewidth=1.3, color=colors.get(run), linestyle=("-", "--", ":", "-.")[number % 4], label=f"{run}: {tag}")
                ax.set_title(title, fontsize=11)
                ax.set_xlabel("Optimizer step")
                ax.grid(True, alpha=0.2)
                if matching:
                    ax.legend(fontsize=6, loc="best")
                else:
                    ax.text(0.5, 0.5, empty_label + ": no matching finite TensorBoard scalar events\nNo values inferred or fabricated", ha="center", va="center", transform=ax.transAxes, fontsize=9,
                            color="#b71c1c" if display == "STOPPED" else "#333333")
            scope = "SELF-TEST / explicit archived run; not formal results" if self.test_mode else "Formal teachers only; no smoke data"
            banner = f"{display} | stage={self.state.get('experiment_stage', 'unknown')} | teacher={self.state.get('experiment_teacher') or 'none'}"
            reason_summary = " ".join((self.state.get("experiment_error") or "").split())[:240]
            if reason_summary:
                banner += "\n" + "\n".join(textwrap.wrap(reason_summary, width=120))
            fig.suptitle("SharedExpert training metrics\n" + banner + "\n" + scope + "\n" + timestamp(), fontsize=12,
                         color="#b71c1c" if display == "STOPPED" else "#222222")
            temp_png = self.output / ".training_metrics.png.tmp"
            fig.savefig(temp_png, format="png", dpi=110)
            plt.close(fig)
            temp_png.replace(self.output / "training_metrics.png")
            self.state.update(last_render_at=timestamp(), last_render_epoch=time.time(), render_reason=reason,
                              render_status="stopped" if display == "STOPPED" else "rendered" if rows else "waiting_for_training_data" if display == "RUNNING" else "no_training_data",
                              plot_banner=banner, empty_panel_label=empty_label,
                              metrics_rows=len(rows), metric_groups_present=groups_present,
                              latest_metric_steps={k: v.get("latest_step") for k, v in inventories.items()})
            atomic_json(self.output / "render_status.json", {k: self.state[k] for k in
                        ("last_render_at", "render_reason", "render_status", "metrics_rows", "latest_metric_steps", "metric_groups_present",
                         "plot_banner", "empty_panel_label")})
        except Exception as exc:
            self.state["render_status"] = "error"
            self.error("render", exc)
        self.save()

    def run(self, interval=5.0, render_interval=60.0, once=False):
        # Preserve first monitor baseline across restarts; instance baseline is
        # separate and never replaces last_events used to bridge downtime.
        try:
            self.state["instance_baseline_events"] = cgroup_sample()["events"]
        except Exception as exc:
            self.error("instance_baseline", exc)
        self.save()
        next_render = 0.0
        terminal_complete = False
        training_was_complete = bool(self.state.get("training_complete_observed"))
        try:
            while not self.stop_requested:
                started = time.monotonic()
                try:
                    sample = self.sample()
                    progress = sample.get("progress", {})
                    complete = (not self.test_mode and len(progress) == 3 and all(p["checkpoint_step"] >= 300 for p in progress.values()))
                    final_transition = complete and not training_was_complete
                    if time.monotonic() >= next_render or final_transition or sample.get("stage_changed") or once:
                        self.render("training_complete" if final_transition else "stage_changed" if sample.get("stage_changed") else "once" if once else "periodic")
                        next_render = time.monotonic() + render_interval
                    if final_transition:
                        self.state["training_complete_observed"] = timestamp()
                        self.save()
                        training_was_complete = True
                    if not once and not self.test_mode and sample.get("experiment_stage") == "complete":
                        self.render("experiment_complete")
                        terminal_complete = True
                        break
                    if once:
                        break
                except Exception as exc:
                    self.error("sample_loop", exc)
                    self.save()
                    if once:
                        break
                while not self.stop_requested and time.monotonic() - started < interval:
                    time.sleep(min(0.5, max(0, interval - (time.monotonic() - started))))
        finally:
            if not once and not terminal_complete:
                self.render("monitor_exit")
            self.state.update(status="once_complete" if once else "complete" if terminal_complete else "stopped", stopped_at=timestamp())
            self.save()


def args_parse():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", nargs="?", choices=("run", "launch", "status"), default="run")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--interval", type=float, default=5.0)
    parser.add_argument("--render-interval", type=float, default=60.0)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--run-dir", type=Path, action="append", help="Explicit self-test run; requires --once and a separate output directory")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not 5 <= args.interval <= 10:
        parser.error("Sampling interval must be between 5 and 10 seconds")
    if args.render_interval < 10:
        parser.error("Render interval must be at least 10 seconds")
    if args.run_dir and (not args.once or args.output == DEFAULT_OUTPUT.resolve()):
        parser.error("Explicit test runs require --once and --output separate from formal monitoring")
    if args.once and args.action == "launch":
        parser.error("Use foreground --once for an isolated self-test")
    return args


def lock_path(output):
    key = hashlib.sha256(str(Path(output).resolve()).encode()).hexdigest()[:24]
    return Path(tempfile.gettempdir()) / ("sharedexpert-monitor-" + key + ".lock")


def status(output):
    state = read_json(Path(output) / "state.json", {})
    result = dict(state)
    result.update(running=process_alive(state.get("pid"), state.get("start_ticks")),
                  heartbeat_age_seconds=None if not state.get("heartbeat_epoch") else max(0, time.time() - state["heartbeat_epoch"]))
    # Full cursor context is persisted but is not useful in a status response.
    result.pop("log_cursors", None)
    return result


def main():
    args = args_parse()
    if args.action == "status":
        print(json.dumps(status(args.output), ensure_ascii=False, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(args.output / ".matplotlib")
    if args.action == "launch":
        current = status(args.output)
        if current.get("running"):
            print(json.dumps({"already_running": True, "pid": current["pid"], "output": str(args.output)}))
            return
        # Child takes the authoritative lock. This preliminary check avoids
        # duplicate launch attempts before its first heartbeat is persisted.
        with lock_path(args.output).open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(json.dumps({"already_running": True, "lock_held": True, "output": str(args.output)}))
                return
        command = [sys.executable, str(Path(__file__).resolve()), "run", "--output", str(args.output),
                   "--interval", str(args.interval), "--render-interval", str(args.render_interval)]
        with (args.output / "monitor.log").open("ab") as log:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, close_fds=True, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        print(json.dumps({"pid": child.pid, "output": str(args.output), "state": str(args.output / "state.json"), "python_executable": sys.executable}))
        return
    with lock_path(args.output).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({"already_running": True, "output": str(args.output)}))
            return
        runs = {p.name: p.resolve() for p in args.run_dir} if args.run_dir else {role: HERE / "runs" / role for role in ROLES}
        monitor = Monitor(args.output, runs, include_driver=not bool(args.run_dir), test_mode=bool(args.run_dir))
        def stop(signum, frame):
            monitor.stop_requested = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        monitor.run(args.interval, args.render_interval, args.once)
        print(json.dumps(status(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
