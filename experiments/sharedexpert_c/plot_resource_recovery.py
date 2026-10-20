#!/usr/bin/env python3
"""Plot recorded resource windows; rerun with /opt/venv/main/bin/python3."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
from zoneinfo import ZoneInfo

for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["MPLBACKEND"] = "Agg"

TZ = ZoneInfo("Asia/Shanghai")
GIB = 2 ** 30
HERE = Path(__file__).resolve().parent


def read_windows(source, bounds):
    """Read a fixed byte snapshot line by line, retaining only plotting fields."""
    windows, partial = [[] for _ in bounds], 0
    snapshot_bytes = source.stat().st_size
    with source.open("rb") as stream:
        while stream.tell() < snapshot_bytes:
            line = stream.readline(snapshot_bytes - stream.tell())
            if not line.endswith(b"\n"):
                partial += 1
                break
            row = json.loads(line)  # A malformed committed row is an error.
            if row.get("test_mode"):
                continue
            epoch = row.get("epoch")
            if epoch is None:
                epoch = datetime.fromisoformat(row["time"]).timestamp()
            index = next((i for i, (start, end) in enumerate(bounds) if start <= epoch <= end), None)
            if index is None:
                continue
            group = row.get("cgroup") or {}
            events = group.get("events") or {}
            windows[index].append({"epoch": epoch, "current": group.get("current_bytes"),
                "working": group.get("working_set_bytes"), "limit": group.get("limit_bytes"),
                "events": {key: events.get(key) for key in ("oom", "oom_kill")},
                "gpus": {int(g["index"]): g.get("memory_used_mib") for g in row.get("gpus", [])
                         if int(g["index"]) in (3, 4, 5, 6)}})
    for rows in windows:
        rows.sort(key=lambda row: row["epoch"])
    return windows, snapshot_bytes, partial


def points(rows, value, gap_seconds):
    """Break long gaps and missing values; no resampling, filling, or smoothing."""
    times, values, previous = [], [], None
    for row in rows:
        epoch = row["epoch"]
        if previous is not None and epoch - previous > gap_seconds:
            times.append(datetime.fromtimestamp(previous + .001, TZ))
            values.append(math.nan)
        item = value(row)
        times.append(datetime.fromtimestamp(epoch, TZ))
        values.append(math.nan if item is None else item)
        previous = epoch
    return times, values


def event_delta(rows, key):
    values = [r["events"][key] for r in rows if r["events"][key] is not None]
    if len(values) < 2:
        return "unavailable"
    if any(b < a for a, b in zip(values, values[1:])):
        return "counter reset; delta unknown"
    return str(values[-1] - values[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=HERE / "monitoring/samples.jsonl")
    parser.add_argument("--output", type=Path, default=HERE / "monitoring/resource_recovery.png")
    parser.add_argument("--date", default="2026-10-20")
    parser.add_argument("--restart", default="11:02")
    parser.add_argument("--failure-end", default="11:29")
    parser.add_argument("--batch30-start", default="11:56:08")
    parser.add_argument("--batch30-end", default="12:36")
    parser.add_argument("--batch30-recovery", type=Path, default=HERE / "runtime/recovery_20261020_1234.json")
    parser.add_argument("--repair-launch", type=Path, default=HERE / "runtime/unused_topk_fix_launch.json")
    parser.add_argument("--state", type=Path, default=HERE / "runtime/experiment_state.json")
    parser.add_argument("--gap-seconds", type=float, default=15)
    args = parser.parse_args()
    local = lambda clock: datetime.fromisoformat(args.date + "T" + clock).replace(tzinfo=TZ)
    launch = json.loads(args.repair_launch.read_text())
    repaired_at = datetime.fromisoformat(launch["launch_window_start"]).astimezone(TZ)
    state = json.loads(args.state.read_text())
    matching_state = state.get("driver_pid") == launch["driver_pid"]
    bounds = [(local("00:45").timestamp(), local("01:03").timestamp()),
              (local(args.restart).timestamp(), local(args.failure_end).timestamp()),
              (local(args.batch30_start).timestamp(), local(args.batch30_end).timestamp()),
              (repaired_at.timestamp(), math.inf)]
    if bounds[3][0] <= bounds[2][1]:
        raise SystemExit("The repaired attempt must start after the fixed historical windows")
    recovery = json.loads(args.batch30_recovery.read_text())
    failed_at = datetime.fromisoformat(recovery["previous_state"]["updated_at"]).astimezone(TZ)
    if not bounds[2][0] <= failed_at.timestamp() <= bounds[2][1]:
        raise SystemExit("Recorded batch30 failure must fall inside the fixed historical window")
    windows, size, partial = read_windows(args.samples, bounds)
    if not all(windows):
        raise SystemExit("All four requested windows need recorded samples; no synthetic data is drawn")
    limits = {r["limit"] for rows in windows for r in rows if r["limit"] is not None}
    if limits != {64 * GIB}:
        raise SystemExit(f"Expected the actual 64 GiB container limit; recorded limits={limits}")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 4, figsize=(24.6, 9.4), sharey="row")
    fig.subplots_adjust(left=.045, right=.985, bottom=.235, top=.80, hspace=.43, wspace=.13)
    cutoff = datetime.fromtimestamp(windows[3][-1]["epoch"], TZ)
    fig.suptitle("SharedExpert | Recorded resource recovery", fontsize=19, weight="bold", x=.055, ha="left", y=.98)
    fig.text(.055, .925, f"Asia/Shanghai (UTC+08:00)  |  Latest included sample: {cutoff:%Y-%m-%d %H:%M:%S}", color="#425466", fontsize=11)
    titles = ["Batch 40: initial failure\n00:45-01:03",
              f"Batch 40: port 8112 service released\n{args.restart}-{args.failure_end}",
              f"Batch 30: second update CUDA OOM\n{args.batch30_start}-{args.batch30_end} | ports 8110 + 8112 off",
              f"Batch 30: unused top-k disabled\n{repaired_at:%H:%M:%S}-{cutoff:%H:%M:%S} | formal attempt"]
    summary = []
    for column, rows in enumerate(windows):
        top, bottom = axes[:, column]
        top.set_title(titles[column], loc="left", fontsize=12, weight="bold", pad=13)
        for field, color, label in [("current", "#0072B2", "Cgroup current"), ("working", "#D55E00", "Cgroup working set")]:
            top.plot(*points(rows, lambda r, key=field: None if r[key] is None else r[key] / GIB,
                             args.gap_seconds), color=color, label=label, linewidth=1.7, marker=".", markersize=2.5)
        top.axhspan(64 * .94, 64, color="#D55E00", alpha=.07)
        top.axhline(64, color="#263238", linestyle="--", linewidth=1.1, label="64 GiB limit")
        top.axhline(64 * .94, color="#B2182B", linestyle=":", linewidth=1.4, label="94% guard: 60.16 GiB")
        top.set_ylim(0, 68)
        valid = [r for r in rows if r["working"] is not None]
        peak = max(valid, key=lambda r: r["working"]) if valid else None
        if peak:
            text = f"Observed working-set peak: {peak['working']/GIB:.3f} GiB"
            top.text(.025, .06, text, transform=top.transAxes, fontsize=9,
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .88})
        for gpu, color, style in zip((3, 4, 5, 6), ("#0072B2", "#E69F00", "#009E73", "#CC79A7"), ("-", "--", "-.", ":")):
            bottom.plot(*points(rows, lambda r, g=gpu: None if r["gpus"].get(g) is None else r["gpus"][g] / 1024,
                                args.gap_seconds), color=color, linestyle=style, linewidth=1.7, label=f"GPU {gpu}")
        bottom.set_ylim(0, 46)
        bottom.set_title(f"Host cgroup deltas: OOM {event_delta(rows, 'oom')}  |  kills {event_delta(rows, 'oom_kill')}",
                         loc="left", fontsize=10, pad=10)
        if column == 2:
            for ax in (top, bottom):
                ax.axvline(failed_at, color="#B2182B", linestyle="--", linewidth=1.1)
            top.text(.025, .87, f"Driver failed {failed_at:%H:%M:%S}\nCUDA allocation failed during update 2",
                     transform=top.transAxes, fontsize=9, color="#9B1C31",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
        elif column == 3:
            if matching_state:
                saved_at = datetime.fromisoformat(state["updated_at"]).astimezone(TZ)
                status = state["stage"].replace("_", " ")
                if state.get("teacher"):
                    status += " / " + state["teacher"]
                status_text = f"Saved status: {status}\nRecorded {saved_at:%H:%M:%S}; driver {launch['driver_pid']}"
            else:
                status_text = "Current saved state belongs to a different driver"
            top.text(.025, .87, status_text, transform=top.transAxes, fontsize=9, color="#1B6A50",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
        bottom.set_xlabel("Local time (Asia/Shanghai)")
        end = bounds[column][1] if math.isfinite(bounds[column][1]) else rows[-1]["epoch"]
        for ax in (top, bottom):
            ax.set_xlim(datetime.fromtimestamp(bounds[column][0], TZ), datetime.fromtimestamp(end, TZ))
            span = end - bounds[column][0]
            locator = (mdates.SecondLocator(interval=30, tz=TZ) if span < 120 else
                       mdates.MinuteLocator(interval=1, tz=TZ) if span < 300 else
                       mdates.AutoDateLocator(minticks=3, maxticks=6, tz=TZ))
            ax.xaxis.set_major_locator(locator)
            crossed_date = datetime.fromtimestamp(end, TZ).date() != datetime.fromtimestamp(bounds[column][0], TZ).date()
            clock_format = "%H:%M:%S" if span < 300 else "%H:%M"
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M" if crossed_date else clock_format, tz=TZ))
            ax.grid(axis="y", alpha=.18)
        gaps = sum(b["epoch"] - a["epoch"] > args.gap_seconds for a, b in zip(rows, rows[1:]))
        summary.append({"window": titles[column], "samples": len(rows), "gaps_over_threshold": gaps,
                        "observed_peak_working_gib": None if peak is None else peak["working"] / GIB,
                        "oom_delta": event_delta(rows, "oom"), "oom_kill_delta": event_delta(rows, "oom_kill")})
    axes[0, 0].set_ylabel("Container memory (GiB)")
    axes[1, 0].set_ylabel("GPU memory used (GiB)")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0, 1.01), fontsize=8, framealpha=.92)
    axes[1, 0].legend(loc="upper left", ncol=4, fontsize=9, framealpha=.92)
    fig.text(.055, .153, f"Source: monitoring/samples.jsonl  |  Actual samples only; gaps > {args.gap_seconds:g}s and missing values stay blank.",
             fontsize=9, color="#425466")
    fig.text(.055, .125, "The intermediate batch-40 attempt (11:32-11:47) was interrupted at the user's request and is omitted here; all original samples remain available.",
             fontsize=9, color="#425466")
    fig.text(.055, .097, "Service occupancy differs between windows; this is not a controlled batch-size-only ablation. Probe intervals between attempts are excluded.",
             fontsize=9, color="#425466")
    fig.text(.055, .069, "CUDA OOM is separate from host cgroup OOM counters: zero cgroup OOM increments does not mean zero CUDA OOM.",
             fontsize=9, color="#9B1C31", weight="bold")
    fig.text(.055, .041, "Shared-container memory includes other services. Peaks are sampled peaks; guard events are not counted as OOMs.",
             fontsize=9, color="#425466")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.stem + ".tmp.png")
    fig.savefig(temporary, dpi=160, facecolor="white")
    plt.close(fig)
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output), "latest_included_sample": cutoff.isoformat(),
                      "batch30_driver_failed_at": failed_at.isoformat(), "batch30_recovery": str(args.batch30_recovery),
                      "repaired_attempt_start": repaired_at.isoformat(), "repair_launch": str(args.repair_launch),
                      "repaired_attempt_saved_state": {key: state.get(key) for key in ("stage", "teacher", "updated_at", "driver_pid")} if matching_state else None,
                      "snapshot_bytes": size, "partial_tail_ignored": partial, "windows": summary}, indent=2))


if __name__ == "__main__":
    main()
