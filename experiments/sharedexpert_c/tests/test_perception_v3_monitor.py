"""CPU checks for direct training recovery and monitor completion evidence."""
import importlib.util
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "perception_v3_monitor", ROOT / "runtime/monitor_perception_v3.py")
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)
DASHBOARD_SPEC = importlib.util.spec_from_file_location(
    "perception_v3_dashboard", ROOT / "runtime/live_training_dashboard.py")
DASHBOARD = importlib.util.module_from_spec(DASHBOARD_SPEC)
DASHBOARD_SPEC.loader.exec_module(DASHBOARD)


class QueueHealth(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.proc = self.root / "proc"
        self.run = self.root / "run"
        self.run.mkdir()

    def process(self, script, pid=123, ticks=456, status="S"):
        directory = self.proc / str(pid)
        directory.mkdir(parents=True, exist_ok=True)
        fields = [status] + ["0"] * 18 + [str(ticks)]
        (directory / "stat").write_text(f"{pid} (python) " + " ".join(fields))
        argv = ["/venv/bin/python", "-u", f"/runtime/{script}", "--config", "/run/config.yaml"]
        (directory / "cmdline").write_bytes(b"\0".join(arg.encode() for arg in argv) + b"\0")
        return {"pid": pid, "start_ticks": ticks, "argv": argv}

    def health(self, state):
        return MONITOR.queue_health(state, self.proc, self.run)

    def test_live_fast_start_survives_stale_pipeline_identity(self):
        identity = self.process("fast_start_perception_v3.py")
        state = {"stage": "training", "pid": 999, "start_ticks": 1,
                 "error": "previous interruption", "active_training": identity}
        self.assertEqual(self.health(state), ("training", None))

    def test_recycled_pid_is_not_accepted(self):
        identity = self.process("fast_start_perception_v3.py")
        identity["start_ticks"] -= 1
        self.assertEqual(self.health({"stage": "training", "active_training": identity})[0], "failed")

    def test_dead_or_zombie_training_is_not_accepted(self):
        identity = self.process("fast_start_perception_v3.py", status="Z")
        state = {"stage": "training", "active_training": identity}
        self.assertEqual(self.health(state)[0], "failed")
        (self.proc / str(identity["pid"]) / "stat").unlink()
        self.assertEqual(self.health(state)[0], "failed")

    def test_changed_config_argv_is_not_accepted(self):
        identity = self.process("fast_start_perception_v3.py")
        identity["argv"][-1] = "/different/config.yaml"
        self.assertEqual(self.health({"stage": "training", "active_training": identity})[0], "failed")

    def test_unrelated_script_is_not_accepted(self):
        identity = self.process("another_training.py")
        self.assertEqual(self.health({"stage": "training", "active_training": identity})[0], "failed")

    def test_original_pipeline_still_works(self):
        identity = self.process("run_perception_v3_pipeline.py")
        for stage in ("screening", "training"):
            self.assertEqual(self.health({"stage": stage, **identity}), (stage, None))

    def complete_export(self):
        export = self.run / "hf/global_step_100"
        export.mkdir(parents=True)
        (self.run / "fast_start.json").write_text(json.dumps({
            "config_sha256": "config", "started_at": "2026-10-30T05:00:00Z"}))
        (self.run / "fast_start_complete.json").write_text(json.dumps({
            "status": "complete", "step": 100, "config_sha256": "config",
            "finished_at": "2026-10-30T06:00:00Z", "hf_model": str(export)}))
        (export / "sharedexpert_export_complete.json").write_text('{"step":100}')
        (export / "config.json").write_text('{}')
        (export / "model.safetensors").write_bytes(b"fixture weights")
        return export

    def test_actual_export_and_current_receipt_complete_after_launcher_exit(self):
        self.complete_export()
        self.assertEqual(self.health({"stage": "training", "active_training": {"pid": 999}}),
                         ("complete", None))

    def test_receipt_without_weights_is_not_completion(self):
        export = self.complete_export()
        (export / "model.safetensors").unlink()
        self.assertEqual(self.health({"stage": "training", "active_training": {"pid": 999}})[0], "failed")

    def test_receipt_from_previous_launch_is_not_completion(self):
        self.complete_export()
        (self.run / "fast_start.json").write_text(json.dumps({
            "config_sha256": "config", "started_at": "2026-10-30T07:00:00Z"}))
        self.assertEqual(self.health({"stage": "training", "active_training": {"pid": 999}})[0], "failed")


class ResumedDashboard(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "checkpoints").mkdir()
        (self.root / "checkpoints/latest_checkpointed_iteration.txt").write_text("50")
        (self.root / "fast_start.json").write_text(json.dumps({
            "started_at": "2026-10-30T07:00:00Z", "resume_from_checkpoint": 50}))
        self.log = self.root / "training.log"
        for name, value in (("RUN", self.root), ("LOG", self.log), ("MON", self.root)):
            mock = patch.object(DASHBOARD, name, value)
            mock.start()
            self.addCleanup(mock.stop)

    def test_empty_resumed_log_reports_saved_step(self):
        self.log.write_text("Loading checkpoint\n")
        self.assertEqual(DASHBOARD.progress_step(), 50)

    def test_new_log_progress_overrides_saved_step(self):
        self.log.write_text("Training Progress:  51%|xxxxx| 51/100\n")
        self.assertEqual(DASHBOARD.progress_step(), 51)

    def test_old_log_does_not_report_discarded_step(self):
        import os
        self.log.write_text("Training Progress:  68%|xxxxx| 68/100\n")
        os.utime(self.log, (1, 1))
        self.assertEqual(DASHBOARD.progress_step(), 50)

    def test_scalar_values_exclude_previous_session(self):
        epoch = DASHBOARD.session_started_epoch()
        (self.root / "metrics.csv").write_text(
            "step,tag,value,wall_time\n"
            f"68,critic/rewards/mean,0.99,{epoch - 60}\n"
            f"51,critic/rewards/mean,0.62,{epoch + 60}\n")
        self.assertEqual(DASHBOARD.latest_metrics(100), {"critic/rewards/mean": (51, 0.62)})

    def test_no_current_scalars_means_no_values(self):
        epoch = DASHBOARD.session_started_epoch()
        (self.root / "metrics.csv").write_text(
            f"step,tag,value,wall_time\n50,critic/rewards/mean,0.99,{epoch - 60}\n")
        self.assertEqual(DASHBOARD.latest_metrics(50), {})

    def test_watchdog_follows_current_container_limit(self):
        self.assertEqual(DASHBOARD.host_watchdog(92 * DASHBOARD.GIB), 84 * DASHBOARD.GIB)
        self.assertEqual(DASHBOARD.host_watchdog(100 * DASHBOARD.GIB), 92 * DASHBOARD.GIB)

    def test_live_page_embeds_versioned_training_chart(self):
        chart = self.root / "training_metrics.png"
        chart.write_bytes(b"fixture chart")
        row = {
            "gpu_used": 10 * DASHBOARD.GIB, "gpu_total": 84 * DASHBOARD.GIB,
            "anon": 20 * DASHBOARD.GIB, "shmem": 5 * DASHBOARD.GIB,
            "file": 10 * DASHBOARD.GIB, "current": 35 * DASHBOARD.GIB,
            "limit": 92 * DASHBOARD.GIB, "status": "RUNNING", "alive": True,
            "step": 52,
        }
        html = self.root / "memory_live.html"
        with patch.object(DASHBOARD, "HTML", html):
            DASHBOARD.write_html([row])
        text = html.read_text()
        self.assertIn('http-equiv="refresh" content="15"', text)
        self.assertRegex(text, r'<img\b[^>]*src="training_metrics\.png\?v=' +
                         str(chart.stat().st_mtime_ns) + r'"')


class ResumedMetricHistory(unittest.TestCase):
    """A rollback must not join fresh scalar values to discarded future steps."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run = self.root / "run"
        self.output = self.root / "monitor"
        self.output.mkdir()
        self.tb = self.run / "tensorboard"
        self.tb.mkdir(parents=True)
        self.write_events(".old", [(49, 0.49, 900), (50, 0.50, 910),
                                   (51, 0.91, 920), (68, 0.98, 930)])
        self.write_events(".resumed", [(51, 0.51, 1010), (52, 0.52, 1020)])
        self.monitor = MONITOR.shared.Monitor(
            self.output, {"perception_v3": self.run}, include_driver=False, test_mode=True)
        self.monitor.state.update(display_status="RUNNING", experiment_stage="training")

    def write_events(self, suffix, values):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        writer = EventFileWriter(str(self.tb), filename_suffix=suffix)
        try:
            for step, value, wall_time in values:
                writer.add_event(Event(wall_time=wall_time, step=step, summary=Summary(
                    value=[Summary.Value(tag="critic/rewards/mean", simple_value=value)])))
            writer.flush()
        finally:
            writer.close()

    def test_default_history_retains_all_steps_and_latest_duplicate(self):
        rows, inventory = self.monitor.metrics()
        values = {row["step"]: row["value"] for row in rows}
        self.assertEqual(set(values), {49, 50, 51, 52, 68})
        self.assertAlmostEqual(values[51], 0.51, places=6)
        self.assertEqual(inventory["perception_v3"]["latest_step"], 68)

    def test_resumed_history_csv_and_inventory_exclude_discarded_steps(self):
        rows, inventory = self.monitor.metrics(resume_step=50, resumed_at=1000)
        values = {row["step"]: row["value"] for row in rows}
        self.assertEqual(set(values), {49, 50, 51, 52})
        self.assertAlmostEqual(values[50], 0.50, places=6)
        self.assertAlmostEqual(values[51], 0.51, places=6)
        self.assertAlmostEqual(values[52], 0.52, places=6)
        self.assertEqual(inventory["perception_v3"]["latest_step"], 52)
        with (self.output / "metrics.csv").open() as stream:
            csv_rows = list(csv.DictReader(stream))
        self.assertEqual({int(row["step"]) for row in csv_rows}, set(values))
        self.assertEqual(json.loads((self.output / "scalar_inventory.json").read_text())[
            "perception_v3"]["latest_step"], 52)

    def test_no_new_session_events_ends_at_checkpoint(self):
        rows, inventory = self.monitor.metrics(resume_step=50, resumed_at=2000)
        self.assertEqual({row["step"] for row in rows}, {49, 50})
        self.assertEqual(inventory["perception_v3"]["latest_step"], 50)

    def test_resume_time_boundary_is_inclusive(self):
        rows, _ = self.monitor.metrics(resume_step=50, resumed_at=1010)
        self.assertEqual({row["step"] for row in rows}, {49, 50, 51, 52})

    def test_resumed_render_publishes_matching_step_metadata(self):
        self.monitor.render(reason="test_resume", resume_step=50, resumed_at=1000)
        status = json.loads((self.output / "render_status.json").read_text())
        self.assertEqual(status["latest_metric_steps"], {"perception_v3": 52})
        self.assertEqual(status["metrics_rows"], 4)
        self.assertEqual(status["render_status"], "rendered")
        self.assertGreater((self.output / "training_metrics.png").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
