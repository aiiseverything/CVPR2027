"""Handoff safety regressions; fake upstream artifacts and GPU responses only."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

SPEC = importlib.util.spec_from_file_location("perception_v2_queue", Path(__file__).resolve().parents[1] / "runtime/queue_perception_v2.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.upstream_process = {"pid": 10, "start_ticks": 100, "argv": ["bash", "autopilot.sh"]}
        self.write("runtime/experiment_state.json", {"teacher": "generalist", "stage": "complete", "error": None, "driver_pid": 11, "child_pid": 12})
        self.write("runs/generalist/protocol.json", {"protocol": "frozen"})
        self.write("results_generalist_step100/protocol.json", {"protocol": "frozen"})
        self.write("runtime/autopilot.log", "old AUTOPILOT_COMPLETE elapsed_min=1\n")
        log = self.root / "runtime/autopilot.log"
        stat = log.stat()
        self.upstream = {"process": self.upstream_process, "log_offset": stat.st_size,
                         "log_device": stat.st_dev, "log_inode": stat.st_ino,
                         "generalist_protocol_sha256": queue.sha256(self.root / "runs/generalist/protocol.json")}
        self.write("runs/generalist/checkpoints/latest_checkpointed_iteration.txt", "100\n")
        self.write("runs/generalist/hf/global_step_100/sharedexpert_export_complete.json",
                   {"step": 100, "source": str(self.root / "runs/generalist/checkpoints/global_step_100/actor")})
        self.write("runs/generalist/hf/global_step_100/config.json", {})
        self.write("runs/generalist/hf/global_step_100/model.safetensors", "fake-weight-bytes")

    def tearDown(self):
        self.temporary.cleanup()

    def write(self, path, value):
        path = self.root / path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) if isinstance(value, dict) else value)

    def complete_log(self):
        with (self.root / "runtime/autopilot.log").open("a") as stream:
            stream.write("[current run] AUTOPILOT_COMPLETE elapsed_min=30\n")

    def gate(self, observe=lambda pid: None):
        return queue.upstream_gate(self.upstream, self.root, observe)

    def test_live_controller_wins_over_old_success_receipts(self):
        ready, _ = self.gate(lambda pid: self.upstream_process if pid == 10 else None)
        self.assertFalse(ready)

    def test_old_success_line_is_insufficient(self):
        with self.assertRaisesRegex(RuntimeError, "without its new AUTOPILOT_COMPLETE"):
            self.gate()

    def test_failed_generalist_never_launches_even_with_export(self):
        self.complete_log()
        self.write("runtime/experiment_state.json", {"teacher": "generalist", "stage": "failed", "error": "OOM"})
        with self.assertRaisesRegex(RuntimeError, "without successful driver completion"):
            self.gate()

    def test_missing_export_never_launches(self):
        self.complete_log()
        (self.root / "runs/generalist/hf/global_step_100/sharedexpert_export_complete.json").unlink()
        with self.assertRaises(FileNotFoundError):
            self.gate()

    def test_full_export_and_archive_are_required(self):
        self.complete_log()
        (self.root / "runs/generalist/hf/global_step_100/model.safetensors").unlink()
        with self.assertRaisesRegex(RuntimeError, "weights are missing"):
            self.gate()

    def test_archival_finishes_before_handoff(self):
        self.complete_log()
        self.write("results_generalist_step100/protocol.json", {"protocol": "different"})
        with self.assertRaisesRegex(RuntimeError, "archive does not match"):
            self.gate()

    def test_success_only_after_controller_and_launcher_exit(self):
        self.complete_log()
        ready, _ = self.gate(lambda pid: {"pid": 12, "argv": ["python", "launch_local.py", "generalist"]} if pid == 12 else None)
        self.assertFalse(ready)
        ready, _ = self.gate()
        self.assertTrue(ready)

    def test_pid_reuse_is_not_the_captured_controller(self):
        self.complete_log()
        replacement = dict(self.upstream_process, start_ticks=101)
        ready, _ = self.gate(lambda pid: replacement if pid == 10 else None)
        self.assertTrue(ready)

    def test_log_rotation_fails_closed(self):
        old = self.root / "runtime/autopilot.log"
        old.rename(old.with_suffix(".old"))
        self.write("runtime/autopilot.log", "AUTOPILOT_COMPLETE elapsed_min=1\n" * 5)
        with self.assertRaisesRegex(RuntimeError, "replaced/truncated"):
            self.gate()

    def test_gpu_compute_pid_blocks_even_with_low_memory(self):
        replies = iter(["0, GPU-a, 20\n1, GPU-b, 10\n", "GPU-a, 999\n"])
        ready, _ = queue.gpu_idle(["0", "1"], lambda *args, **kwargs: next(replies))
        self.assertFalse(ready)

    def test_gpu_high_memory_blocks(self):
        ready, _ = queue.gpu_idle(["0", "1"], lambda *args, **kwargs: "0, GPU-a, 2048\n1, GPU-b, 10\n")
        self.assertFalse(ready)

    def test_gpu_idle_requires_both_checks(self):
        replies = iter(["0, GPU-a, 20\n1, GPU-b, 10\n", ""])
        ready, _ = queue.gpu_idle(["0", "1"], lambda *args, **kwargs: next(replies))
        self.assertTrue(ready)

    def test_duplicate_queue_lock_is_rejected(self):
        path = self.root / "queue.lock"
        with queue.queue_lock(path):
            with self.assertRaisesRegex(RuntimeError, "Another perception_v2 queue"):
                with queue.queue_lock(path):
                    self.fail("duplicate queue acquired lock")
        with queue.queue_lock(path):
            pass

    def test_memory_guard_counts_working_set_not_reclaimable_cache(self):
        self.write("memory.max", "1000")
        self.write("memory.current", "990")
        self.write("memory.stat", "inactive_file 100\n")
        queue.check_memory_pressure(self.root)
        self.write("memory.stat", "inactive_file 10\n")
        with self.assertRaisesRegex(MemoryError, "stopping only this queue's child process group"):
            queue.check_memory_pressure(self.root)

    def test_storage_failure_stops_only_queue_child_and_finishes_copy(self):
        child = Mock(pid=99999, returncode=None)
        child.poll.return_value = None
        storage = Mock()
        storage.tick.side_effect = RuntimeError("storage copy verification failed")
        storage.finalize.return_value = {"pending": False, "safe": True}
        state = {"frozen_inputs": {}}
        with patch.object(queue, "STATE", self.root / "queue.json"), \
             patch.object(queue, "LOG", self.root / "queue.log"), \
             patch.object(queue, "verify_inputs"), patch.object(queue, "check_memory_pressure"), \
             patch.object(queue, "log"), patch.object(queue, "process_identity", return_value={"start_ticks": 1}), \
             patch.object(queue.subprocess, "Popen", return_value=child), \
             patch.object(queue, "stop_child") as stop:
            with self.assertRaisesRegex(RuntimeError, "storage copy verification failed"):
                queue.run_child(state, "training", ["fake-trainer"], 100, storage)
        stop.assert_called_once_with(child)
        storage.finalize.assert_called_once()
        self.assertEqual(state["storage_report"], {"pending": False, "safe": True})

    def test_formal_exit_drains_pending_storage_without_starting_new_copy(self):
        storage = Mock()
        storage.finalize.side_effect = [{"pending": True}, {"pending": False}]
        state = {}
        with patch.object(queue, "STATE", self.root / "queue.json"), patch.object(queue.time, "sleep"):
            queue.finish_storage(state, storage)
        self.assertEqual(storage.finalize.call_count, 2)
        storage.tick.assert_not_called()
        self.assertFalse(state["storage_report"]["pending"])


if __name__ == "__main__":
    unittest.main()
