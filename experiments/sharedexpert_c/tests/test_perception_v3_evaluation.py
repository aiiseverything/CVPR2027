"""CPU-only checks for final-export gating and the requested evaluation order."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT / "runtime"))
SPEC = importlib.util.spec_from_file_location(
    "perception_v3_evaluation", EXPERIMENT / "runtime/evaluate_after_perception_v3.py")
QUEUE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(QUEUE)


class ExportGating(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.run = Path(temporary.name)
        self.model = self.run / "hf/global_step_100"
        self.model.mkdir(parents=True)

    def complete(self):
        self.receipt = {"status": "complete", "step": 100,
                        "config_sha256": "expected-config", "hf_model": str(self.model)}
        self.write_receipt()
        (self.model / "sharedexpert_export_complete.json").write_text(json.dumps({
            "step": 100, "source": str(self.run / "checkpoints/global_step_100/actor")}))
        for name in ("config.json", "tokenizer_config.json", "processor_config.json"):
            (self.model / name).write_text("{}")
        (self.model / "model.safetensors").write_bytes(b"fixture weights")

    def write_receipt(self):
        (self.run / "fast_start_complete.json").write_text(json.dumps(self.receipt))

    def test_partial_directory_without_completion_receipt_is_not_ready(self):
        (self.model / "model.safetensors").write_bytes(b"unfinished weights")
        self.assertFalse(QUEUE.export_ready(self.run, "expected-config"))

    def test_complete_formal_export_is_ready(self):
        self.complete()
        self.assertTrue(QUEUE.export_ready(self.run, "expected-config"))

    def test_wrong_training_receipt_is_rejected(self):
        self.complete()
        for key, wrong in (("config_sha256", "another-config"), ("step", 50),
                           ("status", "failed"), ("hf_model", str(self.run / "other-model"))):
            with self.subTest(key=key):
                original = self.receipt[key]
                self.receipt[key] = wrong
                self.write_receipt()
                with self.assertRaises(ValueError):
                    QUEUE.export_ready(self.run, "expected-config")
                self.receipt[key] = original

    def test_wrong_export_source_and_step_are_rejected(self):
        self.complete()
        marker = self.model / "sharedexpert_export_complete.json"
        for step, source in ((50, self.run / "checkpoints/global_step_100/actor"),
                             (100, self.run / "checkpoints/global_step_50/actor")):
            with self.subTest(step=step, source=source):
                marker.write_text(json.dumps({"step": step, "source": str(source)}))
                with self.assertRaises(ValueError):
                    QUEUE.export_ready(self.run, "expected-config")

    def test_missing_or_empty_shard_is_rejected(self):
        self.complete()
        index = self.model / "model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"layer": "missing.safetensors"}}))
        with self.assertRaisesRegex(ValueError, "index references missing"):
            QUEUE.export_ready(self.run, "expected-config")
        index.unlink()
        (self.model / "model.safetensors").write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "nonempty weights"):
            QUEUE.export_ready(self.run, "expected-config")

    def test_missing_processor_is_rejected(self):
        self.complete()
        (self.model / "processor_config.json").unlink()
        with self.assertRaisesRegex(ValueError, "processor metadata"):
            QUEUE.export_ready(self.run, "expected-config")


class OwnershipAndSequence(unittest.TestCase):
    def test_process_identity_handles_parentheses_and_dead_processes(self):
        for state, expected in (("S", 98765), ("Z", None)):
            fields = [state] + ["0"] * 18 + ["98765"]
            with self.subTest(state=state), patch.object(Path, "read_text", return_value=
                    "123 (python worker (ray)) " + " ".join(fields)):
                self.assertEqual(QUEUE.process_identity(123), expected)
        with patch.object(Path, "read_text", side_effect=FileNotFoundError):
            self.assertIsNone(QUEUE.process_identity(123))

    def test_requested_sequence_targets_only_final_v3_model(self):
        jobs = QUEUE.commands("/eval/bin/python", 0)
        self.assertEqual([job["suite"] for job in jobs], ["highres", "ocr", "main"])
        self.assertEqual([job["n_expected"] for job in jobs], [1722, 997, 3423])
        for job in jobs:
            command = job["command"]
            self.assertEqual(command[command.index("--model") + 1], str(QUEUE.MODEL))
            self.assertEqual(command[command.index("--name") + 1], "perception-v3-step100")
            self.assertNotIn("--smoke", command)

    def test_gpu_with_compute_pid_or_memory_is_not_free(self):
        for applications, memory, expected in (("", "0", True),
                ("GPU-selected, 123", "0", False), ("", "300", False),
                ("GPU-other, 123", "0", True)):
            with self.subTest(applications=applications, memory=memory), patch.object(
                    QUEUE.subprocess, "check_output", side_effect=["GPU-selected\n", applications, memory]):
                self.assertEqual(QUEUE.gpu_free(0), expected)

    def test_export_does_not_start_evaluation_while_training_launcher_is_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fast_start.json").write_text('{"config_sha256":"expected"}')
            (root / "perception_v3_resume_20261030.json").write_text(json.dumps({
                "process": {"pid": 123, "start_ticks": 456}}))
            overrides = {"RUN": root, "RUNTIME": root, "STATE": root / "state.json"}
            with patch.multiple(QUEUE, **overrides), \
                    patch.object(sys, "argv", ["queue", "--python", sys.executable]), \
                    patch.object(QUEUE, "file_sha", return_value="expected"), \
                    patch.object(QUEUE, "load_suite", return_value=([], {})), \
                    patch.object(QUEUE, "process_identity", return_value=456), \
                    patch.object(QUEUE, "export_ready", return_value=True), \
                    patch.object(QUEUE, "training_step", return_value=100), \
                    patch.object(QUEUE.time, "sleep", side_effect=KeyboardInterrupt), \
                    patch.object(QUEUE.signal, "signal"), \
                    patch.object(QUEUE.subprocess, "Popen") as launch:
                with self.assertRaises(KeyboardInterrupt):
                    QUEUE.main()
                launch.assert_not_called()
            state = json.loads((root / "state.json").read_text())
            self.assertTrue(state["export_ready"])
            self.assertTrue(state["training_launcher_alive"])
            self.assertEqual([job["stage"] for job in state["jobs"]], ["pending"] * 3)

    def test_completion_written_during_launcher_exit_is_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fast_start.json").write_text('{"config_sha256":"expected"}')
            (root / "perception_v3_resume_20261030.json").write_text(json.dumps({
                "process": {"pid": 123, "start_ticks": 456}}))
            overrides = {"RUN": root, "RUNTIME": root, "STATE": root / "state.json"}
            with patch.multiple(QUEUE, **overrides), \
                    patch.object(sys, "argv", ["queue", "--python", sys.executable]), \
                    patch.object(QUEUE, "file_sha", return_value="expected"), \
                    patch.object(QUEUE, "load_suite", return_value=([], {})), \
                    patch.object(QUEUE, "process_identity", side_effect=[456, None]), \
                    patch.object(QUEUE, "export_ready", side_effect=[False, True]) as ready, \
                    patch.object(QUEUE, "training_step", return_value=100), \
                    patch.object(QUEUE.signal, "signal"), \
                    patch.object(QUEUE.subprocess, "Popen", side_effect=RuntimeError("registration reached")) as launch:
                with self.assertRaisesRegex(RuntimeError, "registration reached"):
                    QUEUE.main()
                self.assertEqual(ready.call_count, 2)
                launch.assert_called_once()
                self.assertIn("register_model.py", launch.call_args.args[0][1])


if __name__ == "__main__":
    unittest.main()
