"""CPU integration coverage for the serial post-training evaluation supervisor."""
import importlib.util
import json
from pathlib import Path
import signal
import sys
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "runtime/evaluate_after_perception_v3.py"
sys.path.insert(0, str(SCRIPT.parent))
SPEC = importlib.util.spec_from_file_location("v3_evaluation_sequence_under_test", SCRIPT)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)
from common import PROTOCOL


class SequenceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.runtime = self.root / "runtime"
        self.evaluation = self.root / "evaluation"
        self.inputs = self.runtime / "inputs"
        self.results = self.evaluation / "results"
        self.run = self.root / "runs/perception_v3"
        self.state = self.runtime / "state.json"
        for directory in (self.runtime, self.evaluation, self.inputs, self.run):
            directory.mkdir(parents=True, exist_ok=True)
        self.rows = {name: {"id": name + ":1", "benchmark": "vstar" if name == "highres" else name,
                            "task": "ocrbench" if name == "ocr" else "mcq", "images": []}
                     for name in ("highres", "ocr", "main")}
        self.write_suite(self.evaluation / "suite.jsonl", self.rows["main"])
        self.write_suite(self.inputs / "suite_ocr.jsonl", self.rows["ocr"])
        source = self.evaluation / "data/highres_perception_v1"
        self.write_suite(source / "suite.jsonl", self.rows["highres"])
        for name in ("selection.json", "source_manifest.json", "dataset_manifest.json", "questions.jsonl"):
            self.write_json(source / name, {})
        self.write_json(self.inputs / "server.json", {"enforce_eager": True})
        (self.run / "resolved_config.fast.yaml").write_text("frozen config\n")
        self.write_json(self.run / "fast_start.json", {
            "config_sha256": queue.file_sha(self.run / "resolved_config.fast.yaml")})
        self.write_json(self.runtime / "perception_v3_resume_20261030.json", {
            "process": {"pid": 99999999, "start_ticks": 1}})
        self.events = []
        self.environments = {}
        self.handlers = {}
        self.failure = None
        self.bad_prediction = None
        self.bad_summary = None
        self.interrupt = None
        self.children = []
        self.real_commands = queue.commands
        patches = [
            patch.multiple(queue, RUNTIME=self.runtime, EVAL=self.evaluation, INPUTS=self.inputs,
                           RESULTS=self.results, RUN=self.run, MODEL=self.run / "hf/global_step_100",
                           STATE=self.state, ROOT=self.root),
            patch.object(queue, "export_ready", return_value=True),
            patch.object(queue, "process_identity", return_value=None),
            patch.object(queue, "training_step", return_value=100),
            patch.object(queue, "gpu_free", return_value=True),
            patch.object(queue, "commands", side_effect=self.small_commands),
            patch.object(queue, "local_lock_path", return_value=self.root / "queue.lock"),
            patch.object(queue.subprocess, "Popen", side_effect=self.launch),
            patch.object(queue.signal, "signal", side_effect=self.set_handler),
            patch.object(queue.time, "sleep"),
            patch.object(sys, "argv", [str(SCRIPT), "--python", sys.executable, "--poll-seconds", "1"]),
        ]
        for mocked in patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    @staticmethod
    def write_json(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value) + "\n")

    def write_suite(self, path, row):
        self.write_json(path, row)
        self.write_json(path.with_suffix(".meta.json"), {
            "protocol": PROTOCOL, "protocol_sha256": queue.digest(PROTOCOL),
            "suite_sha256": queue.file_sha(path)})

    def small_commands(self, python, gpu):
        jobs = self.real_commands(python, gpu)
        for job in jobs:
            job["n_expected"] = 1
        return jobs

    def set_handler(self, sig, handler):
        self.handlers[sig] = handler

    def launch(self, command, **kwargs):
        script = next(Path(arg) for arg in command if str(arg).endswith(".py"))
        if script.name == "register_model.py":
            event = "register"
            self.write_suite(self.inputs / "highres/suite.jsonl", self.rows["highres"])
        else:
            infer = script.name in ("run.py", "run_checkpoint.py")
            if script.name in ("run.py", "scoring.py"):
                suite = "highres"
            else:
                suite = "ocr" if "suite_ocr.jsonl" in command[command.index("--suite") + 1] else "main"
            event = ("infer:" if infer else "summary:") + suite
            if infer:
                record = {"id": self.rows[suite]["id"], "raw_response": "answer",
                          "row_sha256": queue.digest(self.rows[suite])}
                if self.bad_prediction == suite:
                    record["error"] = "transport failure"
                self.write_json(self.results / suite / "full" / queue.NAME / "predictions.jsonl", record)
            elif self.failure != event:
                result = {"status": "complete", "n_expected": 1}
                if self.bad_summary == suite:
                    result["status"] = "incomplete"
                self.write_json(self.results / suite / "summary.json", {
                    "models" if suite == "highres" else "status": {queue.NAME: result}})
        self.events.append(event)
        self.environments[event] = kwargs["env"]
        self.assertEqual(kwargs["stdin"], queue.subprocess.DEVNULL)
        self.assertTrue(kwargs["start_new_session"])
        owner = self

        class FakeProcess:
            pid = 5000 + len(owner.events)
            returncode = None if owner.interrupt == event else int(owner.failure == event)
            terminated = False

            def poll(self):
                return self.returncode

            def wait(self, timeout):
                if self.returncode is None:
                    owner.handlers[signal.SIGTERM](signal.SIGTERM, None)
                return self.returncode

            def terminate(self):
                self.terminated = True
                self.returncode = -signal.SIGTERM

        process = FakeProcess()
        self.children.append(process)
        return process

    def saved_state(self):
        return json.loads(self.state.read_text())

    def test_full_sequence_finishes_each_summary_before_next_inference(self):
        self.assertEqual(queue.main(), 0)
        self.assertEqual(self.events, ["register", "infer:highres", "summary:highres",
                                     "infer:ocr", "summary:ocr", "infer:main", "summary:main"])
        self.assertEqual(self.saved_state()["stage"], "complete")
        self.assertTrue(all(job["stage"] == "complete" for job in self.saved_state()["jobs"]))
        for event in ("register", "summary:highres", "summary:ocr", "summary:main"):
            self.assertEqual(self.environments[event]["CUDA_VISIBLE_DEVICES"], "")

    def test_highres_inference_failure_blocks_both_later_suites(self):
        self.failure = "infer:highres"
        with self.assertRaisesRegex(RuntimeError, "highres failed"):
            queue.main()
        self.assertEqual(self.events, ["register", "infer:highres"])
        self.assertEqual([j["stage"] for j in self.saved_state()["jobs"]], ["failed", "pending", "pending"])

    def test_cpu_registration_failure_starts_no_inference(self):
        self.failure = "register"
        with self.assertRaisesRegex(RuntimeError, "CPU registration"):
            queue.main()
        self.assertEqual(self.events, ["register"])
        self.assertEqual(self.saved_state()["stage"], "failed")

    def test_ocr_inference_failure_blocks_main(self):
        self.failure = "infer:ocr"
        with self.assertRaisesRegex(RuntimeError, "ocr failed"):
            queue.main()
        self.assertEqual(self.events, ["register", "infer:highres", "summary:highres", "infer:ocr"])
        self.assertEqual([j["stage"] for j in self.saved_state()["jobs"]], ["complete", "failed", "pending"])

    def test_highres_summary_failure_blocks_ocr(self):
        self.failure = "summary:highres"
        with self.assertRaisesRegex(RuntimeError, "highres summary failed"):
            queue.main()
        self.assertEqual(self.events, ["register", "infer:highres", "summary:highres"])
        self.assertEqual(self.saved_state()["jobs"][1]["stage"], "pending")

    def test_ocr_summary_failure_blocks_main(self):
        self.failure = "summary:ocr"
        with self.assertRaisesRegex(RuntimeError, "ocr summary failed"):
            queue.main()
        self.assertEqual(self.events[-1], "summary:ocr")
        self.assertNotIn("infer:main", self.events)
        self.assertEqual(self.saved_state()["jobs"][2]["stage"], "pending")

    def test_successful_exit_with_bad_predictions_blocks_next_stage(self):
        self.bad_prediction = "highres"
        with self.assertRaisesRegex(ValueError, "failed or mismatched prediction"):
            queue.main()
        self.assertEqual(self.events, ["register", "infer:highres"])
        self.assertEqual(self.saved_state()["stage"], "failed")

    def test_successful_summary_exit_with_incomplete_result_blocks_next_suite(self):
        self.bad_summary = "highres"
        with self.assertRaisesRegex(ValueError, "summary does not contain"):
            queue.main()
        self.assertEqual(self.events, ["register", "infer:highres", "summary:highres"])
        self.assertEqual(self.saved_state()["stage"], "failed")

    def test_sigterm_cleans_only_owned_launcher_and_blocks_later_jobs(self):
        self.interrupt = "infer:highres"
        with self.assertRaises(KeyboardInterrupt):
            queue.main()
        self.assertEqual(self.events, ["register", "infer:highres"])
        self.assertFalse(self.children[0].terminated)
        self.assertTrue(self.children[1].terminated)
        self.assertEqual(self.saved_state()["stage"], "interrupted")
        self.assertIsNone(self.saved_state()["child_pid"])
        self.assertNotIn("infer:ocr", self.events)


if __name__ == "__main__":
    unittest.main()
