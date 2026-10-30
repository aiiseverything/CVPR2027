"""CPU-only regression checks for the perception runner and durable queue."""
import json
import itertools
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import run
import launch_queue


def write_selection(root, **changes):
    value = {"schema_version": "highres-selection-v1", "ready": True,
             "benchmarks": {name: {"filters": [{"field": "category", "op": "eq", "value": "perception"}], "expected_count": 1}
                            for name in run.BENCHMARKS}}
    value.update(changes)
    (root / "selection.json").write_text(json.dumps(value))
    return value


class RunnerTests(unittest.TestCase):
    def test_selection_requires_ready_and_all_six_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIsNone(run.selection_snapshot(root, require_ready=False))
            write_selection(root, ready=False)
            self.assertIsNone(run.selection_snapshot(root, require_ready=False))
            with self.assertRaises(ValueError):
                run.selection_snapshot(root)
            selection = write_selection(root)
            frozen = run.selection_snapshot(root)
            self.assertEqual(frozen["n_requested"], 6)
            selection["benchmarks"]["zoom"]["expected_count"] = 0
            (root / "selection.json").write_text(json.dumps(selection))
            with self.assertRaises(ValueError):
                run.selection_snapshot(root)

    def test_selection_hash_catches_filter_change(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            value = write_selection(root)
            before = run.selection_snapshot(root)["sha256"]
            value["benchmarks"]["zoom"]["filters"] = [{"field": "category", "op": "startswith", "value": "perception"}]
            (root / "selection.json").write_text(json.dumps(value))
            self.assertNotEqual(before, run.selection_snapshot(root)["sha256"])
            value["benchmarks"]["zoom"]["filters"][0]["op"] = "unknown"
            (root / "selection.json").write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                run.selection_snapshot(root)

    def test_offline_environment_removes_laptop_proxy(self):
        with patch.dict(os.environ, {"HTTP_PROXY": "http://127.0.0.1:10820", "https_proxy": "bad", "ALL_PROXY": "bad"}):
            env = run.offline_environment()
        for key in ("HTTP_PROXY", "https_proxy", "ALL_PROXY"):
            self.assertNotIn(key, env)
        for key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
            self.assertEqual(env[key], "1")

    def test_cache_tail_recovery_and_prompt_usage(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "predictions.jsonl"
            row = {"id": "x", "prompt_tokens": 10}
            prediction = {"id": "x", "model": "baseline", "row_sha256": run.digest(row),
                          "usage": {"prompt_tokens": 10}, "raw_response": "answer", "error": None}
            path.write_text(json.dumps(prediction) + '\n{"id":')
            self.assertEqual(run.recover_predictions(path, [row], "baseline")["x"], prediction)
            self.assertTrue(list(path.parent.glob("*.bin")))
            prediction["usage"]["prompt_tokens"] = 11
            path.write_text(json.dumps(prediction) + "\n")
            with self.assertRaises(ValueError):
                run.recover_predictions(path, [row], "baseline")

    def test_subset_manifest_filters_frozen_rows_and_records_hash(self):
        rows = [{"id": "a", "benchmark": "mme_rw_en"},
                {"id": "b", "benchmark": "mme_rw_en"},
                {"id": "c", "benchmark": "mme_rw_en"}]
        selection = {"sha256": "parent", "expected_counts": {"mme_rw_en": 3}, "n_requested": 3}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "subset.json"
            manifest = {"schema_version": "highres-subset-v1", "ready": True,
                        "name": "lite", "ids": ["a", "c"]}
            path.write_text(json.dumps(manifest))
            kept, narrowed, subset = run.apply_subset(rows, selection, path)
            self.assertEqual([row["id"] for row in kept], ["a", "c"])
            self.assertEqual(narrowed["expected_counts"], {"mme_rw_en": 2})
            self.assertEqual(narrowed["source_expected_counts"], {"mme_rw_en": 3})
            self.assertEqual(narrowed["n_requested"], 2)
            self.assertEqual(subset["sha256"], run.hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(selection["n_requested"], 3)
            for broken in ({"schema_version": "highres-subset-v1", "ready": True, "ids": ["a", "z"]},
                           {"schema_version": "highres-subset-v1", "ready": True, "ids": ["a", "a"]},
                           {"schema_version": "highres-subset-v1", "ready": False, "ids": ["a"]},
                           {"schema_version": "other", "ready": True, "ids": ["a"]}):
                path.write_text(json.dumps(broken))
                with self.assertRaises(ValueError):
                    run.apply_subset(rows, selection, path)

    def test_queue_owns_prepare_freeze_then_exact_four_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "data"
            root.mkdir()
            output = Path(temporary) / "results"
            write_selection(root)
            (root / "download_state.json").write_text('{"status":"complete"}')
            commands = []
            environments = []

            class FakeChild:
                pid = 99999999
                returncode = 0

                def poll(self):
                    return 0

            def fake_launch(command, **kwargs):
                commands.append(command)
                environments.append(kwargs["env"])
                script = Path(command[1]).name
                if script in {"prepare.py", "freeze.py"}:
                    stage = script.removesuffix(".py")
                    (root / f"{stage}_state.json").write_text('{"status":"complete"}')
                    manifest = "dataset_manifest.json" if stage == "prepare" else "suite.meta.json"
                    (root / manifest).write_text(json.dumps({"selection_sha256": run.selection_snapshot(root)["sha256"]}))
                return FakeChild()

            scorer = types.ModuleType("scoring")
            summaries = []
            scorer.summarize = lambda **kwargs: summaries.append(kwargs)
            arguments = ["launch_queue.py", "--root", str(root), "--output-root", str(output),
                         "--python", sys.executable, "--poll-seconds", "1",
                         "--concurrency", "16", "--adaptive-concurrency"]
            with patch.object(sys, "argv", arguments), patch.dict(sys.modules, {"scoring": scorer}), \
                 patch.object(launch_queue.subprocess, "Popen", fake_launch), \
                 patch.object(launch_queue.signal, "signal"), \
                 patch.object(launch_queue, "gpu_status", return_value={"busy": False}), \
                 patch.object(launch_queue, "local_lock_path", return_value=Path(temporary) / "test.lock"):
                launch_queue.main()
            self.assertEqual([Path(c[1]).name for c in commands],
                             ["prepare.py", "freeze.py", "run.py", "run.py", "run.py", "run.py"])
            self.assertIn("--selection", commands[0])
            models = [c[c.index("--name") + 1] for c in commands[2:]]
            self.assertEqual(models, list(run.AUTHORIZED_MODELS))
            self.assertNotIn("generalist", models)
            self.assertTrue(all("--adaptive-concurrency" in c and c[c.index("--concurrency") + 1] == "16" for c in commands[2:]))
            self.assertEqual(len(summaries), 5)
            state = json.loads((output / "status.json").read_text())
            self.assertEqual(state["stage"], "complete")
            self.assertEqual(state["selection_sha256"], run.selection_snapshot(root)["sha256"])
            self.assertTrue(all(env["HF_HUB_OFFLINE"] == "1" for env in environments))

    def test_scheduler_migration_is_recorded_before_inference_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            before = {"root": "unchanged", "concurrency": 2, "protocol_sha256": "p"}
            after = {**before, "concurrency": 16, "adaptive_concurrency": True,
                     "scheduling_policy": launch_queue.scheduling_configuration()}
            previous = {"config": before, "jobs": {}, "child_pid": None}
            history = launch_queue.config_history(previous, after, output)
            self.assertEqual(len(history), 1)
            self.assertEqual(history[0]["before"], before)
            self.assertEqual(history[0]["after"], after)
            for change in [{"root": "different"}, {"protocol_sha256": "changed"}]:
                with self.assertRaises(ValueError):
                    launch_queue.config_history(previous, {**after, **change}, output)
            for change in [{"jobs": {"baseline": {"stage": "running"}}}, {"child_pid": 123}]:
                with self.assertRaises(ValueError):
                    launch_queue.config_history({**previous, **change}, after, output)
            model = output / "full" / "baseline"
            model.mkdir(parents=True)
            (model / "run.json").write_text('{}')
            with self.assertRaises(ValueError):
                launch_queue.config_history(previous, after, output)

    def test_throughput_fallback_only_for_current_owned_cuda_oom(self):
        for failure in ("cuda_oom", "network", "stale_oom"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary) / "data"
                root.mkdir()
                output = Path(temporary) / "results"
                write_selection(root)
                (root / "download_state.json").write_text('{"status":"complete"}')
                commands, counts = [], {}

                class Child:
                    pid = 123456789
                    returncode = 0
                    def poll(self):
                        return self.returncode

                def fake_launch(command, **kwargs):
                    child = Child()
                    script = Path(command[1]).name
                    if script in {"prepare.py", "freeze.py"}:
                        stage = script.removesuffix(".py")
                        (root / f"{stage}_state.json").write_text('{"status":"complete"}')
                        manifest = "dataset_manifest.json" if stage == "prepare" else "suite.meta.json"
                        (root / manifest).write_text(json.dumps({"selection_sha256": run.selection_snapshot(root)["sha256"]}))
                    else:
                        name = command[command.index("--name") + 1]
                        profile = int(command[command.index("--execution-profile") + 1])
                        commands.append((name, profile, command))
                        counts[name] = counts.get(name, 0) + 1
                        if name == "baseline" and counts[name] == 1:
                            child.returncode = 1
                            status = output / "full" / name / "status.json"
                            status.parent.mkdir(parents=True)
                            status.write_text(json.dumps({"pid": child.pid - (failure == "stale_oom"),
                                "config": {"execution_profile": profile}, "gpu_oom": failure != "network"}))
                    return child

                scorer = types.ModuleType("scoring")
                scorer.summarize = lambda **kwargs: None
                argv = ["launch_queue.py", "--root", str(root), "--output-root", str(output),
                        "--python", sys.executable, "--concurrency", "64", "--throughput-mode"]
                ticks = itertools.count(0, 60)
                with patch.object(sys, "argv", argv), patch.dict(sys.modules, {"scoring": scorer}), \
                     patch.object(launch_queue.subprocess, "Popen", fake_launch), \
                     patch.object(launch_queue.signal, "signal"), \
                     patch.object(launch_queue.time, "monotonic", side_effect=lambda: next(ticks)), \
                     patch.object(launch_queue, "gpu_status", return_value={"busy": False}), \
                     patch.object(launch_queue, "local_lock_path", return_value=Path(temporary) / "test.lock"):
                    launch_queue.main()
                baseline_profiles = [profile for name, profile, _ in commands if name == "baseline"]
                self.assertEqual(baseline_profiles, [0, 1] if failure == "cuda_oom" else [0, 0])
                self.assertEqual({name for name, _, _ in commands}, set(run.AUTHORIZED_MODELS))
                self.assertTrue(all("--throughput-mode" in command for _, _, command in commands))
                state = json.loads((output / "status.json").read_text())
                self.assertEqual(state["stage"], "complete")


if __name__ == "__main__":
    unittest.main()
