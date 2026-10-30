"""CPU checks of formal recipe parity and independently verified filtering evidence."""
from collections import Counter
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.sharedexpert_c.training import launch_perception_v3 as launch
import filter_perception_v3 as runner
import perception_v3_filtering as filtering
import prepare_perception_v2 as v2


class FormalRecipe(unittest.TestCase):
    def setUp(self):
        self.protocol, self.teacher, self.baseline = launch.load_protocol(["0", "1"])

    def test_entire_resolved_tree_matches_v2_except_allowed_paths(self):
        config = launch.compose_and_verify(self.protocol, self.teacher, ["0", "1"])
        self.assertEqual((config.trainer.total_training_steps, config.data.train_batch_size,
                          config.actor_rollout_ref.rollout.n), (100, 30, 8))
        self.assertEqual(config.actor_rollout_ref.actor.optim.lr, 1e-6)
        self.assertEqual(config.actor_rollout_ref.actor.optim.lr_warmup_steps, 10)
        self.assertEqual(config.trainer.save_freq, 25)
        self.assertEqual(config.trainer.max_actor_ckpt_to_keep, 1)
        self.assertTrue(config.data.apply_chat_template_kwargs.enable_thinking)
        self.assertFalse(config.actor_rollout_ref.actor.use_kl_loss)

    def test_numerical_recipe_drift_is_rejected(self):
        for key, changed in (("learning_rate", 2e-6), ("temperature", .9), ("save_every", 20)):
            with self.subTest(key=key), self.assertRaisesRegex(RuntimeError, "Resolved configuration differs"):
                launch.compose_and_verify({**self.protocol, key: changed}, self.teacher, ["0", "1"])

    def test_only_explicit_workspace_and_model_paths_are_relocated(self):
        original = self.baseline["protocol"]["model"]
        value = {"path": original, "workspace": "/autodl-fs/data/CVPR2027/x",
                 "unrelated": "/tmp/other", "scalar": .2}
        changed = launch.relocate_baseline_paths(value, original, self.protocol["model"])
        self.assertEqual(changed["path"], self.protocol["model"])
        self.assertEqual(changed["workspace"], str(launch.WORKSPACE / "x"))
        self.assertEqual(changed["unrelated"], "/tmp/other")
        self.assertEqual(changed["scalar"], .2)

    def test_shared_sources_and_repo_match_frozen_v2(self):
        identity, _ = launch.recipe_identity(self.protocol, self.baseline)
        self.assertEqual(identity["reward_sha256"], self.baseline["reward_sha256"])
        self.assertEqual(identity["repository_diff_sha256"], self.baseline["repository_diff_sha256"])

    def test_separate_smoke_is_not_a_launcher_mode(self):
        result = subprocess.run([sys.executable, str(Path(launch.__file__)), "--smoke"],
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("--check | --dry-run | --execute | --merge-only", result.stderr)

    def test_checkpoint_capacity_uses_two_checkpoint_peak_and_resume_credit(self):
        size, export = 54_493_602_998, 10_350_019_328
        reserve = 3 * 1024**3
        self.assertEqual(launch.storage_requirement(size, export), size * 2 + reserve)
        self.assertEqual(launch.storage_requirement(size, export, 50, size), size + reserve)
        self.assertEqual(launch.storage_requirement(size, export, 100, size), export + reserve)

    def test_inode_exhaustion_blocks_formal_training_despite_free_bytes(self):
        filesystem = SimpleNamespace(f_files=200_000, f_ffree=2047, f_favail=2047)
        with patch.object(launch.os, "statvfs", return_value=filesystem):
            with self.assertRaisesRegex(RuntimeError, "Insufficient filesystem inodes"):
                launch.resource_check(launch.RUN_DIR, 0)

    def test_launcher_disables_bytecode_writes(self):
        self.assertTrue(sys.dont_write_bytecode)


class ScreeningEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.image = cls.root / "fixture.png"
        Image.new("RGB", (16, 16), "white").save(cls.image)
        cls.protocol, _, _ = launch.load_protocol(["0", "1"])
        cls.output = cls.root / "screen"
        cls.output.mkdir()
        spec = {
            "group_size": 8, "target": 3000, "seed": 19,
            "sampling": {key: cls.protocol[key] for key in ("temperature", "top_p", "top_k", "thinking",
                          "prompt_length", "response_length", "min_image_pixels", "max_image_pixels", "max_total_image_pixels")},
            "reward_source_sha256": launch.local.sha256(launch.EXPERIMENT / "rewards.py"),
            "filter_source_sha256": launch.local.sha256(launch.PIPELINE / "perception_v3_filtering.py"),
            "model_files": {"fixture": "not a model"},
        }
        cls.fingerprint = launch.digest(spec)
        (cls.output / "run.json").write_text(json.dumps({"spec": spec, "run_sha256": cls.fingerprint}))
        records, groups = [], {}
        for category, weight in filtering.WEIGHTS.items():
            for index in range(weight * 30):
                answer = "yes" if index % 2 == 0 else "no"
                question = f"Is there a dog in fixture {category} {index}?"
                record = {"id": f"{category}:{index}", "source": "vqav2", "source_id": str(index),
                          "split": "train", "question": question, "question_sha256": v2.question_hash(question),
                          "images": [str(cls.image)], "image_sha256": [v2.pixel_hash(str(cls.image))],
                          "task": "vqa", "answers": [answer] * 10,
                          "perception_category": category, "selection_rule": "fixture"}
                other = "no" if answer == "yes" else "yes"
                completions = [{"index": j, "text": "</think>\\boxed{" +
                                (answer if j < 4 else other) + "}", "finish_reason": "stop"}
                               for j in range(8)]
                groups[record["id"]] = {"id": record["id"], "run_sha256": cls.fingerprint,
                    "candidate_sha256": runner.record_identity(record),
                    "sampling_seed": runner.record_seed(record["id"], 19),
                    "completions": completions, "difficulty_filter": filtering.accept_group(record, completions)}
                records.append(record)
        (cls.output / "groups.jsonl").write_text("".join(json.dumps(group) + "\n" for group in groups.values()))
        assert runner.publish(records, groups, cls.output, cls.protocol, cls.fingerprint, 3000, 19, [])

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_replays_all_selected_raw_groups_and_preserves_training_rows(self):
        report = launch.validate_dataset(self.output / "dataset", self.protocol, verify_model=False)
        self.assertEqual(report["rows"], 3000)
        self.assertEqual(report["raw_groups_replayed"], 3000)
        self.assertEqual(report["verified_image_files"], 1)

    def test_training_base_model_must_match_screening_model(self):
        with patch.object(runner, "model_identity", return_value={"different": "weights"}):
            with self.assertRaisesRegex(RuntimeError, "Training base weights"):
                launch.validate_dataset(self.output / "dataset", self.protocol)

    def test_incomplete_sampling_spec_cannot_pass_by_omitting_changed_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "screen"
            shutil.copytree(self.output, root)
            spec_path = root / "run.json"
            screen = launch.read_json(spec_path)
            del screen["spec"]["sampling"]["temperature"]
            screen["run_sha256"] = launch.digest(screen["spec"])
            spec_path.write_text(json.dumps(screen))
            manifest_path = root / "dataset/manifest.json"
            manifest = launch.read_json(manifest_path)
            manifest["run_sha256"] = screen["run_sha256"]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "sampling or image settings"):
                launch.validate_dataset(root / "dataset", self.protocol, verify_model=False)

    def test_changed_raw_completion_is_rejected_even_with_updated_file_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "screen"
            shutil.copytree(self.output, root)
            path = root / "groups.jsonl"
            groups = [json.loads(line) for line in path.read_text().splitlines()]
            groups[0]["completions"][0]["text"] = "</think>\\boxed{not-the-answer}"
            path.write_text("".join(json.dumps(group) + "\n" for group in groups))
            manifest_path = root / "dataset/manifest.json"
            manifest = launch.read_json(manifest_path)
            manifest["groups_sha256"] = launch.local.sha256(path)
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "independent eight-completion scoring"):
                launch.validate_dataset(root / "dataset", self.protocol, verify_model=False)

    def test_image_mutation_is_rejected_after_all_receipts_replay(self):
        with patch.object(v2, "pixel_hash", return_value="changed"):
            with self.assertRaisesRegex(RuntimeError, "Training image changed"):
                launch.validate_dataset(self.output / "dataset", self.protocol, verify_model=False)


class RetiredExportStorage(unittest.TestCase):
    def test_capacity_checks_work_after_historical_export_directory_is_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "runs/perception_v2"
            baseline.mkdir(parents=True)
            for name in ("protocol.json", "resolved_config.yaml", "resource_usage.json"):
                shutil.copyfile(launch.BASELINE / name, baseline / name)
            (root / "runtime").mkdir()
            shutil.copyfile(launch.EXPERIMENT / "runtime/perception_v2_queue.json", root / "runtime/perception_v2_queue.json")
            reference = root / "storage_reference.json"
            shutil.copyfile(launch.STORAGE_REFERENCE, reference)
            original_read = Path.read_text

            def read_with_idle_cgroup(path, *args, **kwargs):
                overrides = {"/sys/fs/cgroup/memory.max": "max", "/sys/fs/cgroup/memory.current": "0",
                             "/sys/fs/cgroup/memory.stat": "inactive_file 0\n"}
                return overrides.get(str(path), None) if str(path) in overrides else original_read(path, *args, **kwargs)

            filesystem = SimpleNamespace(f_files=200_000, f_ffree=10_000, f_favail=10_000)
            with patch.object(launch, "EXPERIMENT", root), patch.object(launch, "BASELINE", baseline), \
                 patch.object(launch, "STORAGE_REFERENCE", reference), \
                 patch.object(launch.shutil, "disk_usage", return_value=SimpleNamespace(free=200_000_000_000)), \
                 patch.object(launch.os, "statvfs", return_value=filesystem), \
                 patch.object(Path, "read_text", read_with_idle_cgroup):
                report = launch.resource_check(root / "runs/perception_v3", 0)
            self.assertFalse((baseline / "hf").exists())
            self.assertEqual(report["checkpoint_bytes"], 54_493_602_998)
            self.assertEqual(report["export_bytes"], 10_350_019_328)
            self.assertEqual(report["required_free_bytes"], 112_208_431_468)

    def test_tampered_snapshot_hash_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "storage.json"
            value = launch.read_json(launch.STORAGE_REFERENCE)
            value["hf_export_bytes"] = 1
            reference.write_text(json.dumps(value))
            with patch.object(launch, "STORAGE_REFERENCE", reference):
                with self.assertRaisesRegex(RuntimeError, "payload hash mismatch"):
                    launch.load_storage_reference()

    def test_inconsistent_snapshot_sizes_are_rejected_even_with_updated_payload_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / "storage.json"
            value = launch.read_json(launch.STORAGE_REFERENCE)
            value["hf_export_bytes"] = 1
            value["payload_sha256"] = launch.digest({key: item for key, item in value.items() if key != "payload_sha256"})
            reference.write_text(json.dumps(value))
            with patch.object(launch, "STORAGE_REFERENCE", reference):
                with self.assertRaisesRegex(RuntimeError, "byte count mismatch"):
                    launch.load_storage_reference()


class ParallelExecutionEvidence(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.execution = self.root / "executions"
        (self.execution / "sources").mkdir(parents=True)
        self.pipeline = self.root / "pipeline"
        self.pipeline.mkdir()
        (self.pipeline / "filter_perception_v3.py").write_text("# unchanged core\n")
        (self.pipeline / "filter_perception_v3_parallel.py").write_text("# current wrapper\n")
        self.core_hash = launch.local.sha256(self.pipeline / "filter_perception_v3.py")
        self.parallel_hash = launch.local.sha256(self.pipeline / "filter_perception_v3_parallel.py")
        self.legacy = b'{"id":"legacy-a"}\n{"id":"legacy-b"}\n'
        self.final = self.legacy + b'{"id":"parallel-c"}\n'
        (self.root / "groups.jsonl").write_bytes(self.final)
        (self.execution / "legacy-before-parallel.groups.jsonl").write_bytes(self.legacy)
        self.screen = {"run_sha256": "frozen-run", "spec": {"driver_source_sha256": self.core_hash}}
        self.entry = self.make_entry()
        self.write_entry(self.entry)
        self.patched = patch.object(launch, "PIPELINE", self.pipeline)
        self.patched.start()

    def tearDown(self):
        self.patched.stop()
        self.temporary.cleanup()

    def make_entry(self, wrapper_text="# current wrapper\n", started="2026-10-29T12:00:00+00:00", status="complete"):
        source_hash = __import__("hashlib").sha256(wrapper_text.encode()).hexdigest()
        source = Path("sources") / f"parallel-{source_hash}.py"
        (self.execution / source).write_text(wrapper_text)
        return {"schema": 1, "backend": "independent_gpu_replicas", "devices": [0, 1],
                "tensor_parallel_size_per_worker": 1, "core_driver_sha256": self.core_hash,
                "parallel_driver_sha256": source_hash, "parallel_driver_source": str(source),
                "run_sha256": "frozen-run", "initial_groups_sha256": __import__("hashlib").sha256(self.legacy).hexdigest(),
                "initial_groups_bytes": len(self.legacy), "initial_group_count": 2,
                "final_groups_sha256": __import__("hashlib").sha256(self.final).hexdigest(),
                "final_groups_bytes": len(self.final), "final_group_count": 3,
                "batch_prompts": 24, "pid": 100, "start_ticks": 12, "started_at": started,
                "status": status, "workers": [{"gpu": 0, "pid": 101, "start_ticks": 13},
                                                {"gpu": 1, "pid": 102, "start_ticks": 14}]}

    def write_entry(self, entry, name="parallel-20261029T120000-100.json"):
        (self.execution / name).write_text(json.dumps(entry))

    def test_dual_gpu_history_preserves_every_original_group(self):
        report = launch.validate_execution_provenance(self.root, self.screen)
        self.assertEqual(report["backend"], "independent_gpu_replicas")
        self.assertEqual(report["legacy_group_count"], 2)
        self.assertEqual(report["current_parallel_driver_sha256"], self.parallel_hash)

    def test_archived_failed_wrapper_version_can_precede_current_success(self):
        failed = self.make_entry("# earlier failed wrapper\n", "2026-10-29T11:00:00+00:00", "failed")
        failed.update(final_groups_bytes=len(self.legacy), final_group_count=2,
                      final_groups_sha256=failed["initial_groups_sha256"], workers=[])
        self.write_entry(failed, "parallel-20261029T110000-90.json")
        report = launch.validate_execution_provenance(self.root, self.screen)
        self.assertEqual(len(report["execution_receipts"]), 2)

    def test_changed_legacy_prefix_is_rejected(self):
        (self.root / "groups.jsonl").write_bytes(self.final.replace(b"legacy-a", b"tamper-a"))
        with self.assertRaisesRegex(RuntimeError, "single-GPU groups changed"):
            launch.validate_execution_provenance(self.root, self.screen)

    def test_modified_archived_wrapper_is_rejected(self):
        (self.execution / self.entry["parallel_driver_source"]).write_text("# changed archive\n")
        with self.assertRaisesRegex(RuntimeError, "Archived parallel execution source changed"):
            launch.validate_execution_provenance(self.root, self.screen)

    def test_latest_wrapper_must_match_current_execution_source(self):
        (self.pipeline / "filter_perception_v3_parallel.py").write_text("# changed current source\n")
        with self.assertRaisesRegex(RuntimeError, "Latest parallel execution source differs"):
            launch.validate_execution_provenance(self.root, self.screen)

    def test_running_execution_cannot_start_formal_training(self):
        self.entry["status"] = "running"
        self.write_entry(self.entry)
        with self.assertRaisesRegex(RuntimeError, "terminal state"):
            launch.validate_execution_provenance(self.root, self.screen)

    def test_uncovered_journal_tail_is_rejected(self):
        (self.root / "groups.jsonl").write_bytes(self.final + b'{"id":"unrecorded-tail"}\n')
        with self.assertRaisesRegex(RuntimeError, "not covered by a completed"):
            launch.validate_execution_provenance(self.root, self.screen)

    def test_prefix_cannot_end_inside_a_group(self):
        self.entry["initial_groups_bytes"] -= 1
        self.write_entry(self.entry)
        with self.assertRaisesRegex(RuntimeError, "uncommitted group"):
            launch.validate_execution_provenance(self.root, self.screen)

    def test_original_single_gpu_dataset_requires_no_parallel_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            report = launch.validate_execution_provenance(Path(directory), {"spec": {}})
            self.assertEqual(report["backend"], "original_single_gpu")


if __name__ == "__main__":
    unittest.main()
