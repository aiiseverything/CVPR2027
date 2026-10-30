"""CPU tests for v2 protocol parity, mixture integrity, and export gates."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.sharedexpert_c.training import launch_perception_v2 as launch


class PerceptionV2LaunchTests(unittest.TestCase):
    def setUp(self):
        self.protocol = launch.read_json(launch.BASELINE / "protocol.json")["protocol"]
        self.teacher = {"teacher": "perception_v2",
                        "data_file": "experiments/sharedexpert_c/data_pipeline/prepared/perception_v2.parquet",
                        "category_weights": {key: value / 100 for key, value in launch.CATEGORIES.items()}}

    def test_formal_resolved_config_equals_actual_generalist_except_dataset_and_paths(self):
        overrides, teacher, run, steps, batch = launch.make_overrides(self.protocol, self.teacher, ["0", "1"])
        cfg = launch.compose_and_verify(overrides, self.protocol, teacher, run, steps, batch)
        self.assertEqual((cfg.trainer.total_training_steps, cfg.data.train_batch_size,
                          cfg.actor_rollout_ref.rollout.n), (100, 30, 8))
        self.assertEqual(cfg.actor_rollout_ref.model.path, self.protocol["model"])
        self.assertTrue(cfg.data.apply_chat_template_kwargs.enable_thinking)
        self.assertTrue(cfg.actor_rollout_ref.actor.fsdp_config.optimizer_offload)
        self.assertFalse(cfg.actor_rollout_ref.actor.freeze_vision_tower)

    def test_smoke_preserves_lengths_and_has_six_prompts_and_nonzero_initial_lr(self):
        overrides, teacher, run, steps, batch = launch.make_overrides(self.protocol, self.teacher, ["0", "1"], True)
        cfg = launch.compose_and_verify(overrides, self.protocol, teacher, run, steps, batch)
        self.assertEqual((steps, batch), (1, 6))
        self.assertEqual(cfg.actor_rollout_ref.actor.optim.lr_warmup_steps, 0)
        self.assertEqual((cfg.data.max_prompt_length, cfg.data.max_response_length), (2048, 8192))
        self.assertEqual(cfg.actor_rollout_ref.rollout.n, 8)

    def test_silent_hyperparameter_drift_is_rejected(self):
        altered = {**self.protocol, "learning_rate": 2e-6}
        overrides, teacher, run, steps, batch = launch.make_overrides(altered, self.teacher, ["0", "1"])
        with self.assertRaisesRegex(RuntimeError, "Resolved configuration differs"):
            launch.compose_and_verify(overrides, altered, teacher, run, steps, batch)

    def test_shared_recipe_is_still_identical_to_generalist(self):
        baseline = launch.read_json(launch.BASELINE / "protocol.json")
        identity, _ = launch.recipe_identity(self.protocol, baseline)
        self.assertEqual(identity["reward_sha256"], baseline["reward_sha256"])
        altered = {**baseline, "reward_sha256": "changed"}
        with self.assertRaisesRegex(RuntimeError, "Reward changed"):
            launch.recipe_identity(self.protocol, altered)

    def test_data_mixture_rejects_equal_total_but_wrong_order(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        block = [category for category, weight in launch.CATEGORIES.items() for _ in range(weight // 5)]
        def row(category):
            return {"prompt": [{"role": "user", "content": "Question"}], "images": [],
                    "data_source": "vqa", "reward_model": {"ground_truth": json.dumps({"task": "vqa", "answers": ["yes"]})},
                    "extra_info": {"perception_category": category}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.parquet"
            rows = [row(category) for category in block * 150]
            pq.write_table(pa.Table.from_pylist(rows), path)
            self.assertFalse(launch.validate_mixture(path)["errors"])
            rows[0], rows[24] = rows[24], rows[0]
            pq.write_table(pa.Table.from_pylist(rows), path)
            self.assertIn("Mixture drift", launch.validate_mixture(path)["errors"][0])

    def test_export_requires_complete_two_gpu_checkpoint_and_hf_files(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with self.assertRaisesRegex(RuntimeError, "Missing final FSDP"):
                launch.validate_export(run, 100, 2)
            checkpoint = run / "checkpoints/global_step_100/actor"
            export = run / "hf/global_step_100"
            checkpoint.mkdir(parents=True)
            export.mkdir(parents=True)
            (checkpoint / "fsdp_config.json").write_text("{}")
            for kind in ("model", "optim"):
                for rank in range(2):
                    (checkpoint / f"{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"weight")
            (export / "sharedexpert_export_complete.json").write_text('{"step":100}')
            (export / "config.json").write_text("{}")
            (export / "model.safetensors").write_bytes(b"weight")
            self.assertEqual(launch.validate_export(run, 100, 2), export)
            (checkpoint / "model_world_size_2_rank_1.pt").unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing model checkpoint shards"):
                launch.validate_export(run, 100, 2)

    def _smoke_fixture(self, root):
        (root / "checkpoints/global_step_1/actor").mkdir(parents=True)
        (root / "hf/global_step_1").mkdir(parents=True)
        (root / "rollouts").mkdir()
        (root / "tensorboard").mkdir()
        for name in ("protocol.json", "smoke_validation.json", "resource_usage.json"):
            (root / name).write_text('{"passed":true}')
        (root / "rollouts/1.jsonl").write_text('{"score":1}')
        (root / "tensorboard/events.test").write_bytes(b"metrics")
        (root / "hf/global_step_1/config.json").write_text("{}")
        for kind in ("model", "optim"):
            for rank in range(2):
                (root / f"checkpoints/global_step_1/actor/{kind}_world_size_2_rank_{rank}.pt").write_bytes(b"model")
        (root / "hf/global_step_1/model.safetensors").write_bytes(b"export")

    def test_smoke_cleanup_preserves_evidence_and_rejects_evidence_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "perception_v2_smoke"
            self._smoke_fixture(root)
            sentinel = Path(directory) / "generalist.pt"
            sentinel.write_bytes(b"do not touch")
            with patch.object(launch, "SMOKE_DIR", root), patch.object(launch, "validate_smoke") as full_check:
                receipt = launch.cleanup_smoke({"identity": "test"})
                full_check.assert_called_once_with({"identity": "test"}, allow_prepared_cleanup=True)
                self.assertEqual(receipt["status"], "complete")
                self.assertEqual(receipt["total_weight_bytes"], 26)
                self.assertEqual(len(receipt["weights"]), 5)
                self.assertTrue((root / "hf/global_step_1/config.json").exists())
                self.assertEqual(sentinel.read_bytes(), b"do not touch")
                self.assertFalse(launch.smoke_weight_paths())
                launch.validate_cleanup_receipt({"identity": "test"})
                (root / "tensorboard/events.test").write_bytes(b"altered")
                with self.assertRaisesRegex(RuntimeError, "Retained smoke evidence changed"):
                    launch.validate_cleanup_receipt({"identity": "test"})

    def test_partial_smoke_cleanup_requires_explicit_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "perception_v2_smoke"
            self._smoke_fixture(root)
            with patch.object(launch, "SMOKE_DIR", root), patch.object(launch, "validate_smoke"):
                receipt = launch.cleanup_smoke({"identity": "test"})
                receipt["status"] = "prepared"
                launch.write_json(root / "smoke_cleanup.json", receipt)
                with self.assertRaisesRegex(RuntimeError, "cleanup is incomplete"):
                    launch.validate_cleanup_receipt({"identity": "test"})
                self.assertEqual(launch.cleanup_smoke({"identity": "test"})["status"], "complete")

    def test_smoke_cleanup_rejects_outside_paths_and_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "perception_v2_smoke"
            self._smoke_fixture(root)
            outside = Path(directory) / "generalist.pt"
            outside.write_bytes(b"keep")
            with patch.object(launch, "SMOKE_DIR", root):
                for relative in ("../generalist.pt", "logs/other.pt", str(outside)):
                    with self.assertRaises(RuntimeError):
                        launch.safe_smoke_path(relative, weight=True)
                (root / "hf/global_step_1/outside.safetensors").symlink_to(outside)
                with self.assertRaisesRegex(RuntimeError, "Refusing symlink"):
                    launch.smoke_weight_paths()
                self.assertEqual(outside.read_bytes(), b"keep")

    def test_cleaned_smoke_keeps_enforcing_learning_signals_without_export_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "perception_v2_smoke"
            self._smoke_fixture(root)
            data = Path(directory) / "perception_v2_smoke.parquet"
            data.write_bytes(b"data")
            identity = {"recipe": "test"}
            launch.write_json(root / "protocol.json", {**identity, "data_sha256": launch.local.sha256(data)})
            launch.write_json(root / "resource_usage.json", {"memory_events_delta": {"oom": 0, "oom_kill": 0},
                              "working_set_peak_bytes": 10, "gpu_peak_used_bytes": {"0": 1, "1": 1}})
            rows = [{"step": 1, "group_uid": str(group), "prompt_tokens": 10, "response_tokens": 10,
                     "score": float(sample % 2)} for group in range(6) for sample in range(8)]
            (root / "rollouts/1.jsonl").write_text("\n".join(json.dumps(row) for row in rows))
            tags = ["actor/pg_loss", "actor/grad_norm", "actor/lr", "rollout_corr/rollout_is_mean",
                    "rollout_corr/rollout_is_min", "rollout_corr/rollout_is_max"]
            metric_values = dict.fromkeys(tags, 1.0)
            events = SimpleNamespace(Tags=lambda: {"scalars": tags},
                                     Scalars=lambda tag: [SimpleNamespace(step=1, value=metric_values[tag])])
            accumulator = SimpleNamespace(Reload=lambda: events)
            with patch.object(launch, "SMOKE_DIR", root), patch.object(launch, "DATA_DIR", Path(directory)), \
                 patch("tensorboard.backend.event_processing.event_accumulator.EventAccumulator", return_value=accumulator), \
                 patch.object(launch, "validate_export", return_value=root / "hf/global_step_1") as export_check:
                launch.cleanup_smoke(identity)
                validation_hash = launch.local.sha256(root / "smoke_validation.json")
                export_check.side_effect = AssertionError("Archived weights must not be reopened")
                self.assertTrue(launch.validate_smoke(identity)["passed"])
                self.assertEqual(launch.local.sha256(root / "smoke_validation.json"), validation_hash)
                metric_values["actor/grad_norm"] = 0.0
                with self.assertRaisesRegex(RuntimeError, "nonzero gradient"):
                    launch.validate_smoke(identity)


if __name__ == "__main__":
    unittest.main()
