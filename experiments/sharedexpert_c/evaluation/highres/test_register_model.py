"""CPU regression tests for isolated export registration and legacy queue scope."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import launch_queue
import protocol
import register_model
import run


class RegistrationTests(unittest.TestCase):
    def make_inputs(self, temporary):
        source, view, model = (Path(temporary) / name for name in ("source", "view", "model"))
        source.mkdir()
        model.mkdir()
        rows = [{"id": name + ":0", "benchmark": name} for name in sorted(run.BENCHMARKS)]
        selection = {"schema_version": "highres-selection-v1", "ready": True,
                     "benchmarks": {name: {"expected_count": 1, "include_source_ids": ["0"]}
                                    for name in run.BENCHMARKS}}
        (source / "selection.json").write_text(json.dumps(selection))
        for filename in ("suite.jsonl", "questions.jsonl"):
            (source / filename).write_text("".join(json.dumps(row) + "\n" for row in rows))
        for filename in ("source_manifest.json", "dataset_manifest.json"):
            (source / filename).write_text("{}")
        (source / "images.sqlite").write_bytes(b"test database")
        meta = {"protocol": protocol.PROTOCOL, "protocol_sha256": protocol.digest(protocol.PROTOCOL),
                "counts": {name: 1 for name in run.BENCHMARKS}, "models": {"baseline": {"unchanged": True}},
                **{key: protocol.file_sha(source / filename)
                   for filename, key in register_model.SOURCE_FILES.items()}}
        (source / "suite.meta.json").write_text(json.dumps(meta))
        for filename in ("config.json", "tokenizer_config.json", "processor_config.json"):
            (model / filename).write_text("{}")
        (model / "sharedexpert_export_complete.json").write_text('{"step":100}')
        (model / "model.safetensors").write_bytes(b"w" * 1024)
        return source, view, model, rows

    def cpu_validation(self):
        return patch.object(register_model, "validate_processor", return_value={
            "context_capacity": 262144, "processor_class": "TestProcessor",
            "processor_verification": "checked", "processor_checks": {"n_rows": 1}})

    def test_private_registration_preserves_original_bytes_and_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary, self.cpu_validation() as validate:
            source, view, model, rows = self.make_inputs(temporary)
            original = {path.name: path.read_bytes() for path in source.iterdir()}
            receipt = register_model.register(source, view, model, "perception-v3-step100", ["vstar"])
            self.assertEqual(receipt["n_validated"], 1)
            self.assertFalse(receipt["gpu_used"])
            self.assertEqual(original, {path.name: path.read_bytes() for path in source.iterdir()})
            for filename in register_model.LINK_FILES:
                self.assertTrue((view / filename).is_symlink())
                self.assertEqual((view / filename).resolve(), source / filename)
            self.assertFalse((view / "suite.meta.json").is_symlink())
            self.assertEqual(register_model.register(source, view, model, "perception-v3-step100", ["vstar"]), receipt)
            self.assertEqual(validate.call_count, 1)
            meta = json.loads((view / "suite.meta.json").read_text())
            selected = [row for row in rows if row["benchmark"] == "vstar"]
            self.assertEqual(run.resolve_model("perception-v3-step100", model, meta, selected), model)
            with self.assertRaisesRegex(ValueError, "exact frozen subset"):
                run.resolve_model("perception-v3-step100", model, meta, rows)
            with self.assertRaisesRegex(ValueError, "explicit --model"):
                run.resolve_model("perception-v3-step100", None, meta, selected)

    def test_registration_cannot_overwrite_source_or_existing_metadata(self):
        with tempfile.TemporaryDirectory() as temporary, self.cpu_validation():
            source, view, model, _ = self.make_inputs(temporary)
            with self.assertRaisesRegex(ValueError, "private root"):
                register_model.register(source, source, model, "perception-v3-step100", ["vstar"])
            register_model.register(source, view, model, "perception-v3-step100", ["vstar"])
            before = (view / "suite.meta.json").read_bytes()
            with self.assertRaisesRegex(ValueError, "refusing replacement"):
                register_model.register(source, view, model, "perception-v3-step100", ["zoom"])
            self.assertEqual((view / "suite.meta.json").read_bytes(), before)

    def test_changed_source_and_incomplete_export_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, self.cpu_validation() as validate:
            source, view, model, _ = self.make_inputs(temporary)
            (model / "sharedexpert_export_complete.json").write_text('{"step":75}')
            with self.assertRaisesRegex(ValueError, "completed final step100"):
                register_model.register(source, view, model, "perception-v3-step100", ["vstar"])
            (model / "sharedexpert_export_complete.json").write_text('{"step":100}')
            (source / "questions.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "Frozen source artifact changed"):
                register_model.register(source, view, model, "perception-v3-step100", ["vstar"])
            validate.assert_not_called()

    def test_failed_cpu_validation_never_publishes_registration(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, view, model, _ = self.make_inputs(temporary)
            with patch.object(register_model, "validate_processor", side_effect=ValueError("token mismatch")):
                with self.assertRaisesRegex(ValueError, "token mismatch"):
                    register_model.register(source, view, model, "perception-v3-step100", ["vstar"])
            self.assertFalse((view / "suite.meta.json").exists())

    def test_registration_detects_export_changes_during_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, view, model, _ = self.make_inputs(temporary)
            def mutate(*args):
                (model / "config.json").write_text('{"changed":true}')
                return {"processor_verification": "checked"}
            with patch.object(register_model, "validate_processor", side_effect=mutate):
                with self.assertRaisesRegex(ValueError, "changed during registration"):
                    register_model.register(source, view, model, "perception-v3-step100", ["vstar"])
            self.assertFalse((view / "suite.meta.json").exists())

    def test_legacy_queue_still_contains_only_original_four_models(self):
        self.assertEqual(launch_queue.ORDER, ["baseline", "reasoning-step100", "perception-step100", "perception-v2-step100"])
        self.assertNotIn("perception-v3-step100", protocol.MODEL_PATHS)
        self.assertEqual(run.resolve_model("baseline", None, {}, []), protocol.MODEL_PATHS["baseline"].resolve())


if __name__ == "__main__":
    unittest.main()
