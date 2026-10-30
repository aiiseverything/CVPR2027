"""CPU-only tests of explicit domain contracts and corrupt export rejection."""
from copy import deepcopy
import json
from pathlib import Path
import struct
import tempfile
import unittest

from experiments.sharedexpert_c.distillation.config import (
    DEFAULTS, TEACHERS, check_artifacts, load_config, missing_requirements, validate_config,
    validate_vocab_compatibility,
)
from experiments.sharedexpert_c.distillation.routing import get_domain, route_batch, route_record


class RoutingTest(unittest.TestCase):
    def setUp(self):
        self.cfg = deepcopy(DEFAULTS)
        self.cfg["domain_to_teacher"] = {"count": "perception", "math": "reasoning", "vqa": "generalist"}

    def test_explicit_mapping_order_and_dotted_metadata(self):
        self.cfg["domain_key"] = "extra_info.domain"
        rows = [{"extra_info": {"domain": domain}} for domain in ("vqa", "math", "count", "math")]
        self.assertEqual(route_batch(rows, self.cfg), ["generalist", "reasoning", "perception", "reasoning"])

    def test_no_guess_from_source_or_question(self):
        for row in ({"source": "geometry3k"}, {"domain": "geometry3k"}, {"domain": None}, {"domain": 2}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                route_record(row, self.cfg)

    def test_no_implicit_case_normalization_or_fallback(self):
        for value in ("Math", " math", "math ", ""):
            with self.subTest(value=value), self.assertRaises(ValueError):
                route_record({"domain": value}, self.cfg)
        with self.assertRaisesRegex(ValueError, "Row 1"):
            route_batch([{"domain": "math"}, {"domain": "bad"}], self.cfg)

    def test_dotted_missing_path_and_malformed_domain_key(self):
        for key in ("extra_info.domain", "extra_info..domain", ""):
            with self.subTest(key=key), self.assertRaises(ValueError):
                get_domain({"domain": "math"}, key)


class ConfigurationTest(unittest.TestCase):
    def test_reserved_append_only_special_tokens(self):
        student = {"a": 0, "b": 1}
        teacher = {**student, "<audio>": 2}
        self.assertEqual(validate_vocab_compatibility(student, teacher, ["<audio>"], 4), [])
        for bad, special, size in ((teacher, [], 4), (teacher, ["<audio>"], 2),
                                    ({"a": 0, "b": 0}, [], 4),
                                    ({"a": 1, "b": 0}, [], 4), ({"a": 0}, [], 4)):
            with self.subTest(teacher=bad, special=special, size=size):
                self.assertTrue(validate_vocab_compatibility(student, bad, special, size))

    def test_draft_load_never_requires_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draft.json"
            path.write_text('{"student_path": null, "data_path": null}')
            cfg = load_config(path)
        self.assertEqual(validate_config(cfg), [])
        self.assertIn("student_path", missing_requirements(cfg))
        self.assertIn("domain_to_teacher", missing_requirements(cfg))
        self.assertTrue(validate_config(cfg, require_resolved=True))

    def test_path_resolution_is_explicit(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "draft.json"
            path.write_text('{"student_path": "models/student"}')
            self.assertEqual(load_config(path, base_dir=tmp)["student_path"], str(Path(tmp) / "models/student"))

    def test_reject_unsupported_or_nonfinite_settings(self):
        cases = {"algorithm": "grpo", "epochs": 2, "rollout_n": 2, "temperature": 0.7, "top_p": 0.9,
                 "top_k": 20, "steps": True, "prompt_length": 0, "response_length": -1,
                 "learning_rate": float("nan"), "microbatch_size": 7}
        for key, value in cases.items():
            cfg = deepcopy(DEFAULTS)
            cfg[key] = value
            with self.subTest(key=key):
                self.assertTrue(validate_config(cfg))

    def test_reject_bad_mapping_and_teacher_path_types(self):
        for value in ({"math": "unknown"}, {"": "reasoning"}, {"math": []}, []):
            cfg = deepcopy(DEFAULTS)
            cfg["domain_to_teacher"] = value
            with self.subTest(value=value):
                self.assertTrue(validate_config(cfg))


def write_model(path, *, swapped=False):
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": {"vocab_size": 3}}))
    for filename in ("processor_config.json", "tokenizer_config.json"):
        (path / filename).write_text("{}")
    (path / "preprocessor_config.json").write_text(json.dumps({
        "size": {"longest_edge": 1024, "shortest_edge": 64}, "patch_size": 16,
        "temporal_patch_size": 2, "merge_size": 2, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}))
    (path / "chat_template.jinja").write_text("{{ messages }}")
    (path / "tokenizer.json").write_text(json.dumps({"model": {"vocab": {"a": int(swapped), "b": int(not swapped)}},
                                                    "added_tokens": [{"content": "<eos>", "id": 2}]}))
    header = json.dumps({"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + b"\0" * 4)


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = deepcopy(DEFAULTS)
        self.cfg["domain_to_teacher"] = {"math": "reasoning"}
        self.cfg["data_path"] = str(self.root / "train.parquet")
        Path(self.cfg["data_path"]).write_bytes(b"placeholder; parquet schema tested by launcher")
        for role in ("student", *TEACHERS):
            path = self.root / role
            write_model(path)
            if role == "student":
                self.cfg["student_path"] = str(path)
            else:
                self.cfg["teacher_paths"][role] = str(path)

    def test_complete_exports_pass_metadata_check(self):
        self.assertEqual(check_artifacts(self.cfg), [])

    def test_same_size_but_different_token_ids_fails(self):
        path = self.root / "reasoning/tokenizer.json"
        content = json.loads(path.read_text())
        content["model"]["vocab"] = {"a": 1, "b": 0}
        path.write_text(json.dumps(content))
        self.assertTrue(any("reasoning: tokenizer" in error for error in check_artifacts(self.cfg)))

    def test_truncated_weights_fail(self):
        path = self.root / "perception/model.safetensors"
        path.write_bytes(path.read_bytes()[:-1])
        self.assertTrue(any("perception:" in error and "payload" in error for error in check_artifacts(self.cfg)))

    def test_index_missing_shard_and_tensor_mismatch_fail(self):
        index = self.root / "generalist/model.safetensors.index.json"
        index.write_text(json.dumps({"weight_map": {"weight": "absent.safetensors"}}))
        self.assertTrue(any("generalist:" in error for error in check_artifacts(self.cfg)))
        index.write_text(json.dumps({"weight_map": {"wrong": "model.safetensors"}}))
        self.assertTrue(any("generalist:" in error for error in check_artifacts(self.cfg)))

    def test_missing_processor_fails(self):
        (self.root / "generalist/preprocessor_config.json").unlink()
        self.assertTrue(any("generalist:" in error for error in check_artifacts(self.cfg)))

    def test_old_and_combined_processor_metadata_are_compatible(self):
        directory = self.root / "generalist"
        image_config = json.loads((directory / "preprocessor_config.json").read_text())
        (directory / "preprocessor_config.json").unlink()
        (directory / "processor_config.json").write_text(json.dumps({"image_processor": image_config}))
        self.assertEqual(check_artifacts(self.cfg), [])
        image_config["patch_size"] = 32
        (directory / "processor_config.json").write_text(json.dumps({"image_processor": image_config}))
        self.assertTrue(any("geometry/normalization" in error for error in check_artifacts(self.cfg)))


if __name__ == "__main__":
    unittest.main()
