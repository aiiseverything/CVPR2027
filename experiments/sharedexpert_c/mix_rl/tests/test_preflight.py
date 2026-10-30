"""Catch comparison drift, invalid reward references and data corruption on CPU."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from experiments.sharedexpert_c.mix_rl import preflight


class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / "data.parquet"
        self.opd = self.root / "opd.json"
        self.config = dict(
            student_path=str(self.root / "student"), data_path=str(self.data),
            opd_config_path=str(self.opd), steps=1, batch_size=3, rollout_n=8,
            prompt_length=2048, response_length=8192, learning_rate=1e-6,
            temperature=1.0, top_p=1.0, top_k=-1,
        )
        opd = dict(self.config, algorithm="domain_opd", rollout_n=1)
        self.opd.write_text(json.dumps(opd))
        self.rows = [self.row(index) for index in range(3)]
        self.write_data()

    @staticmethod
    def row(index, gold=None):
        return dict(
            prompt=[{"role": "user", "content": "Count objects. Give a boxed answer."}],
            images=[], data_source="sharedexpert_test", domain=preflight.DOMAINS[index % 3],
            opd_index=index,
            reward_model={"style": "rule", "ground_truth": json.dumps(
                gold or {"task": "exact", "answers": ["2"], "format": "boxed"})},
            extra_info={"record_id": f"test:{index}", "split": "train"},
        )

    def write_data(self):
        pq.write_table(pa.Table.from_pylist(self.rows), self.data)
        self.config["expected_data_sha256"] = preflight._sha256(self.data)

    def check(self):
        with patch.object(preflight, "check_student", return_value={"test": "metadata validated"}):
            return preflight.check_inputs(self.config)

    def test_valid_comparison_has_no_teacher_dependency(self):
        report = self.check()
        self.assertEqual(report["errors"], [])
        self.assertFalse(report["comparison"]["teacher_checkpoints_required"])
        self.assertEqual(report["comparison"]["mix_rl_responses"], 24)
        self.assertEqual(report["comparison"]["opd_responses"], 3)
        self.assertEqual(report["reward_validation"]["audited_rows"], 3)

    def test_changed_learning_rate_or_model_fails_comparison(self):
        for field, changed in (("learning_rate", 2e-6), ("student_path", "/missing/another_student")):
            with self.subTest(field=field):
                config = dict(self.config, **{field: changed})
                self.assertTrue(any(field in e for e in preflight.check_comparison(config)["errors"]))

    def test_data_digest_mismatch_fails_even_valid_schema(self):
        self.config["expected_data_sha256"] = "0" * 64
        self.assertTrue(any("SHA-256" in e for e in self.check()["errors"]))

    def test_missing_digest_is_not_silently_accepted(self):
        self.config.pop("expected_data_sha256")
        self.assertTrue(any("expected_data_sha256" in e for e in self.check()["errors"]))

    def test_prompt_reordering_fails_even_after_digest_updated(self):
        self.rows.reverse()
        self.write_data()
        self.assertTrue(any("domain order" in e for e in self.check()["errors"]))

    def test_duplicate_record_rejected(self):
        self.rows[1]["extra_info"]["record_id"] = self.rows[0]["extra_info"]["record_id"]
        self.write_data()
        self.assertTrue(any("duplicate record_id" in e for e in self.check()["errors"]))

    def test_incomplete_student_metadata_fails(self):
        report = preflight.check_inputs(self.config)
        self.assertTrue(any(e.startswith("student:") for e in report["errors"]))

    def test_invalid_json_content_task_rejected(self):
        row = self.row(0, {"task": "json", "content_task": "unknown", "format": "json", "answers": ["2"]})
        with self.assertRaisesRegex(ValueError, "content_task"):
            preflight._reference_score(row)

    def test_reversed_bbox_rejected(self):
        row = self.row(0, {"task": "bbox", "format": "boxed", "bbox": [10, 10, 0, 0]})
        with self.assertRaisesRegex(ValueError, "bbox"):
            preflight._reference_score(row)

    def test_all_seven_gold_schemas_receive_reward(self):
        cases = [
            {"task": "bbox", "format": "boxed", "bbox": [10, 10, 20, 20]},
            {"task": "vqa", "format": "boxed", "answers": [str(i) for i in range(10)]},
            {"task": "math", "format": "boxed", "answers": [r"\boxed{\frac{1}{2}}"]},
            {"task": "anls", "format": "boxed", "answers": ["hello"]},
            {"task": "relaxed_numeric", "format": "boxed", "answers": ["123"]},
            {"task": "exact", "format": "boxed", "answers": ["2"]},
            {"task": "json", "content_task": "vqa", "format": "json", "answers": ["cat"] * 10},
        ]
        for gold in cases:
            with self.subTest(task=gold["task"]):
                task, score = preflight._reference_score(self.row(0, gold))
                self.assertEqual(task, gold["task"])
                self.assertGreater(score, 0)
                self.assertLessEqual(score, 1)
                if task == "vqa":
                    self.assertAlmostEqual(score, .3)


if __name__ == "__main__":
    unittest.main()
