"""Guard the expensive training input's quotas, annotation provenance and routing."""
import importlib.util
import json
from collections import Counter
from pathlib import Path
import random
import unittest

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("perception_v2_data", ROOT / "data_pipeline/prepare_perception_v2.py")
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


class SelectionRules(unittest.TestCase):
    def vqa(self, question, answer="yes", votes=10):
        return {"source": "vqav2", "question": question,
                "answers": [answer] * votes + ["disagreement"] * (10 - votes)}

    def test_rejects_inference_and_false_existence(self):
        for question in ["Is there enough room for more people?", "Can you see through the window?",
                         "Are there three birds in the picture?", "Why is the man on the left smiling?",
                         "Are both people of the same ethnic background?", "Can someone walk there in under an hour?"]:
            self.assertIsNone(DATA.classify(self.vqa(question)), question)

    def test_preserves_actual_presence_and_absence(self):
        for answer in ("yes", "no"):
            self.assertEqual(DATA.classify(self.vqa("Is there a dog in this picture?", answer))[0],
                             "existence_confusion")
        self.assertIsNone(DATA.classify(self.vqa("Is there a dog?", "yes", 7)))

    def test_document_direct_fields_only(self):
        for question in ["How many names are listed?", "What is the difference between the rates?",
                         "Which country would have similar interest as Europe?"]:
            self.assertIsNone(DATA.classify({"source": "docvqa", "question": question}))
        self.assertEqual(DATA.classify({"source": "docvqa", "question": "What is the area code?"})[0],
                         "ocr_document")

    def test_insufficient_unique_pool_never_cycles(self):
        with self.assertRaisesRegex(RuntimeError, "no repeats allowed"):
            DATA.take_unique([{"id": "only"}], 2, random.Random(0), "fixture")


class PreparedArtifact(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prepared = ROOT / "data_pipeline/prepared"
        path = cls.prepared / "perception_v2.parquet"
        if not path.exists():
            raise unittest.SkipTest("Build perception_v2 first")
        cls.rows = pq.read_table(path).to_pylist()
        cls.records = DATA.read_jsonl(cls.prepared / "perception_v2.records.jsonl")

    def test_exact_blocks_and_unique_records(self):
        self.assertEqual(len(self.rows), 3000)
        self.assertEqual(len({r["extra_info"]["record_id"] for r in self.rows}), 3000)
        self.assertEqual(len({(tuple(r["image_sha256"]), r["question_sha256"])
                              for r in self.records}), 3000)
        for size in (20, 60, 3000):
            expected = {key: size * weight // 100 for key, weight in DATA.WEIGHTS.items()}
            for offset in range(0, 3000, size):
                self.assertEqual(dict(Counter(r["extra_info"]["perception_category"]
                                              for r in self.rows[offset:offset + size])), expected)

    def test_preserves_original_labels_and_full_reward(self):
        originals = {r["id"]: r for source in DATA.SOURCES
                     for r in DATA.read_jsonl(DATA.ROOT / "normalized" / (source + ".jsonl"))}
        for row, selected in zip(self.rows, self.records, strict=True):
            original = originals[selected["id"]]
            self.assertEqual({k: selected[k] for k in original}, original)
            self.assertEqual(json.loads(row["reward_model"]["ground_truth"]), DATA.reward_payload(original))
            self.assertEqual(row["ability"], "perception_v2")
        self.assertEqual(DATA.validate_references(self.records)["score_histogram"], {"1.0": 3000})

    def test_current_evaluation_exclusion_and_presence_balance(self):
        blocks, _ = DATA.evaluation_blocks(sorted((ROOT / "evaluation").glob("suite*.jsonl")))
        for record in self.records:
            self.assertFalse(blocks["images"].intersection(record["image_sha256"]))
            self.assertNotIn(record["question_sha256"], blocks["questions"])
            if record.get("image_id"):
                self.assertNotIn(int(record["image_id"]), blocks["coco_ids"])
        labels = Counter(DATA.consensus(r)[0] for r in self.records
                         if r["perception_category"] == "existence_confusion")
        self.assertEqual(labels, {"yes": 225, "no": 225})

    def test_smoke_has_six_categories(self):
        rows = pq.read_table(self.prepared / "perception_v2_smoke.parquet").to_pylist()
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["extra_info"]["perception_category"] for r in rows}, set(DATA.WEIGHTS))


if __name__ == "__main__":
    unittest.main()
