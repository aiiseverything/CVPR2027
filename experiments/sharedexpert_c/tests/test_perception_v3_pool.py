"""Protect label fidelity, leakage checks, and incremental candidate identity."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("perception_v3_pool", ROOT / "data_pipeline/prepare_perception_v3_pool.py")
DATA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DATA)


class CandidateContracts(unittest.TestCase):
    def record(self, identity, category="object_attribute"):
        return {
            "id": identity, "source": "vqav2", "source_id": identity, "split": "train",
            "task": "vqa", "question": "What color is the car?",
            "question_sha256": identity + "-question", "images": ["/example/image.png"],
            "image_sha256": [identity + "-image"], "answers": ["red"] * 10,
            "perception_category": category, "selection_rule": "color",
        }

    def test_candidate_preserves_training_inputs_and_marks_unfiltered(self):
        protocol = json.loads((ROOT / "training/config/common_local_100.json").read_text())
        record = self.record("fixture")
        original = DATA.ORIGINAL.package(record, 4, protocol)
        candidate = DATA.package_candidate(record, 4, protocol)
        for key in ("prompt", "images", "data_source", "reward_model"):
            self.assertEqual(candidate[key], original[key])
        self.assertFalse(candidate["extra_info"]["difficulty_filtered"])
        self.assertEqual(candidate["extra_info"]["dataset_stage"], "candidate_pool")

    def test_no_records_lost_or_repeated_and_order_reproducible(self):
        pools = {key: [self.record(f"{key}-{index}", key) for index in range(count)]
                 for (key, count) in zip(DATA.ORIGINAL.WEIGHTS, (3, 29, 1, 7, 4, 2))}
        records = DATA.ordered_candidates(pools, 123)
        self.assertEqual(len(records), 46)
        self.assertEqual(len({row["id"] for row in records}), 46)
        self.assertEqual(records, DATA.ordered_candidates({key: list(reversed(rows)) for key, rows in pools.items()}, 123))
        pools["grounding"].append(pools["object_attribute"][0])
        with self.assertRaisesRegex(ValueError, "Repeated candidate"):
            DATA.ordered_candidates(pools, 123)

    def test_appending_pool_preserves_prefix_and_rejects_changed_gold(self):
        prior = [self.record("b"), self.record("a")]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            records = path / "candidates.records.jsonl"
            records.write_text("".join(json.dumps(row) + "\n" for row in prior))
            (path / "manifest.json").write_text(json.dumps({
                "protocol_sha256": "protocol", "records_sha256": DATA.ORIGINAL.sha256(records),
            }))
            combined = DATA.preserve_previous_prefix([self.record("c"), *reversed(prior)], path, "protocol")
            self.assertEqual([row["id"] for row in combined], ["b", "a", "c"])
            changed = [{**row, "answers": ["blue"]} if row["id"] == "a" else row for row in combined]
            with self.assertRaisesRegex(ValueError, "changed or is no longer eligible"):
                DATA.preserve_previous_prefix(changed, path, "protocol")

    def test_supplemental_sources_deduplicate_but_reject_conflicting_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            normalized = root / "normalized"
            normalized.mkdir()
            record = self.record("same-source-record")
            record["question_sha256"] = DATA.ORIGINAL.question_hash(record["question"])
            for source in DATA.ORIGINAL.SOURCES:
                (normalized / (source + ".jsonl")).write_text(
                    json.dumps(record) + "\n" if source == "vqav2" else "")
            extra = root / "extra.jsonl"
            extra.write_text(json.dumps(record) + "\n")
            blocks = {key: set() for key in ("images", "questions", "ids", "source_ids", "coco_ids")}
            with patch.object(DATA, "ROOT", root):
                pools, audit = DATA.make_pools(blocks, [extra])
                self.assertEqual(len(pools["object_attribute"]), 1)
                self.assertEqual(audit["counts"]["vqav2"]["duplicate_record_or_image_question_removed"], 1)
                extra.write_text(json.dumps({**record, "answers": ["blue"] * 10}) + "\n")
                with self.assertRaisesRegex(ValueError, "Conflicting labels"):
                    DATA.make_pools(blocks, [extra])

    def test_nested_sqlite_image_excludes_same_rgb_with_different_encoding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = Image.new("RGB", (20, 10), "red")
            candidate = root / "candidate.bmp"
            image.save(candidate)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            payload = buffer.getvalue()
            image_sha = hashlib.sha256(payload).hexdigest()
            suite = root / "suite.jsonl"
            suite.write_text(json.dumps({"id": "eval", "question": "other question", "image": {
                "sha256": image_sha, "width": 20, "height": 10,
            }}) + "\n")
            with sqlite3.connect(root / "images.sqlite") as connection:
                connection.execute("CREATE TABLE images (sha256 TEXT, data BLOB)")
                connection.execute("INSERT INTO images VALUES (?,?)", (image_sha, payload))
            record = self.record("candidate")
            record["images"] = [str(candidate)]
            blocks = {"images": set()}
            audit = DATA.add_nested_image_blocks(blocks, [suite], {"object_attribute": [record]})
            self.assertEqual(audit["decoded_images"], 1)
            self.assertIn(DATA.ORIGINAL.pixel_hash(str(candidate)), blocks["images"])
            with sqlite3.connect(root / "images.sqlite") as connection:
                connection.execute("UPDATE images SET data=?", (b"changed",))
            with self.assertRaisesRegex(ValueError, "payload missing or changed"):
                DATA.add_nested_image_blocks(blocks, [suite], {"object_attribute": [record]})


if __name__ == "__main__":
    unittest.main()
