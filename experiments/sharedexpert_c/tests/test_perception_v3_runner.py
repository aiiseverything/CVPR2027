"""CPU regression checks for screening resume, input identity and publication."""
from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "perception_v3_runner", ROOT / "data_pipeline/filter_perception_v3.py")
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)
import perception_v3_filtering as FILTER
import prepare_perception_v2 as V2


PROTOCOL = {"thinking": True, "prompt_length": 2048, "min_image_pixels": 4096,
            "max_image_pixels": 786432, "max_total_image_pixels": 1572864}


def make_record(category, index):
    answer = "yes" if index % 2 == 0 else "no"
    return {"id": f"{category}:{index}", "source": "vqav2", "source_id": str(index),
            "split": "train", "question": f"Fixture {category} {index}?",
            "question_sha256": f"question-{category}-{index}",
            "images": ["/fixture/image.jpg"], "image_sha256": ["pixels"],
            "task": "vqa", "answers": [answer] * 10,
            "perception_category": category, "selection_rule": "fixture"}


def accepted_fixture():
    records, groups = [], {}
    for category, weight in FILTER.WEIGHTS.items():
        for index in range(40 * weight // 100):
            row = make_record(category, index)
            answer = row["answers"][0]
            other = "no" if answer == "yes" else "yes"
            completions = [{"index": j, "text": "</think>\\boxed{" +
                            (answer if j < 4 else other) + "}", "finish_reason": "stop"}
                           for j in range(8)]
            groups[row["id"]] = {"id": row["id"], "completions": completions,
                                  "difficulty_filter": FILTER.accept_group(row, completions)}
            records.append(row)
    return records, groups


class Journal(unittest.TestCase):
    def test_truncates_only_uncommitted_tail_and_preserves_complete_group(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.jsonl"
            committed = b'{"id":"complete","value":1}\n'
            path.write_bytes(committed + b'{"id":"partial"')
            self.assertEqual(RUNNER.read_journal(path), {"complete": {"id": "complete", "value": 1}})
            self.assertEqual(path.read_bytes(), committed)
            self.assertEqual(RUNNER.read_journal(path)["complete"]["value"], 1)

    def test_malformed_committed_line_is_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.jsonl"
            path.write_bytes(b'{"id":"first"}\nnot-json\n')
            with self.assertRaises(json.JSONDecodeError):
                RUNNER.read_journal(path)

    def test_duplicate_committed_group_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "groups.jsonl"
            path.write_text('{"id":"same"}\n{"id":"same"}\n')
            with self.assertRaisesRegex(ValueError, "Duplicate committed group"):
                RUNNER.read_journal(path)

    def test_seed_does_not_depend_on_candidate_position(self):
        ids = ["a", "b", "c"]
        forward = {identity: RUNNER.record_seed(identity, 19) for identity in ids}
        reverse = {identity: RUNNER.record_seed(identity, 19) for identity in reversed(ids)}
        self.assertEqual(forward, reverse)
        self.assertEqual(len(set(forward.values())), 3)
        self.assertNotEqual(forward["a"], RUNNER.record_seed("a", 20))


class PromptPreparation(unittest.TestCase):
    def processor(self, ids):
        class Qwen2VLImageProcessorFixture:
            patch_size = 16

        class TensorFixture:
            def squeeze(self, axis):
                return self

            def tolist(self):
                return ids

        class ProcessorFixture:
            image_processor = Qwen2VLImageProcessorFixture()
            image_token_id, video_token_id = 900, 901

            def apply_chat_template(self, messages, **kwargs):
                self.messages, self.template_kwargs = messages, kwargs
                return "rendered-prompt"

            def __call__(self, **kwargs):
                self.processor_kwargs = kwargs
                return {"input_ids": TensorFixture()}

        return ProcessorFixture()

    def test_training_template_and_full_token_identity_survive_visual_dedup(self):
        processor = self.processor([10, 900, 900, 900, 11])
        image = SimpleNamespace(size=(320, 240))
        vision = SimpleNamespace(process_vision_info=lambda messages, image_patch_size: ([image], None))
        with patch.dict(sys.modules, {"qwen_vl_utils": vision}), patch.object(V2, "pixel_hash", return_value="pixels"):
            prompt, audit = RUNNER.prepare_prompt(make_record("object_attribute", 0), PROTOCOL, processor)
        self.assertEqual(prompt["prompt_token_ids"], [10, 900, 11])
        self.assertEqual(audit["expected_prompt_token_ids"], [10, 900, 900, 900, 11])
        self.assertEqual(audit["prompt_tokens"], 5)
        self.assertEqual(processor.template_kwargs,
                         {"tools": [], "add_generation_prompt": True, "tokenize": False, "enable_thinking": True})
        self.assertEqual(processor.processor_kwargs["images"], [image])
        self.assertFalse(processor.processor_kwargs["do_sample_frames"])

    def test_over_budget_input_is_rejected_instead_of_text_truncated(self):
        processor = self.processor([1] * 2049)
        vision = SimpleNamespace(process_vision_info=lambda messages, image_patch_size: ([], None))
        with patch.dict(sys.modules, {"qwen_vl_utils": vision}), patch.object(V2, "pixel_hash", return_value="pixels"):
            prompt, audit = RUNNER.prepare_prompt(make_record("object_attribute", 0), PROTOCOL, processor)
        self.assertIsNone(prompt)
        self.assertEqual(audit, {"prompt_tokens": 2049, "reason": "prompt_over_budget"})

    def test_changed_image_fails_before_inference(self):
        vision = SimpleNamespace(process_vision_info=lambda *args, **kwargs: ([], None))
        with patch.dict(sys.modules, {"qwen_vl_utils": vision}), patch.object(V2, "pixel_hash", return_value="changed"):
            with self.assertRaisesRegex(ValueError, "Candidate image changed"):
                RUNNER.prepare_prompt(make_record("object_attribute", 0), PROTOCOL, self.processor([1]))


class Publication(unittest.TestCase):
    def test_incomplete_quota_does_not_publish(self):
        records, groups = accepted_fixture()
        groups.pop("grounding:0")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            self.assertFalse(RUNNER.publish(records, groups, output, PROTOCOL, "run", 40, 19, []))
            self.assertFalse((output / "dataset").exists())

    def test_all_exported_rows_have_verified_receipts_and_exact_blocks(self):
        import pyarrow.parquet as pq
        records, groups = accepted_fixture()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "groups.jsonl").write_text("".join(json.dumps(group) + "\n" for group in groups.values()))
            self.assertTrue(RUNNER.publish(records, groups, output, PROTOCOL, "run", 40, 19, []))
            manifest = json.loads((output / "dataset/manifest.json").read_text())
            rows = pq.read_table(output / "dataset/perception_v3.parquet").to_pylist()
            originals = [json.loads(line) for line in (output / "dataset/perception_v3.records.jsonl").read_text().splitlines()]
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["parquet_sha256"], RUNNER.file_sha(output / "dataset/perception_v3.parquet"))
            for row, original in zip(rows, originals, strict=True):
                self.assertEqual(row["extra_info"]["record_id"], original["id"])
                self.assertEqual(row["extra_info"]["screening_n_correct"], 4)
                self.assertEqual(row["extra_info"]["screening_n_completions"], 8)
                self.assertEqual(json.loads(row["reward_model"]["ground_truth"]), V2.reward_payload(original))
            for start in (0, 20):
                self.assertEqual(Counter(row["extra_info"]["perception_category"] for row in rows[start:start + 20]),
                                 {key: weight // 5 for key, weight in FILTER.WEIGHTS.items()})

    def test_tampered_accepted_candidate_fails_publication(self):
        records, groups = accepted_fixture()
        records[0]["question"] = "Changed after screening"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "screening receipt"):
                RUNNER.publish(records, groups, Path(directory), PROTOCOL, "run", 40, 19, [])


if __name__ == "__main__":
    unittest.main()
