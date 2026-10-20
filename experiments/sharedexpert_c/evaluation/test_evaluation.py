"""CPU-only tests for scoring failures that would bias teacher comparisons."""
import unittest
import json
import tempfile
from pathlib import Path
from common import final_answer, score, score_prediction, digest, PROTOCOL, image_size_for_count
from run_checkpoint import completed_cache


class ScoringTests(unittest.TestCase):
    def test_multimage_budget_preserves_single_and_two_image_limits(self):
        self.assertEqual(image_size_for_count(1)["longest_edge"], 786432)
        self.assertEqual(image_size_for_count(2)["longest_edge"], 786432)
        self.assertEqual(image_size_for_count(5)["longest_edge"], 314572)
        for count in range(1, 9):
            self.assertLessEqual(count * image_size_for_count(count)["longest_edge"], 1572864)
            self.assertEqual(image_size_for_count(count)["shortest_edge"], 4096)

    def test_reasoning_is_not_an_answer(self):
        row = {"task": "mcq", "answer": "B", "choices": {"A": "cat", "B": "dog"}}
        self.assertFalse(score(row, "<think>Maybe B\nB")["correct"])
        self.assertFalse(score(row, "<think>B</think>\n<answer>A</answer>")["correct"])
        self.assertTrue(score(row, "<think>A seems plausible.</think>\n<answer>B</answer>")["correct"])

    def test_mcq_cannot_match_gold_mentioned_in_explanation(self):
        row = {"task": "mcq", "answer": "B", "choices": {"A": "cat", "B": "dog"}}
        self.assertFalse(score(row, "</think><answer>A. not B</answer>")["correct"])
        self.assertFalse(score(row, "</think><answer>A or B</answer>")["parse_success"])
        self.assertTrue(score(row, "</think><answer>dog</answer>")["correct"])

    def test_fraction_and_balanced_box(self):
        row = {"task": "exact", "answer": "0.5"}
        self.assertTrue(score(row, r"</think>\boxed{\frac{1}{2}}")["correct"])
        self.assertFalse(score(row, "</think><answer>0.51</answer>")["correct"])

    def test_grounding_range_and_iou(self):
        row = {"task": "grounding", "answer": "[0,0,100,100]", "bbox_xyxy_1000": [0, 0, 100, 100]}
        self.assertTrue(score(row, "</think><answer>[0,0,100,100]</answer>")["correct"])
        self.assertFalse(score(row, "</think><answer>[-10,0,100,100]</answer>")["parse_success"])
        self.assertFalse(score(row, "</think><answer>[100,100,0,0]</answer>")["parse_success"])
        self.assertFalse(score(row, "</think><answer>[false,false,100,100]</answer>")["parse_success"])

    def test_yes_no_word_boundary(self):
        row = {"task": "yesno", "answer": "No"}
        self.assertTrue(score(row, "</think><answer>No.</answer>")["correct"])
        self.assertFalse(score(row, "</think><answer>Nothing here.</answer>")["parse_success"])

    def test_length_penalty_applies_even_after_correct_final(self):
        row = {"task": "exact", "answer": "2"}
        raw = {"raw_response": "</think><answer>2</answer>", "finish_reason": "length", "error": None}
        inference_hash = digest(PROTOCOL)
        scored = score_prediction(row, raw)
        self.assertFalse(scored["correct"])
        self.assertTrue(scored["correct_before_length_penalty"])
        self.assertTrue(scored["parse_success"])
        self.assertTrue(scored["length_terminated"])
        self.assertEqual(scored["raw_response"], raw["raw_response"])
        self.assertEqual(digest(PROTOCOL), inference_hash)
        raw["finish_reason"] = "stop"
        self.assertTrue(score_prediction(row, raw)["correct"])

    def test_cached_old_true_is_rescored_without_generation(self):
        row = {"task": "grounding", "answer": "[0,0,100,100]", "bbox_xyxy_1000": [0, 0, 100, 100]}
        raw = {"raw_response": "</think><answer>[0,0,100,100]</answer>", "finish_reason": "length", "error": None, "correct": True, "iou": 1.0}
        scored = score_prediction(row, raw)
        self.assertFalse(scored["correct"])
        self.assertEqual(scored["iou"], 0.0)
        self.assertEqual(scored["iou_before_length_penalty"], 1.0)
        self.assertTrue(raw["correct"], "Raw cache must remain untouched")


class CheckpointResumeTests(unittest.TestCase):
    def test_only_exact_successful_checkpoint_cache_can_skip_inference(self):
        rows = [{"id": "test:1", "answer": "2"}]
        suite_meta = {"suite_sha256": "frozen-suite", "protocol_sha256": digest(PROTOCOL)}
        provenance = {"path": "/checkpoint-a", "weight_files": [{"name": "model.safetensors", "size": 123, "mtime_ns": 1}]}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out / "checkpoint.json").write_text(json.dumps(provenance))
            (out / "run.json").write_text(json.dumps({**suite_meta, "protocol": PROTOCOL,
                "name": "perception", "served_model": "sharedexpert-perception", "n_requested": 1, "smoke": False}))
            record = {"id": "test:1", "row_sha256": digest(rows[0]), "raw_response": "</think><answer>2</answer>", "error": None}
            def check():
                (out / "predictions.jsonl").write_text(json.dumps(record)+"\n")
                return completed_cache(out, rows, suite_meta, provenance, "perception", "sharedexpert-perception")
            self.assertTrue(check())
            record["error"] = "Connection refused"
            self.assertFalse(check())
            record["error"] = None
            record["row_sha256"] = "changed-question"
            self.assertFalse(check())
            record["row_sha256"] = digest(rows[0])
            (out / "checkpoint.json").write_text(json.dumps({**provenance, "path": "/different-checkpoint"}))
            with self.assertRaises(SystemExit):
                check()


if __name__ == "__main__":
    unittest.main()
