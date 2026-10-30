"""CPU-only regressions for the real screening loop using mocked inference."""
from contextlib import ExitStack, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("v3_runner_main_regression", ROOT / "data_pipeline/filter_perception_v3.py")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class MainLoopRegression(unittest.TestCase):
    def exercise(self, output, completion_count=8, over_budget=False, duplicate_index=False):
        protocol_path = output / "protocol.json"
        protocol = json.loads((ROOT / "training/config/common_local_100.json").read_text())
        protocol_path.write_text(json.dumps(protocol))
        record = {
            "id": "vqav2:fixture", "source": "vqav2", "source_id": "fixture", "split": "train",
            "question": "What color is the car?", "question_sha256": "fixture-question-hash",
            "images": ["unused-mocked-image.png"], "image_sha256": ["fixture-image-hash"],
            "task": "vqa", "answers": ["red"] * 10,
            "perception_category": "object_attribute", "selection_rule": "color",
        }
        completions = [SimpleNamespace(index=(0 if duplicate_index and index == 7 else index),
                                      token_ids=[100 + index], finish_reason="stop")
                       for index in range(completion_count)]
        llm = Mock()
        llm.generate.return_value = [SimpleNamespace(prompt_token_ids=[42], outputs=completions)]
        llm_class = Mock(return_value=llm)
        processor = SimpleNamespace(
            image_processor=object(),
            tokenizer=SimpleNamespace(decode=lambda tokens, **kwargs:
                                      "</think>\\boxed{" + ("red" if tokens[0] < 104 else "blue") + "}"),
        )
        transformers = SimpleNamespace(AutoProcessor=SimpleNamespace(from_pretrained=Mock(return_value=processor)))
        vllm = SimpleNamespace(LLM=llm_class, SamplingParams=lambda **kwargs: SimpleNamespace(**kwargs))

        def prepare(*args):
            if over_budget:
                return None, {"prompt_tokens": 2049, "reason": "prompt_over_budget"}
            return {"prompt_token_ids": [42], "multi_modal_data": {"image": []}}, {
                "prompt_tokens": 1, "expected_prompt_token_ids": [42],
            }

        def fake_gpu_query(args, **kwargs):
            return "GPU-FAKE\n" if "--query-gpu=uuid" in args else ""

        evidence = [{"directory": str(output / "pool"), "manifest": {
            "protocol_sha256": DRIVER.file_sha(protocol_path),
        }}]
        arguments = ["filter_perception_v3.py", "--pool-dir", str(output / "pool"),
                     "--output-dir", str(output / "screen"), "--model", str(output / "model"),
                     "--protocol", str(protocol_path), "--target", "40", "--batch-prompts", "1", "--gpu", "99"]
        path_open = Path.open
        handles = []

        def tracked_open(path, *args, **kwargs):
            handle = path_open(path, *args, **kwargs)
            if path.name in {".lock", "mock-gpu.lock"}:
                handles.append(handle)
            return handle

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, "open", tracked_open))
            stack.callback(lambda: [handle.close() for handle in handles])
            stack.enter_context(redirect_stdout(io.StringIO()))
            stack.enter_context(patch.dict(os.environ, {}, clear=False))
            stack.enter_context(patch.dict(sys.modules, {"vllm": vllm, "transformers": transformers}))
            stack.enter_context(patch.object(sys, "argv", arguments))
            stack.enter_context(patch.object(DRIVER, "load_candidates", return_value=([record], evidence)))
            stack.enter_context(patch.object(DRIVER, "model_identity", return_value={"fixture.safetensors": {"sha256": "fixture", "bytes": 1}}))
            stack.enter_context(patch.object(DRIVER, "prepare_prompt", side_effect=prepare))
            stack.enter_context(patch.object(DRIVER, "publish", return_value=False))
            stack.enter_context(patch.object(DRIVER.importlib.metadata, "version", return_value="fixture-version"))
            stack.enter_context(patch.object(DRIVER.subprocess, "check_output", side_effect=fake_gpu_query))
            stack.enter_context(patch.object(DRIVER, "open", create=True,
                                             side_effect=lambda *args, **kwargs: (output / "mock-gpu.lock").open("a")))
            DRIVER.main()
        return llm, record

    def test_valid_batch_is_scored_before_receipt_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            llm, record = self.exercise(root)
            rows = [json.loads(line) for line in (root / "screen/groups.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            receipt = rows[0]
            self.assertEqual(receipt["id"], record["id"])
            self.assertEqual(receipt["difficulty_filter"]["n_correct"], 4)
            self.assertEqual(receipt["difficulty_filter"]["n_completions"], 8)
            self.assertTrue(receipt["difficulty_filter"]["accepted"])
            self.assertEqual(len(receipt["completions"]), 8)
            self.assertEqual(json.loads((root / "screen/status.json").read_text())["accepted"], 1)
            llm.generate.assert_called_once()

    def test_incomplete_or_duplicate_group_aborts_without_committing(self):
        for count, duplicate in ((7, False), (8, True)):
            with self.subTest(count=count, duplicate=duplicate), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                with self.assertRaisesRegex(RuntimeError, "Uncommitted incomplete group"):
                    self.exercise(root, completion_count=count, duplicate_index=duplicate)
                self.assertEqual((root / "screen/groups.jsonl").read_text(), "")

    def test_overbudget_input_has_rejection_without_model_answer_or_score(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            llm, record = self.exercise(root, over_budget=True)
            rows = [json.loads(line) for line in (root / "screen/groups.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertFalse(rows[0]["difficulty_filter"]["accepted"])
            self.assertEqual(rows[0]["difficulty_filter"]["rejection_reason"], "prompt_over_budget")
            self.assertNotIn("completions", rows[0])
            self.assertNotIn("n_correct", rows[0]["difficulty_filter"])
            self.assertEqual(json.loads((root / "screen/status.json").read_text())["accepted"], 0)
            llm.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
