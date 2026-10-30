"""CPU regressions for excluding inputs incompatible with frozen preprocessing."""
from contextlib import contextmanager
import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "v3_image_eligibility_checks", ROOT / "data_pipeline/filter_perception_v3_parallel.py")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


class ImageSize(dict):
    """Expose the dictionary and attribute interfaces of transformers SizeDict."""
    __getattr__ = dict.__getitem__


def cpu_smart_resize(height, width, factor=28, min_pixels=3136, max_pixels=1003520):
    """Small dependency substitute; real HF tensor equality was checked separately."""
    h = round(height / factor) * factor
    w = round(width / factor) * factor
    if h * w > max_pixels:
        beta = math.sqrt(height * width / max_pixels)
        h = max(factor, math.floor(height / beta / factor) * factor)
        w = max(factor, math.floor(width / beta / factor) * factor)
    elif h * w < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h = math.ceil(height * beta / factor) * factor
        w = math.ceil(width * beta / factor) * factor
    return h, w


@contextmanager
def resize_dependency():
    name = "transformers.models.qwen2_vl.image_processing_qwen2_vl"
    module = ModuleType(name)
    module.smart_resize = Mock(side_effect=cpu_smart_resize)
    with patch.dict(sys.modules, {name: module}):
        yield module.smart_resize


class ImageEligibilityTests(unittest.TestCase):
    def fixture(self, sizes=((128, 192),), minimum=65536):
        record = {"id": "vqav2:142298002"}
        protocol = {"min_image_pixels": 4096, "max_image_pixels": 786432}
        processor = SimpleNamespace(image_processor=SimpleNamespace(
            size=ImageSize(shortest_edge=minimum, longest_edge=16777216),
            patch_size=16, temporal_patch_size=2, merge_size=2))
        prompt = {"prompt_token_ids": [10, 20, 30],
                  "multi_modal_data": {"image": [SimpleNamespace(size=size) for size in sizes]}}
        metadata = {"prompt_tokens": 108, "expected_prompt_token_ids": [10, 20, 20, 30],
                    "prompt_token_ids_sha256": "frozen-training-ids",
                    "first_resize_wh": [list(size) for size in sizes]}
        prepare = Mock(return_value=(prompt, metadata))
        return record, protocol, processor, prompt, metadata, prepare

    def evaluate(self, fixture):
        record, protocol, processor, _, _, prepare = fixture
        return DRIVER.compatible_training_prompt(record, protocol, processor, prepare=prepare)

    def test_failed_small_image_is_excluded_with_both_actual_grid_shapes(self):
        fixture = self.fixture()
        with resize_dependency():
            prompt, metadata = self.evaluate(fixture)
        self.assertIsNone(prompt)
        self.assertEqual(metadata["reason"], "training_inference_image_grid_mismatch")
        self.assertEqual(metadata["training_grid_thw"], [[1, 20, 14]])
        self.assertEqual(metadata["inference_grid_thw"], [[1, 12, 8]])
        self.assertEqual(metadata["first_resize_wh"], [[128, 192]])
        self.assertEqual(metadata["prompt_tokens"], 108)
        self.assertNotIn("expected_prompt_token_ids", metadata)
        self.assertNotIn("completions", metadata)
        self.assertNotIn("n_correct", metadata)
        fixture[-1].assert_called_once_with(*fixture[:3])

    def test_normal_image_preserves_prompt_and_metadata_without_changes(self):
        fixture = self.fixture(((640, 512),))
        with resize_dependency():
            prompt, metadata = self.evaluate(fixture)
        self.assertIs(prompt, fixture[3])
        self.assertIs(metadata, fixture[4])
        self.assertEqual(metadata["expected_prompt_token_ids"], [10, 20, 20, 30])

    def test_small_image_is_allowed_when_training_processor_uses_same_minimum(self):
        fixture = self.fixture(minimum=4096)
        with resize_dependency():
            prompt, metadata = self.evaluate(fixture)
        self.assertIs(prompt, fixture[3])
        self.assertIs(metadata, fixture[4])

    def test_existing_overbudget_rejection_is_forwarded_without_processing(self):
        metadata = {"prompt_tokens": 2050, "reason": "prompt_over_budget"}
        prepare = Mock(return_value=(None, metadata))
        with resize_dependency() as resize:
            result = DRIVER.compatible_training_prompt({}, {}, None, prepare=prepare)
        self.assertIsNone(result[0])
        self.assertIs(result[1], metadata)
        resize.assert_not_called()

    def test_all_multimodal_grids_are_checked_and_reported_in_image_order(self):
        fixture = self.fixture(((640, 512), (128, 192), (512, 640)))
        with resize_dependency():
            prompt, metadata = self.evaluate(fixture)
        self.assertIsNone(prompt)
        self.assertEqual(metadata["training_grid_thw"], [[1, 32, 40], [1, 20, 14], [1, 40, 32]])
        self.assertEqual(metadata["inference_grid_thw"], [[1, 32, 40], [1, 12, 8], [1, 40, 32]])

    def test_equal_token_counts_cannot_hide_different_grid_dimensions(self):
        fixture = self.fixture()
        with patch.object(DRIVER, "_image_grid", side_effect=[[1, 12, 8], [1, 8, 12]]):
            prompt, metadata = self.evaluate(fixture)
        self.assertIsNone(prompt)
        self.assertEqual(metadata["reason"], "training_inference_image_grid_mismatch")
        self.assertEqual(metadata["training_grid_thw"], [[1, 12, 8]])
        self.assertEqual(metadata["inference_grid_thw"], [[1, 8, 12]])

    def test_prepare_only_installs_guard_and_restores_it_even_after_failure(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                fixture = self.fixture()
                base_prepare = fixture[-1]
                def core_main():
                    self.assertIsNot(DRIVER.core.prepare_prompt, base_prepare)
                    prompt, metadata = DRIVER.core.prepare_prompt(*fixture[:3])
                    self.assertIsNone(prompt)
                    self.assertEqual(metadata["reason"], "training_inference_image_grid_mismatch")
                    if fail:
                        raise RuntimeError("CPU preflight failed")
                with patch.object(sys, "argv", [DRIVER.__file__, "--prepare-only"]), \
                     patch.dict(os.environ, {}, clear=False), \
                     patch.dict(sys.modules, {"vllm": None}), \
                     patch.object(DRIVER.core, "prepare_prompt", base_prepare), \
                     patch.object(DRIVER.core, "main", side_effect=core_main), \
                     patch.object(DRIVER, "ParallelLLM") as backend, resize_dependency():
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "CPU preflight failed"):
                            DRIVER.main()
                    else:
                        DRIVER.main()
                    self.assertIs(DRIVER.core.prepare_prompt, base_prepare)
                    backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
