"""CPU-only tests of the OPD scorer's sequence and vocabulary alignment."""

from types import SimpleNamespace
import unittest

import torch

from experiments.sharedexpert_c.distillation.teacher_scorer import FrozenTeacherScorer, _assert_compatible


class TinyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.head_sizes = []
        self.weight = torch.arange(35, dtype=torch.float32).reshape(5, 7) / 100

    def model(self, **kwargs):
        self.calls.append(kwargs)
        # Token- and position-dependent output detects off-by-one alignment.
        ids = kwargs["input_ids"]
        hidden = torch.nn.functional.one_hot(ids % 5, num_classes=5).float()
        return SimpleNamespace(last_hidden_state=hidden)

    def lm_head(self, hidden):
        self.head_sizes.append(len(hidden))
        return hidden @ self.weight


def scorer(temperature=1):
    instance = FrozenTeacherScorer.__new__(FrozenTeacherScorer)
    instance.device = torch.device("cpu")
    instance.logit_chunk_size = 1
    instance.temperature = temperature
    instance.teacher_paths = {"generalist": "unused"}
    instance.fake = TinyTeacher()
    instance._get_model = lambda name: instance.fake
    return instance


def sample():
    return {
        "input_ids": torch.tensor([0, 0, 3, 4, 5, 6, 0]),
        "attention_mask": torch.tensor([0, 0, 1, 1, 1, 1, 0]),
        "position_ids": torch.arange(7).repeat(4, 1) + 40,
        "responses": torch.tensor([5, 6, 0]),
        "response_mask": torch.tensor([1, 1, 0]),
        "multi_modal_inputs": {
            "pixel_values": torch.randn(4, 12),
            "image_grid_thw": torch.tensor([[1, 2, 2]]),
            "images_seqlens": torch.tensor([4]),
        },
    }


class TeacherScorerTests(unittest.TestCase):
    def test_real_qwen35_cpu_matches_full_logits(self):
        # Random tiny weights only: no checkpoint reads, network or CUDA work.
        # Installed FLA selects Triton even for CPU tensors, so this test alone
        # selects the stock HF torch fallbacks on the tiny model's instances.
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            torch_chunk_gated_delta_rule,
            torch_recurrent_gated_delta_rule,
        )

        cfg = Qwen3_5Config(
            text_config={
                "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 2, "num_attention_heads": 4,
                "num_key_value_heads": 2, "head_dim": 8,
                "layer_types": ["linear_attention", "full_attention"],
                "linear_num_key_heads": 2, "linear_num_value_heads": 2,
                "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                "rope_parameters": {
                    "rope_type": "default", "rope_theta": 10000,
                    "mrope_section": [1, 1, 2], "partial_rotary_factor": 1.0,
                },
            },
            vision_config={
                "hidden_size": 32, "intermediate_size": 64, "depth": 1,
                "num_heads": 4, "out_hidden_size": 32,
                "num_position_embeddings": 16, "patch_size": 2,
                "temporal_patch_size": 1, "spatial_merge_size": 2,
            },
            image_token_id=28, video_token_id=29,
            vision_start_token_id=30, vision_end_token_id=31,
        )
        # Preserve the calling suite's CPU RNG; never seed or initialize CUDA.
        state = torch.get_rng_state()
        try:
            torch.random.default_generator.manual_seed(19)
            model = Qwen3_5ForConditionalGeneration(cfg).cpu().eval().requires_grad_(False)
        finally:
            torch.set_rng_state(state)
        for layer in model.model.language_model.layers:
            if hasattr(layer, "linear_attn"):
                layer.linear_attn.chunk_gated_delta_rule = torch_chunk_gated_delta_rule
                layer.linear_attn.recurrent_gated_delta_rule = torch_recurrent_gated_delta_rule
                # If causal-conv1d is installed, do not call its CUDA kernels.
                layer.linear_attn.causal_conv1d_fn = None
                layer.linear_attn.causal_conv1d_update = None
        backend = scorer()
        backend._get_model = lambda name: model
        row = sample()
        row["multi_modal_inputs"] = {}
        with torch.inference_mode():
            actual = backend.score("generalist", [row])[0]
            logits = model(
                input_ids=row["input_ids"][None, 2:6],
                attention_mask=torch.ones((1, 4), dtype=torch.long),
                position_ids=row["position_ids"][:, None, 2:6],
                use_cache=False,
            ).logits
            expected = logits[0, 1:3].float().log_softmax(-1)[
                torch.arange(2), row["responses"][:2]
            ]
        torch.testing.assert_close(actual[:2], expected, rtol=0, atol=1e-6)
        self.assertEqual(actual[2].item(), 0.0)
        self.assertEqual(actual.device.type, "cpu")

    def test_probability_alignment_temperature_and_outer_padding(self):
        for temperature in (1, 0.7):
            backend = scorer(temperature)
            row = sample()
            actual = backend.score("generalist", [row])[0]
            logits = backend.fake.weight[torch.tensor([4, 0])] / temperature
            expected = logits.log_softmax(-1)[torch.arange(2), torch.tensor([5, 6])]
            torch.testing.assert_close(actual[:2], expected)
            self.assertEqual(actual[2].item(), 0)
            self.assertEqual(actual.dtype, torch.float32)
            self.assertEqual(actual.device.type, "cpu")
            self.assertFalse(actual.requires_grad)
            forward = backend.fake.calls[0]
            torch.testing.assert_close(forward["input_ids"], torch.tensor([[3, 4, 5, 6]]))
            torch.testing.assert_close(forward["position_ids"], row["position_ids"][:, None, 2:6])
            torch.testing.assert_close(forward["pixel_values"], row["multi_modal_inputs"]["pixel_values"])
            self.assertNotIn("images_seqlens", forward)
            self.assertFalse(forward["use_cache"])
            self.assertEqual(backend.fake.head_sizes, [1, 1])

    def test_interior_response_mask_preserves_causal_context(self):
        backend = scorer()
        row = sample()
        row["response_mask"] = torch.tensor([0, 1, 0])
        actual = backend.score("generalist", [row])[0]
        self.assertEqual(actual[0].item(), 0)
        expected = backend.fake.weight[0].log_softmax(-1)[6]
        torch.testing.assert_close(actual[1], expected)
        torch.testing.assert_close(backend.fake.calls[0]["input_ids"], torch.tensor([[3, 4, 5, 6]]))

    def test_reject_retokenized_responses(self):
        row = sample()
        row["responses"][0] = 2
        with self.assertRaisesRegex(ValueError, "exact input_ids suffix"):
            scorer().score("generalist", [row])

    def test_reject_attention_holes(self):
        row = sample()
        row["input_ids"] = torch.tensor([0, 2, 3, 4, 5, 6, 0])
        row["attention_mask"] = torch.tensor([0, 1, 0, 1, 1, 1, 0])
        with self.assertRaisesRegex(ValueError, "interior gaps"):
            scorer().score("generalist", [row])

    def test_reject_nonfinite_teacher(self):
        backend = scorer()
        backend.fake.weight[:] = float("nan")
        with self.assertRaises(FloatingPointError):
            backend.score("generalist", [sample()])

    def test_no_response_tokens_does_not_run_forward(self):
        backend = scorer()
        row = sample()
        row["response_mask"].zero_()
        torch.testing.assert_close(backend.score("generalist", [row])[0], torch.zeros(3))
        self.assertEqual(backend.fake.calls, [])

    def test_text_positions_keep_original_values(self):
        backend = scorer()
        row = sample()
        row["position_ids"] = torch.arange(7) + 40
        backend.score("generalist", [row])
        torch.testing.assert_close(backend.fake.calls[0]["position_ids"], torch.tensor([[42, 43, 44, 45]]))

    def test_compatibility_checks_vocabulary_and_patch_geometry(self):
        def config():
            return SimpleNamespace(
                model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"],
                text_config=SimpleNamespace(vocab_size=7), image_token_id=5, video_token_id=6,
                vision_start_token_id=3, vision_end_token_id=4,
                vision_config=SimpleNamespace(in_channels=3, patch_size=16,
                                              temporal_patch_size=2, spatial_merge_size=2),
            )
        tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1}, eos_token_id=1)
        other_tokenizer = SimpleNamespace(get_vocab=lambda: {"a": 1, "b": 0}, eos_token_id=1)
        _assert_compatible(config(), config(), tokenizer, tokenizer, "teacher")
        with self.assertRaisesRegex(ValueError, "vocabularies"):
            _assert_compatible(config(), config(), tokenizer, other_tokenizer, "teacher")
        incompatible = config()
        incompatible.vision_config.patch_size = 14
        with self.assertRaisesRegex(ValueError, "patch_size"):
            _assert_compatible(config(), incompatible, tokenizer, tokenizer, "teacher")

    def test_compatibility_allows_only_appended_specials_in_reserved_rows(self):
        cfg = SimpleNamespace(
            model_type="qwen3_5", architectures=["Qwen3_5ForConditionalGeneration"],
            text_config=SimpleNamespace(vocab_size=7),
            vision_config=SimpleNamespace(),
        )
        student = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1})
        exported = SimpleNamespace(get_vocab=lambda: {"a": 0, "b": 1, "<audio>": 2},
                                   all_special_tokens=["<audio>"])
        _assert_compatible(cfg, cfg, student, exported, "teacher")
        cases = [
            ({"a": 0, "b": 1, "ordinary": 2}, []),  # ordinary added token
            ({"a": 0, "b": 1, "<audio>": 7}, ["<audio>"]),  # outside head
            ({"a": 0, "b": 1, "<audio>": 1}, ["<audio>"]),  # reused ID
            ({"a": 0, "b": 2, "<audio>": 3}, ["<audio>"]),  # remapping
            ({"a": 0, "<audio>": 2}, ["<audio>"]),  # removed base token
        ]
        for vocab, specials in cases:
            with self.subTest(vocab=vocab):
                invalid = SimpleNamespace(get_vocab=lambda: vocab, all_special_tokens=specials)
                with self.assertRaisesRegex(ValueError, "incompatible tokenizer"):
                    _assert_compatible(cfg, cfg, student, invalid, "teacher")


if __name__ == "__main__":
    unittest.main()
