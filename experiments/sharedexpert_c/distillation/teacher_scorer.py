"""Frozen, local HF teachers scoring the student's *actual* rollout tensors.

This module deliberately does not render chat templates, decode responses, or
process images. Inputs must be copied from the same VERL DataProto used by the
actor. Dense Qwen3.5 is the only architecture verified by this adapter.
"""

from collections import OrderedDict
import gc
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import validate_vocab_compatibility


TEACHER_NAMES = frozenset({"generalist", "perception", "reasoning"})
_VISUAL_FIELDS = ("image_token_id", "video_token_id", "vision_start_token_id", "vision_end_token_id")
_PATCH_FIELDS = ("in_channels", "patch_size", "temporal_patch_size", "spatial_merge_size")


def _assert_compatible(student_config, teacher_config, student_tokenizer, teacher_tokenizer, name):
    """Fail before loading weights when teacher likelihoods cannot align."""
    for label, model_config in (("student", student_config), (name, teacher_config)):
        if model_config.model_type != "qwen3_5" or list(model_config.architectures or []) != [
            "Qwen3_5ForConditionalGeneration"
        ]:
            raise ValueError(f"{label}: only dense Qwen3_5ForConditionalGeneration is supported")
    if student_config.text_config.vocab_size != teacher_config.text_config.vocab_size:
        raise ValueError(f"{name}: model output vocabulary size differs from student")
    # Exported Qwen3.5 processors may declare additional audio/tts specials in
    # already reserved output rows. The student remains the authority for all
    # sampled IDs; permit only strictly appended special-token declarations.
    errors = validate_vocab_compatibility(
        student_tokenizer.get_vocab(), teacher_tokenizer.get_vocab(),
        getattr(teacher_tokenizer, "all_special_tokens", []), student_config.text_config.vocab_size,
    )
    if errors:
        raise ValueError(f"{name}: incompatible tokenizer vocabularies: {'; '.join(errors)}")
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if getattr(student_tokenizer, key, None) != getattr(teacher_tokenizer, key, None):
            raise ValueError(f"{name}: tokenizer {key} differs from student")
    for key in _VISUAL_FIELDS:
        if getattr(student_config, key, None) != getattr(teacher_config, key, None):
            raise ValueError(f"{name}: visual token configuration {key} differs from student")
    for key in _PATCH_FIELDS:
        if getattr(student_config.vision_config, key, None) != getattr(teacher_config.vision_config, key, None):
            raise ValueError(f"{name}: image patch configuration {key} differs from student")


class FrozenTeacherScorer:
    """One actor-local, bounded pool of frozen teachers (default: one resident).

    Config keys: ``student_path``, ``teacher_paths`` mapping the three teacher
    names to standalone HF checkpoints, and ``teacher_backend`` containing
    ``device``, ``dtype``, ``max_resident_teachers`` and ``logit_chunk_size``.
    This class does not reserve Ray resources: its owner must reserve a GPU.
    Scores are normalized log probabilities at ``temperature`` (default 1),
    which must match the actor's likelihood temperature, never raw logits.
    """

    def __init__(self, config: Mapping[str, Any]):
        from transformers import AutoConfig, AutoTokenizer

        backend = config.get("teacher_backend", {})
        self.device = torch.device(backend.get("device", "cuda"))
        dtype_name = backend.get("dtype", "bfloat16")
        if dtype_name not in {"bfloat16", "float16", "float32"}:
            raise ValueError("teacher_backend.dtype must be bfloat16, float16 or float32")
        self.dtype = getattr(torch, dtype_name)
        self.max_resident = int(backend.get("max_resident_teachers", 1))
        self.logit_chunk_size = int(backend.get("logit_chunk_size", 128))
        self.temperature = float(config.get("temperature", 1.0))
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if not 1 <= self.max_resident <= 3 or self.logit_chunk_size < 1:
            raise ValueError("max_resident_teachers must be 1..3 and logit_chunk_size must be positive")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("FrozenTeacherScorer requires its reserved CUDA device")
        self.teacher_paths = dict(config["teacher_paths"])
        if set(self.teacher_paths) != TEACHER_NAMES:
            raise ValueError(f"teacher_paths must contain exactly {sorted(TEACHER_NAMES)}")
        paths = {"student": config["student_path"], **self.teacher_paths}
        for name, path in paths.items():
            if not Path(path).is_dir():
                raise FileNotFoundError(f"{name}: local HF checkpoint directory not found: {path}")
        local = dict(local_files_only=True, trust_remote_code=False)
        student_config = AutoConfig.from_pretrained(paths["student"], **local)
        student_tokenizer = AutoTokenizer.from_pretrained(paths["student"], **local)
        self._configs = {}
        for name, path in self.teacher_paths.items():
            teacher_config = AutoConfig.from_pretrained(path, **local)
            teacher_tokenizer = AutoTokenizer.from_pretrained(path, **local)
            _assert_compatible(student_config, teacher_config, student_tokenizer, teacher_tokenizer, name)
            self._configs[name] = teacher_config
        self._models = OrderedDict()

    def _get_model(self, name):
        from transformers import AutoModelForImageTextToText

        if name not in self.teacher_paths:
            raise ValueError(f"Unknown teacher: {name}")
        if name in self._models:
            self._models.move_to_end(name)
            return self._models[name]
        # Evict BEFORE loading the next model, including its temporary tensors.
        while len(self._models) >= self.max_resident:
            _, evicted = self._models.popitem(last=False)
            del evicted
            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        model = AutoModelForImageTextToText.from_pretrained(
            self.teacher_paths[name],
            config=self._configs[name],
            dtype=self.dtype,
            attn_implementation="sdpa",
            local_files_only=True,
            trust_remote_code=False,
        ).to(self.device)
        model.requires_grad_(False)
        model.eval()
        if not hasattr(model, "model") or not hasattr(model, "lm_head"):
            raise TypeError(f"{name}: teacher must expose model and lm_head")
        self._models[name] = model
        return model

    @torch.inference_mode()
    def score(self, teacher_name: str, samples: Sequence[Mapping[str, Any]]) -> list[torch.Tensor]:
        """Return CPU float32 [response_length] log-probs; masked tokens are 0.

        Each sample contains 1-D input_ids, attention_mask, responses and
        response_mask, plus position_ids [L] or [4,L] and multi_modal_inputs.
        Image patch tensors are already flattened by the rollout processor;
        they must NOT receive an extra batch dimension. Microbatch size is one.
        """
        if teacher_name not in self.teacher_paths:
            raise ValueError(f"Unknown teacher: {teacher_name}")
        if not samples:
            return []
        model = self._get_model(teacher_name)
        return [self._score_sample(model, sample) for sample in samples]

    def _score_sample(self, model, sample):
        ids = sample["input_ids"]
        attention = sample["attention_mask"]
        responses = sample["responses"]
        response_mask = sample["response_mask"]
        positions = sample["position_ids"]
        for name, value in (("input_ids", ids), ("attention_mask", attention),
                            ("responses", responses), ("response_mask", response_mask)):
            if not isinstance(value, torch.Tensor) or value.ndim != 1:
                raise ValueError(f"{name} must be a one-dimensional tensor")
        length, response_length = ids.numel(), responses.numel()
        if attention.shape != ids.shape or response_mask.shape != responses.shape:
            raise ValueError("Attention and response masks must match their respective token tensors")
        if not 0 < response_length < length:
            raise ValueError("Sample requires at least one prompt and one padded response token")
        if not torch.equal(ids[-response_length:], responses):
            raise ValueError("responses must equal the exact input_ids suffix from the student rollout")
        if not torch.all((response_mask == 0) | (response_mask == 1)):
            raise ValueError("response_mask must be binary")
        if not torch.all((attention == 0) | (attention == 1)):
            raise ValueError("attention_mask must be binary")
        valid = response_mask.bool()
        if not torch.all(attention[-response_length:][valid] == 1):
            raise ValueError("Valid response tokens must be visible in attention_mask")
        if not torch.all(attention[-response_length - 1:-1][valid] == 1):
            raise ValueError("Every valid response token must have an unmasked predecessor")
        if positions.ndim == 1 and positions.shape == ids.shape:
            positions = positions.unsqueeze(0)  # [batch=1,L], text-only
        elif positions.ndim == 2 and positions.shape == (4, length):
            positions = positions.unsqueeze(1)  # [text+t/h/w,batch=1,L]
        else:
            raise ValueError("Qwen3.5 position_ids must have shape [L] or [4,L]")
        result = torch.zeros(response_length, dtype=torch.float32)
        if not valid.any():
            return result
        visible = attention.nonzero(as_tuple=False).flatten()
        begin, end = int(visible[0]), int(visible[-1]) + 1
        if not torch.all(attention[begin:end] == 1):
            raise ValueError("Only outer padding is supported; attention_mask contains interior gaps")
        # The pinned HF Qwen3.5 recurrent layer does not apply padding masks
        # for batch size one. Remove outer pads while keeping every real token
        # and its exact rollout position IDs; do not recompute mRoPE positions.
        positions = positions[..., begin:end]
        multimodal = sample.get("multi_modal_inputs") or {}
        if not isinstance(multimodal, Mapping):
            raise TypeError("multi_modal_inputs must be a mapping of exact rollout tensors")
        reserved = {"input_ids", "attention_mask", "position_ids", "use_cache", "return_dict", "inputs_embeds"}
        if reserved.intersection(multimodal):
            raise ValueError("multi_modal_inputs cannot override token inputs or forward settings")
        inputs = {}
        for name, value in multimodal.items():
            if name == "images_seqlens":  # VERL sequence-parallel bookkeeping
                continue
            if value is not None and not isinstance(value, torch.Tensor):
                raise TypeError(f"multi_modal_inputs[{name!r}] must be a tensor or None")
            if name == "mm_token_type_ids" and value is not None:
                if value.ndim == 1:
                    value = value.unsqueeze(0)
                value = value[..., begin:end]
            inputs[name] = value.to(self.device) if value is not None else None
        output = model.model(
            input_ids=ids[begin:end].unsqueeze(0).to(self.device),
            attention_mask=attention[begin:end].unsqueeze(0).to(self.device),
            position_ids=positions.to(self.device),
            use_cache=False,
            return_dict=True,
            **inputs,
        )
        # Position L-R-1 predicts response[0]; the final input token predicts
        # a token outside the sampled sequence and is intentionally excluded.
        hidden = output.last_hidden_state[0]
        valid_indices = valid.nonzero(as_tuple=False).flatten()
        for start in range(0, len(valid_indices), self.logit_chunk_size):
            indices = valid_indices[start:start + self.logit_chunk_size]
            predictor_indices = (length - response_length - 1 - begin + indices).to(self.device)
            logits = model.lm_head(hidden[predictor_indices]).float() / self.temperature
            targets = responses[indices].to(self.device)
            log_probs = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(logits, dim=-1)
            if not torch.isfinite(log_probs).all():
                raise FloatingPointError("Teacher emitted nonfinite sampled-token log probabilities")
            result[indices.cpu()] = log_probs.cpu()
        return result
