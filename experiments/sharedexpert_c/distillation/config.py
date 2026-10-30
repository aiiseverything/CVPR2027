"""Dependency-free configuration and CPU artifact checks for domain OPD.

Loading a configuration never loads model tensors, initializes CUDA, or reads the
training data. Call ``check_artifacts`` explicitly for local artifact validation.
Relative paths are relative to the workspace, rather than the shell's cwd.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
import math
from pathlib import Path
import struct


WORKSPACE = Path(__file__).resolve().parents[3]
TEACHERS = ("generalist", "perception", "reasoning")
DEFAULTS = {
    "algorithm": "domain_opd",
    "student_path": None,
    "data_path": None,
    "teacher_paths": {name: None for name in TEACHERS},
    "domain_key": "domain",
    "domain_to_teacher": {},
    "steps": 200,
    "batch_size": 32,
    "rollout_n": 1,
    "prompt_length": 2048,
    "response_length": 8192,
    "learning_rate": 1e-6,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": -1,
    "microbatch_size": 1,
    "epochs": 1,
}


def load_config(path: str | Path, *, base_dir: str | Path | None = None) -> dict:
    """Read JSON, merge defaults, and resolve configured filesystem paths.

    Null paths and an empty domain mapping are permitted in a draft. Structural
    errors raise ValueError; ``missing_requirements`` lists unresolved choices.
    """
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError("Configuration must be a JSON object")
    cfg = deepcopy(DEFAULTS)
    cfg.update(raw)
    if isinstance(raw.get("teacher_paths"), dict):
        cfg["teacher_paths"] = {**DEFAULTS["teacher_paths"], **raw["teacher_paths"]}
    errors = validate_config(cfg)
    if errors:
        raise ValueError("Invalid OPD configuration: " + "; ".join(errors))
    base = Path(base_dir) if base_dir is not None else WORKSPACE

    def resolved(value):
        if value is None:
            return None
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else base / p).resolve())

    for key in ("student_path", "data_path"):
        cfg[key] = resolved(cfg[key])
    cfg["teacher_paths"] = {name: resolved(value) for name, value in cfg["teacher_paths"].items()}
    return cfg


def missing_requirements(cfg: Mapping) -> list[str]:
    """Return user choices still needed before an executable run can be built."""
    missing = []
    for key in ("student_path", "data_path"):
        if not cfg.get(key):
            missing.append(key)
    teachers = cfg.get("teacher_paths")
    teachers = teachers if isinstance(teachers, Mapping) else {}
    for name in TEACHERS:
        if not teachers.get(name):
            missing.append(f"teacher_paths.{name}")
    if not cfg.get("domain_to_teacher"):
        missing.append("domain_to_teacher")
    return missing


def validate_config(cfg: Mapping, *, require_resolved: bool = False) -> list[str]:
    """Return errors without performing I/O; never infer a domain mapping."""
    if not isinstance(cfg, Mapping):
        return ["Configuration must be an object"]
    errors = []
    if cfg.get("algorithm", "domain_opd") != "domain_opd":
        errors.append("algorithm must be 'domain_opd'; only prompt-domain hard routing is implemented")
    for key in ("student_path", "data_path"):
        value = cfg.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{key} must be a nonempty path string or null")
    paths = cfg.get("teacher_paths", {})
    if not isinstance(paths, Mapping):
        errors.append("teacher_paths must be an object")
    else:
        for name, value in paths.items():
            if name not in TEACHERS:
                errors.append(f"Unknown teacher role: {name!r}")
            if value is not None and (not isinstance(value, str) or not value.strip()):
                errors.append(f"teacher_paths.{name} must be a nonempty path string or null")
    domain_key = cfg.get("domain_key", "domain")
    if not isinstance(domain_key, str) or not domain_key or any(not part for part in domain_key.split(".")):
        errors.append("domain_key must be a nonempty dotted metadata path")
    mapping = cfg.get("domain_to_teacher", {})
    if not isinstance(mapping, Mapping):
        errors.append("domain_to_teacher must be an object")
    else:
        for domain, teacher in mapping.items():
            if not isinstance(domain, str) or not domain.strip():
                errors.append("domain_to_teacher keys must be nonempty strings")
            if not isinstance(teacher, str) or teacher not in TEACHERS:
                errors.append(f"domain_to_teacher[{domain!r}] must name one of {TEACHERS}")
    positive_ints = ("steps", "batch_size", "rollout_n", "prompt_length", "response_length", "microbatch_size", "epochs")
    for key in positive_ints:
        value = cfg.get(key, DEFAULTS[key])
        if type(value) is not int or value <= 0:
            errors.append(f"{key} must be a positive integer")
    for key in ("learning_rate", "temperature", "top_p"):
        value = cfg.get(key, DEFAULTS[key])
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            errors.append(f"{key} must be finite and positive")
    # The basic implementation has no correction for truncation or off-policy
    # repeated optimization. Refuse those switches instead of mislabelling loss.
    for key, expected in (("epochs", 1), ("rollout_n", 1), ("microbatch_size", 1),
                          ("temperature", 1.0), ("top_p", 1.0), ("top_k", -1)):
        if cfg.get(key, expected) != expected:
            errors.append(f"{key} must be {expected} for this on-policy implementation")
    batch = cfg.get("batch_size", DEFAULTS["batch_size"])
    micro = cfg.get("microbatch_size", DEFAULTS["microbatch_size"])
    if type(batch) is int and type(micro) is int and micro > 0 and batch % micro:
        errors.append("batch_size must be divisible by microbatch_size")
    backend = cfg.get("teacher_backend", {})
    if not isinstance(backend, Mapping):
        errors.append("teacher_backend must be an object")
    else:
        if backend.get("device", "cuda") not in ("cuda", "cpu"):
            errors.append("teacher_backend.device must be cuda or cpu (Ray assigns device indices)")
        if backend.get("dtype", "bfloat16") not in ("bfloat16", "float16", "float32"):
            errors.append("teacher_backend.dtype must be bfloat16, float16 or float32")
        for key in ("max_resident_teachers", "logit_chunk_size", "num_cpus"):
            value = backend.get(key, {"max_resident_teachers": 1, "logit_chunk_size": 128, "num_cpus": 2}[key])
            if type(value) is not int or value <= 0 or (key == "max_resident_teachers" and value > 3):
                errors.append(f"teacher_backend.{key} must be a positive integer" + (" in 1..3" if key == "max_resident_teachers" else ""))
    if require_resolved:
        errors.extend(f"Missing required choice: {key}" for key in missing_requirements(cfg))
    return errors


def _read_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _safetensors_header(path: Path) -> dict:
    """Validate the complete tensor payload extent without loading its data."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f"Truncated safetensors file: {path}")
        header_size = struct.unpack("<Q", raw)[0]
        if header_size <= 0 or header_size > min(size - 8, 100_000_000):
            raise ValueError(f"Invalid safetensors header length: {path}")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path}")
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError(f"No tensors in {path}")
    intervals = []
    byte_sizes = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
                  "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
                  "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    for name, metadata in tensors.items():
        if not isinstance(metadata, dict):
            raise ValueError(f"Invalid tensor metadata for {name} in {path}")
        offsets, shape = metadata.get("data_offsets"), metadata.get("shape")
        dtype = metadata.get("dtype")
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(x) is not int for x in offsets)
                or offsets[0] < 0 or offsets[1] < offsets[0]
                or not isinstance(shape, list) or any(type(x) is not int or x < 0 for x in shape)
                or dtype not in byte_sizes):
            raise ValueError(f"Invalid tensor metadata for {name} in {path}")
        if math.prod(shape) * byte_sizes[dtype] != offsets[1] - offsets[0]:
            raise ValueError(f"Tensor shape/payload mismatch for {name} in {path}")
        intervals.append(tuple(offsets))
    end = 0
    for start, stop in sorted(intervals):
        if start != end:
            raise ValueError(f"Overlapping or missing tensor payload in {path}")
        end = stop
    if end != size - 8 - header_size:
        raise ValueError(f"Truncated or extra tensor payload in {path}")
    return tensors


def _check_weights(directory: Path) -> None:
    index_file = directory / "model.safetensors.index.json"
    if not index_file.exists():
        _safetensors_header(directory / "model.safetensors")
        return
    index = _read_object(index_file)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError("Safetensors index has no weight_map")
    headers = {}
    for filename in set(weight_map.values()):
        if not isinstance(filename, str) or Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError(f"Invalid shard name: {filename!r}")
        headers[filename] = _safetensors_header(directory / filename)
    indexed = set(weight_map)
    actual = set()
    for filename, header in headers.items():
        if actual & set(header):
            raise ValueError("Duplicate tensor name across weight shards")
        actual.update(header)
        for name in header:
            if weight_map.get(name) != filename:
                raise ValueError(f"Tensor {name} does not match its indexed shard")
    if indexed != actual:
        raise ValueError("Safetensors index references missing tensors")


def validate_vocab_compatibility(student_vocab: Mapping, teacher_vocab: Mapping,
                                 teacher_special_tokens, output_vocab_size: int) -> list[str]:
    """Permit only appended SPECIAL declarations in already reserved logit rows.

    The frozen teacher consumes the student's exact token IDs. A newer HF
    processor may name previously unnamed output rows without changing any
    existing embedding row. This exception never permits remapping IDs, ordinary
    added tokens, or growing the model vocabulary.
    """
    errors = []
    if not student_vocab or not teacher_vocab or type(output_vocab_size) is not int or output_vocab_size <= 0:
        return ["Token vocabularies and output_vocab_size must be nonempty and valid"]
    for label, vocab in (("student", student_vocab), ("teacher", teacher_vocab)):
        if any(not isinstance(token, str) or type(idx) is not int or not 0 <= idx < output_vocab_size
               for token, idx in vocab.items()):
            errors.append(f"{label} vocabulary contains invalid or out-of-range IDs")
        if len(set(vocab.values())) != len(vocab):
            errors.append(f"{label} vocabulary has token ID collisions")
    if errors:
        return errors
    if any(token not in teacher_vocab for token in student_vocab):
        errors.append("Teacher vocabulary removes student tokens")
    if any(token in teacher_vocab and teacher_vocab[token] != idx for token, idx in student_vocab.items()):
        errors.append("Teacher vocabulary remaps student token IDs")
    special = set(teacher_special_tokens)
    max_student_id = max(student_vocab.values())
    for token in teacher_vocab.keys() - student_vocab.keys():
        if token not in special or teacher_vocab[token] <= max_student_id:
            errors.append("Extra teacher declarations must be appended special tokens in reserved output rows")
            break
    return errors


def _tokenizer_signature(directory: Path) -> dict:
    tokenizer = _read_object(directory / "tokenizer.json")
    vocab = tokenizer.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer.json must contain the shared BPE token-to-ID vocabulary")
    if any(not isinstance(token, str) or type(idx) is not int or idx < 0 for token, idx in vocab.items()):
        raise ValueError("Malformed tokenizer vocabulary")
    vocab = dict(vocab)
    added = tokenizer.get("added_tokens", [])
    if not isinstance(added, list):
        raise ValueError("Malformed tokenizer added_tokens")
    for item in added:
        if not isinstance(item, dict) or not isinstance(item.get("content"), str) or type(item.get("id")) is not int:
            raise ValueError("Malformed tokenizer added token")
        if item["id"] < 0 or (item["content"] in vocab and vocab[item["content"]] != item["id"]):
            raise ValueError("Conflicting added token ID")
        vocab[item["content"]] = item["id"]
    if len(set(vocab.values())) != len(vocab):
        raise ValueError("Multiple token strings use the same token ID")
    tokenizer_config = _read_object(directory / "tokenizer_config.json")
    model_config = _read_object(directory / "config.json")
    text_config = model_config.get("text_config", model_config)
    vocab_size = text_config.get("vocab_size")
    if type(vocab_size) is not int or max(vocab.values()) >= vocab_size:
        raise ValueError("Model vocab_size does not cover all tokenizer IDs")
    special_ids = {key: model_config.get(key) for key in (
        "bos_token_id", "eos_token_id", "pad_token_id", "image_token_id", "video_token_id",
        "vision_start_token_id", "vision_end_token_id")}
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        token = tokenizer_config.get(key.removesuffix("_id"))
        token = token.get("content") if isinstance(token, dict) else token
        tokenizer_id = vocab.get(token)
        if tokenizer_id is not None:
            special_ids[key] = tokenizer_id
    model = deepcopy(tokenizer["model"])
    # Tokenizers 0.2x serialized BPE merges as strings, newer versions as pairs.
    if isinstance(model.get("merges"), list):
        model["merges"] = [item.split(" ", 1) if isinstance(item, str) else item for item in model["merges"]]
    return {"vocab": vocab, "vocab_size": vocab_size, "special_ids": special_ids,
            "normalizer": tokenizer.get("normalizer"), "pre_tokenizer": tokenizer.get("pre_tokenizer"),
            "model": model, "added": {item["content"]: item for item in added},
            "special_tokens": {item["content"] for item in added if item.get("special") is True}}


def _tokenizer_errors(student: dict, teacher: dict) -> list[str]:
    errors = validate_vocab_compatibility(student["vocab"], teacher["vocab"],
                                          teacher["special_tokens"], student["vocab_size"])
    for key in ("vocab_size", "special_ids", "normalizer", "pre_tokenizer", "model"):
        if student[key] != teacher[key]:
            errors.append(f"Tokenizer/model {key} differs from student")
    for token, settings in student["added"].items():
        if teacher["added"].get(token) != settings:
            errors.append("Existing added-token behavior differs from student")
            break
    return errors


def _image_signature(directory: Path) -> dict:
    """Normalize old separate and new combined HF processor metadata."""
    combined = directory / "processor_config.json"
    separate = directory / "preprocessor_config.json"
    image_config = _read_object(combined).get("image_processor") if combined.exists() else None
    if image_config is None and separate.exists():
        image_config = _read_object(separate)
    if not isinstance(image_config, dict) or not image_config:
        raise ValueError("Missing image processor configuration (combined or separate)")
    required = ("size", "patch_size", "temporal_patch_size", "merge_size", "image_mean", "image_std")
    if any(key not in image_config for key in required):
        raise ValueError("Image processor metadata lacks size/patch/normalization settings")
    # These defaults are the settings of the project's Qwen2VL image processor.
    defaults = {"do_convert_rgb": True, "do_normalize": True, "do_rescale": True,
                "do_resize": True, "resample": 3, "rescale_factor": 1 / 255}
    return {**{key: image_config[key] for key in required},
            **{key: image_config.get(key, value) for key, value in defaults.items()}}


def check_artifacts(cfg: Mapping) -> list[str]:
    """Check paths, exported weights, and identical tokenizer IDs on CPU.

    This verifies files and metadata, not weight numerical correctness or GPU
    compatibility. Training-data schema and domain coverage need a parquet scan
    in the launcher; no parquet reader is imported by this module.
    """
    errors = validate_config(cfg, require_resolved=True)
    if errors:
        return errors
    data = Path(cfg["data_path"])
    if not data.is_file():
        errors.append(f"data_path is not a local file: {data}")
    signatures, image_signatures = {}, {}
    paths = {"student": cfg["student_path"], **cfg["teacher_paths"]}
    for role, raw in paths.items():
        directory = Path(raw)
        try:
            if not directory.is_dir():
                raise ValueError(f"Not a local model directory: {directory}")
            model = _read_object(directory / "config.json")
            if model.get("model_type") != "qwen3_5":
                raise ValueError("Only the project's Qwen3.5 dense multimodal checkpoints are supported")
            image_signatures[role] = _image_signature(directory)
            tokenizer_cfg = _read_object(directory / "tokenizer_config.json")
            template = directory / "chat_template.jinja"
            if not tokenizer_cfg.get("chat_template") and not (template.is_file() and template.stat().st_size):
                raise ValueError("Missing chat template")
            _check_weights(directory)
            signatures[role] = _tokenizer_signature(directory)
        except (OSError, ValueError, TypeError, KeyError, struct.error) as exc:
            errors.append(f"{role}: {exc}")
    if "student" in signatures:
        for role in TEACHERS:
            if role in signatures:
                errors.extend(f"{role}: tokenizer compatibility: {error}"
                              for error in _tokenizer_errors(signatures["student"], signatures[role]))
    if "student" in image_signatures:
        for role in TEACHERS:
            if role in image_signatures and image_signatures[role] != image_signatures["student"]:
                errors.append(f"{role}: image processor geometry/normalization differs from student")
    return errors
