"""CPU-only admission from this vLLM startup's measured KV token capacity.

The launcher owns profile changes: advance only after a confirmed owned-server
OOM. These helpers do not alter model, image, decoding, or selection settings.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
import re


VERSION = "profiled-kv-admission-v1"
MAX_INFLIGHT = 64
FALLBACK_INFLIGHT = 32
MAX_PROMPT_TOKENS = 40960
MAX_VISUAL_TOKENS = 32768
MAX_OUTPUT_TOKENS = 8192
PER_REQUEST_OVERHEAD_TOKENS = 4096
_PROFILES = ((64, 32768, 0.95), (32, 16384, 0.95), (32, 16384, 0.92))
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_CAPACITY = re.compile(
    r"GPU\s+KV\s+cache\s+size:\s*((?:\d{1,3}(?:,\d{3})+)|\d+)\s+tokens\b",
    re.IGNORECASE,
)
_OOM = re.compile(
    r"\bCUDA(?:\s+error:)?\s+out\s+of\s+memory\b"
    r"|\btorch(?:\.cuda)?\.OutOfMemoryError\b",
    re.IGNORECASE,
)
_OOM_TAIL_BYTES = 1024 * 1024


def configuration() -> dict:
    """Independent JSON-compatible policy snapshot for freezing with a run."""
    return {
        "version": VERSION,
        "max_inflight": MAX_INFLIGHT,
        "max_num_seqs": 64,
        "max_num_batched_tokens": 32768,
        "gpu_memory_utilization": 0.95,
        "kv_fraction": 0.9,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "per_request_overhead_tokens": PER_REQUEST_OVERHEAD_TOKENS,
        "fallback_inflight": FALLBACK_INFLIGHT,
        "max_prompt_tokens_per_row": MAX_PROMPT_TOKENS,
        "max_visual_tokens_per_row": MAX_VISUAL_TOKENS,
        "profiles": [
            {"max_num_seqs": seqs, "max_num_batched_tokens": batch,
             "gpu_memory_utilization": memory}
            for seqs, batch, memory in _PROFILES
        ],
        "profile_transition": "next profile only after confirmed owned-server OOM",
    }


def _offset(offset: int) -> None:
    if type(offset) is not int or offset < 0:
        raise ValueError("Log offset must be a nonnegative byte offset")


def read_cache_capacity(log_path: str | Path, offset: int) -> int | None:
    """Last reported KV token capacity after ``offset``, never an older boot.

    Reads a bounded snapshot in streaming lines. A missing log or a segment
    without a report returns None. If the log is shorter than the recorded
    offset, raise rather than silently trusting information from its start.
    """
    _offset(offset)
    try:
        stream = Path(log_path).open("rb")
    except FileNotFoundError:
        return None
    with stream:
        end = stream.seek(0, 2)
        if offset > end:
            raise ValueError("Log is shorter than this startup's recorded offset")
        stream.seek(offset)
        capacity = None
        while stream.tell() < end:
            line = stream.readline(end - stream.tell()).decode("utf-8", errors="replace")
            for match in _CAPACITY.finditer(_ANSI.sub("", line)):
                value = int(match.group(1).replace(",", ""))
                if value <= 0:
                    raise ValueError("Reported GPU KV cache capacity must be positive")
                capacity = value
        return capacity


def runtime_budget(capacity_tokens: int | None, profile_index: int = 0) -> dict:
    """Reserve 90% of measured KV capacity, or use a bounded unknown fallback."""
    if type(profile_index) is not int or not 0 <= profile_index < len(_PROFILES):
        raise ValueError("Unknown throughput profile index")
    if capacity_tokens is not None and (type(capacity_tokens) is not int or capacity_tokens <= 0):
        raise ValueError("KV capacity must be a positive integer or None")
    profile = configuration()["profiles"][profile_index]
    limit = min(MAX_INFLIGHT, profile["max_num_seqs"])
    if capacity_tokens is None:
        limit = min(limit, FALLBACK_INFLIGHT)
    return {
        "version": VERSION,
        "profile_index": profile_index,
        "profile": profile,
        "capacity_tokens": capacity_tokens,
        "max_inflight": limit,
        "reserved_token_budget": None if capacity_tokens is None else capacity_tokens * 9 // 10,
        "capacity_source": "unknown; bounded inflight fallback" if capacity_tokens is None else "current startup GPU KV cache log",
    }


def validate_row(row: Mapping) -> None:
    """Validate frozen per-row counts without modifying or resizing the row."""
    if not isinstance(row, Mapping):
        raise ValueError("A scheduling row must be a mapping")
    visual, prompt = row.get("visual_tokens"), row.get("prompt_tokens")
    if type(prompt) is not int or not 1 <= prompt <= MAX_PROMPT_TOKENS:
        raise ValueError(f"prompt_tokens must be an integer in [1, {MAX_PROMPT_TOKENS}]")
    if type(visual) is not int or not 0 <= visual <= MAX_VISUAL_TOKENS:
        raise ValueError(f"visual_tokens must be an integer in [0, {MAX_VISUAL_TOKENS}]")
    if visual > prompt:
        raise ValueError("visual_tokens cannot exceed prompt_tokens")


def _reservation(row: Mapping) -> int:
    validate_row(row)
    return row["prompt_tokens"] + MAX_OUTPUT_TOKENS + PER_REQUEST_OVERHEAD_TOKENS


def can_admit(row: Mapping, active_rows: Iterable[Mapping], budget: Mapping) -> bool:
    """Admit the FIFO head when both profiled token and inflight budgets fit.

    An individually valid row that cannot fit even alone raises ValueError;
    the caller must record this explicit resource failure instead of waiting
    forever. Unknown KV capacity uses only the fallback inflight limit.
    """
    wanted = _reservation(row)
    active = [_reservation(item) for item in active_rows]
    if not isinstance(budget, Mapping):
        raise ValueError("Runtime budget must be a mapping")
    limit = budget.get("max_inflight")
    if type(limit) is not int or not 1 <= limit <= MAX_INFLIGHT:
        raise ValueError("Runtime max_inflight must be an integer in [1, 64]")
    if "reserved_token_budget" not in budget:
        raise ValueError("Runtime budget is missing reserved_token_budget")
    tokens = budget["reserved_token_budget"]
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        raise ValueError("Reserved token budget must be a nonnegative integer or None")
    if tokens is None and limit > FALLBACK_INFLIGHT:
        raise ValueError("Unknown KV capacity requires at most 32 inflight requests")
    if tokens is not None and wanted > tokens:
        raise ValueError(
            f"Single row needs {wanted} reserved tokens but measured budget is {tokens}"
        )
    return len(active) + 1 <= limit and (tokens is None or wanted + sum(active) <= tokens)


def owned_server_oom(log_path: str | Path, offset: int) -> bool:
    """Detect explicit GPU OOM evidence in at most the last 1 MiB of this boot.

    The caller supplies the exclusively owned server log and pre-launch byte
    offset. Exit codes, network failures, or an old boot's OOM are insufficient.
    """
    _offset(offset)
    try:
        stream = Path(log_path).open("rb")
    except FileNotFoundError:
        return False
    with stream:
        end = stream.seek(0, 2)
        if offset > end:
            raise ValueError("Log is shorter than this startup's recorded offset")
        start = max(offset, end - _OOM_TAIL_BYTES)
        stream.seek(start)
        tail = stream.read(end - start).decode("utf-8", errors="replace")
    return _OOM.search(_ANSI.sub("", tail)) is not None
