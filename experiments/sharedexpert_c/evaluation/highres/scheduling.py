"""Pure FIFO admission budgets for the full-image evaluation runner.

The runner must peek at the next pending row without skipping it, call
``can_admit`` with the rows of all unfinished requests, and wait for an active
request to finish when admission fails. This module neither sorts nor submits
rows. The policy changes concurrency only, never a row's image/token budget.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping


VERSION = "visual-token-admission-v1"
CAPACITY_SLOTS = 16
MAX_INFLIGHT = 16
MAX_VISUAL_TOKENS_PER_ROW = 32768
MAX_INFLIGHT_VISUAL_TOKENS = 65536
MAX_PROMPT_TOKENS_PER_ROW = 40960
MAX_INFLIGHT_PROMPT_TOKENS = 81920
_COST_BANDS = ((4096, 1), (8192, 2), (16384, 4), (None, 8))


def configuration() -> dict:
    """Return an independent JSON-compatible snapshot of the fixed policy."""
    return {
        "version": VERSION,
        "capacity_slots": CAPACITY_SLOTS,
        "max_inflight": MAX_INFLIGHT,
        "max_visual_tokens_per_row": MAX_VISUAL_TOKENS_PER_ROW,
        "max_inflight_visual_tokens": MAX_INFLIGHT_VISUAL_TOKENS,
        "max_prompt_tokens_per_row": MAX_PROMPT_TOKENS_PER_ROW,
        "max_inflight_prompt_tokens": MAX_INFLIGHT_PROMPT_TOKENS,
        "visual_token_costs": [
            {"max_visual_tokens_inclusive": threshold, "slots": cost}
            for threshold, cost in _COST_BANDS
        ],
        "pending_order": "FIFO; never skip a blocked head row",
    }


def _tokens(row: Mapping) -> tuple[int, int]:
    if not isinstance(row, Mapping):
        raise ValueError("A scheduling row must be a mapping")
    values = []
    for name in ("visual_tokens", "prompt_tokens"):
        value = row.get(name)
        # bool is an int subclass, but is not a measured token count.
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
        values.append(value)
    visual, prompt = values
    if visual > MAX_VISUAL_TOKENS_PER_ROW:
        raise ValueError(f"visual_tokens must not exceed {MAX_VISUAL_TOKENS_PER_ROW}")
    if not 0 < prompt <= MAX_PROMPT_TOKENS_PER_ROW:
        raise ValueError(
            f"prompt_tokens must be between 1 and {MAX_PROMPT_TOKENS_PER_ROW}"
        )
    if visual > prompt:
        raise ValueError("visual_tokens cannot exceed prompt_tokens")
    return visual, prompt


def _cost(visual_tokens: int) -> int:
    for upper, slots in _COST_BANDS:
        if upper is None or visual_tokens <= upper:
            return slots
    raise AssertionError("The final cost band must be unbounded")


def row_cost(row: Mapping) -> int:
    """Validate frozen token counts and return this row's admission slot cost."""
    visual, _ = _tokens(row)
    return _cost(visual)


def can_admit(row: Mapping, active_rows: Iterable[Mapping]) -> bool:
    """Whether the FIFO head fits alongside every unfinished request.

    Invalid candidate or active-row token metadata raises ``ValueError`` even
    if a budget is already full. Every individually valid row fits an empty
    active set, so FIFO eventually advances when active requests finish.
    """
    visual, prompt = _tokens(row)
    active = [_tokens(item) for item in active_rows]
    return (
        len(active) + 1 <= MAX_INFLIGHT
        and _cost(visual) + sum(_cost(v) for v, _ in active) <= CAPACITY_SLOTS
        and visual + sum(v for v, _ in active) <= MAX_INFLIGHT_VISUAL_TOKENS
        and prompt + sum(p for _, p in active) <= MAX_INFLIGHT_PROMPT_TOKENS
    )
