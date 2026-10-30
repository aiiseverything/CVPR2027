"""Strict prompt-domain routing: one explicit teacher per prompt."""
from __future__ import annotations

from collections.abc import Mapping

from .config import TEACHERS


def get_domain(record: Mapping, domain_key: str = "domain") -> str:
    """Read an exact dotted metadata path; missing/unknown values never fallback.

    Pass a parquet row, or a row assembled from VERL's ``non_tensor_batch``.
    For example ``extra_info.domain`` reads ``record['extra_info']['domain']``.
    Metadata containers must be mappings: JSON strings are not silently parsed.
    """
    if not isinstance(domain_key, str) or not domain_key or any(not key for key in domain_key.split(".")):
        raise ValueError("domain_key must be a nonempty dotted metadata path")
    value = record
    for part in domain_key.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"Missing domain metadata at {domain_key!r}")
        value = value[part]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Domain at {domain_key!r} must be a nonempty string")
    return value


def route_record(record: Mapping, cfg: Mapping) -> str:
    """Return a teacher role, requiring an explicit domain-to-teacher entry."""
    domain = get_domain(record, cfg.get("domain_key", "domain"))
    mapping = cfg.get("domain_to_teacher", {})
    if not isinstance(mapping, Mapping):
        raise ValueError("domain_to_teacher must be an object")
    if domain not in mapping:
        raise ValueError(f"Unmapped domain {domain!r}; add an explicit domain_to_teacher entry")
    teacher = mapping[domain]
    if not isinstance(teacher, str) or teacher not in TEACHERS:
        raise ValueError(f"Invalid teacher role {teacher!r} for domain {domain!r}")
    return teacher


def route_batch(records, cfg: Mapping) -> list[str]:
    """Preserve row order and annotate routing failures with the row index."""
    routed = []
    for index, record in enumerate(records):
        try:
            routed.append(route_record(record, cfg))
        except ValueError as exc:
            raise ValueError(f"Row {index}: {exc}") from exc
    return routed
