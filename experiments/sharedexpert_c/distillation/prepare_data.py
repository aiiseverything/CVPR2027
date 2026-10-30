"""Build the user-requested 1:1:1 domain mixture from audited training files.

No teacher or student is loaded. Existing files remain unchanged. Evaluation
metadata is used only to reject overlapping identities, never as training data.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random

from .config import TEACHERS, WORKSPACE


DOMAIN_ORDER = ("perception", "reasoning", "generalist")
SOURCE_WEIGHTS = {
    "perception": {"tallyqa": 4, "clevr": 4, "refcoco": 2},
    "reasoning": {"geometry3k": 4, "mmk12": 2, "virl39k": 4},
    # Generalist retains VQA : document/chart : IF = 4 : 1 : 1.
    "generalist": {"vqav2": 8, "docvqa": 1, "chartqa": 1, "if_format": 2},
}
EXPERIMENT = WORKSPACE / "experiments/sharedexpert_c"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_hashes(value) -> list[str]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise ValueError("image_sha256 must be a nonempty list (or JSON list) of hashes")
    return value


def allocate(total: int, weights: dict[str, int]) -> dict[str, int]:
    """Largest-remainder quotas; ties follow the documented source order."""
    if type(total) is not int or total < 0 or not weights or any(type(v) is not int or v <= 0 for v in weights.values()):
        raise ValueError("Allocation needs a nonnegative count and positive integer weights")
    denominator = sum(weights.values())
    quotas = {key: total * weight // denominator for key, weight in weights.items()}
    order = sorted(weights, key=lambda key: -(total * weights[key] % denominator))
    for key in order[:total - sum(quotas.values())]:
        quotas[key] += 1
    return quotas


def _identity(row) -> tuple:
    info = row.get("extra_info")
    if not isinstance(info, dict):
        raise ValueError("Missing extra_info metadata")
    if info.get("split") != "train":
        raise ValueError(f"Refusing non-training record: {info.get('record_id')}")
    for key in ("source", "record_id", "question_sha256"):
        if not isinstance(info.get(key), str) or not info[key]:
            raise ValueError(f"Missing {key}")
    hashes = tuple(image_hashes(info.get("image_sha256")))
    return info["record_id"], info["question_sha256"], hashes


def select_domain(rows: list[dict], domain: str, count: int, seed: int) -> tuple[list[dict], dict]:
    """Deduplicate repeated exposures before sampling each source's quota."""
    weights = SOURCE_WEIGHTS[domain]
    pools = {source: [] for source in weights}
    seen = {}
    excluded = Counter()
    for row in rows:
        record_id, question_hash, images = _identity(row)
        source = row["extra_info"]["source"]
        if source not in pools:
            # The generalist teacher deliberately contains p/r replay; exclude
            # those rows from the student's third domain to avoid double counting.
            if domain != "generalist" or source not in SOURCE_WEIGHTS["perception"] | SOURCE_WEIGHTS["reasoning"]:
                raise ValueError(f"Unexpected source {source!r} in {domain} input")
            excluded[source] += 1
            continue
        signature = (source, question_hash, images, json.dumps(row.get("reward_model"), sort_keys=True),
                     json.dumps(row.get("prompt"), sort_keys=True))
        if record_id in seen:
            if seen[record_id] != signature:
                raise ValueError(f"Conflicting duplicate training identity: {record_id}")
            continue
        seen[record_id] = signature
        pools[source].append(row)
    quotas = allocate(count, weights)
    selected = []
    source_stats = {}
    for source, quota in quotas.items():
        pool = pools[source]
        if quota and not pool:
            raise ValueError(f"No eligible {source} training examples")
        # Stable stream per source: adding another domain does not perturb this
        # source's random choices. Exhaust each shuffled deck before repetition.
        rng = random.Random(f"{seed}:{domain}:{source}")
        remaining = quota
        while remaining:
            deck = list(pool)
            rng.shuffle(deck)
            taken = deck[:remaining]
            selected.extend(deepcopy(row) for row in taken)
            remaining -= len(taken)
        source_stats[source] = {"eligible_unique_records": len(pool), "selected_rows": quota,
                                "repeated_exposures": max(0, quota - len(pool))}
    random.Random(f"{seed}:{domain}:order").shuffle(selected)
    for row in selected:
        row["domain"] = domain
    return selected, {"source_weights": weights, "source_counts": quotas, "sources": source_stats,
                      "excluded_teacher_replay_rows": dict(excluded)}


def mix_domains(inputs: dict[str, list[dict]], count: int = 6400, seed: int = 20261023) -> tuple[list[dict], dict]:
    if type(count) is not int or count < 3:
        raise ValueError("At least three rows are required for the three-domain mixture")
    quotas = allocate(count, {domain: 1 for domain in DOMAIN_ORDER})
    selected, stats = {}, {}
    for domain in DOMAIN_ORDER:
        selected[domain], stats[domain] = select_domain(inputs[domain], domain, quotas[domain], seed)
    cursors = Counter()
    rows = []
    for index in range(count):
        domain = DOMAIN_ORDER[index % 3]
        row = selected[domain][cursors[domain]]
        cursors[domain] += 1
        # Keep original extra_info.index as provenance; dataset order is explicit
        # through this new top-level field and parquet row position.
        row["opd_index"] = index
        rows.append(row)
    return rows, {"rows": count, "seed": seed, "domain_order": list(DOMAIN_ORDER),
                  "domain_counts": dict(cursors), "domains": stats,
                  "ordering": "cyclic perception,reasoning,generalist; shuffle=False",
                  "ratio_note": "Exact 1:1:1 in every consecutive 96-row block; final domain counts differ by at most one when total rows is not divisible by 3."}


def read_eval_identities(path: Path) -> tuple[set, set, set]:
    """Read only identity fields; answer fields are intentionally ignored."""
    images, pairs, records = set(), set(), set()
    with path.open() as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            hashes = row.get("image_sha256", [])
            if isinstance(hashes, str):
                hashes = json.loads(hashes)
            if not isinstance(hashes, list):
                raise ValueError("Invalid evaluation image identity metadata")
            images.update(hashes)
            question = row.get("question_sha256")
            if question:
                pairs.add((question, tuple(hashes)))
            # Only canonical training record IDs are directly comparable.
            if row.get("record_id"):
                records.add(row["record_id"])
    return images, pairs, records


def audit_rows(rows: list[dict], eval_identities, *, check_image_paths: bool = True) -> dict:
    eval_images, eval_pairs, eval_records = eval_identities
    record_ids, pairs, all_images = set(), set(), set()
    domain_images = {domain: set() for domain in DOMAIN_ORDER}
    paths = set()
    for index, row in enumerate(rows):
        record_id, question, hashes = _identity(row)
        if set(hashes) & eval_images or (question, hashes) in eval_pairs or record_id in eval_records:
            raise ValueError(f"Evaluation overlap at output row {index}: {record_id}")
        record_ids.add(record_id)
        pairs.add((question, hashes))
        all_images.update(hashes)
        domain_images[row["domain"]].update(hashes)
        images = row.get("images")
        if not isinstance(images, list) or len(images) != len(hashes):
            raise ValueError(f"Image payload/identity length mismatch for {record_id}")
        for image in images:
            path = image.get("path")
            if not isinstance(path, str) or not path:
                raise ValueError(f"Invalid image path for {record_id}")
            paths.add(path)
    if check_image_paths:
        for path in paths:
            if not Path(path).is_file():
                raise ValueError(f"Missing training image: {path}")
    cross_domain_images = {}
    for index, left in enumerate(DOMAIN_ORDER):
        for right in DOMAIN_ORDER[index + 1:]:
            cross_domain_images[f"{left}:{right}"] = len(domain_images[left] & domain_images[right])
    return {"unique_records": len(record_ids), "repeated_record_exposures": len(rows) - len(record_ids),
            "unique_question_image_pairs": len(pairs), "duplicate_question_image_pairs": len(rows) - len(pairs),
            "unique_image_hashes": len(all_images), "unique_image_paths": len(paths),
            "cross_domain_shared_images": cross_domain_images, "evaluation_identity_overlap": 0,
            "scope": "Checks existing exact RGB hashes and identity metadata; does not recompute image hashes or prove semantic decontamination.",
            "teacher_training_overlap": "Intentional reuse of existing teacher train pools; not an independent teacher holdout."}


def build(input_dir: Path, output: Path, eval_metadata: Path | list[Path], *, count=6400, seed=20261023) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq

    source_paths = {domain: input_dir / f"{domain}.parquet" for domain in DOMAIN_ORDER}
    if output.resolve() in {path.resolve() for path in source_paths.values()}:
        raise ValueError("Output must not overwrite a teacher dataset")
    # Compute before reading, then recheck before publishing to detect concurrent
    # preparation or accidental modification of source files during construction.
    hashes = {domain: file_sha256(path) for domain, path in source_paths.items()}
    eval_paths = [eval_metadata] if isinstance(eval_metadata, Path) else list(eval_metadata)
    if not eval_paths:
        raise ValueError("At least one evaluation identity manifest is required")
    eval_hashes = {str(path.resolve()): file_sha256(path) for path in eval_paths}
    inputs = {domain: pq.read_table(path).to_pylist() for domain, path in source_paths.items()}
    rows, manifest = mix_domains(inputs, count=count, seed=seed)
    identities = (set(), set(), set())
    for path in eval_paths:
        for target, source in zip(identities, read_eval_identities(path)):
            target.update(source)
    manifest["audit"] = audit_rows(rows, identities)
    manifest["sources"] = {domain: {"path": str(path.resolve()), "sha256": hashes[domain], "rows": len(inputs[domain])}
                           for domain, path in source_paths.items()}
    manifest["evaluation_metadata"] = {"files": [{"path": path, "sha256": digest} for path, digest in eval_hashes.items()],
                                       "usage": "identity exclusion only; no evaluation examples or labels in training"}
    if (hashes != {domain: file_sha256(path) for domain, path in source_paths.items()}
            or eval_hashes != {str(path.resolve()): file_sha256(path) for path in eval_paths}):
        raise ValueError("Input artifacts changed while building the OPD dataset")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary)
    manifest["output"] = {"path": str(output.resolve()), "sha256": file_sha256(temporary)}
    manifest_path = output.with_suffix(".manifest.json")
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(output)
    manifest_tmp.replace(manifest_path)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=EXPERIMENT / "data_pipeline/prepared")
    parser.add_argument("--output", type=Path, default=EXPERIMENT / "data_pipeline/prepared/domain_opd_6400.parquet")
    parser.add_argument("--eval-metadata", type=Path, action="append",
                        help="Repeat for heldout identity manifests; default checks main and BLINK/MMVP suites")
    parser.add_argument("--count", type=int, default=6400)
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args()
    if args.eval_metadata is None:
        args.eval_metadata = [EXPERIMENT / "evaluation/suite.jsonl", EXPERIMENT / "evaluation/suite_blink_mmvp.jsonl"]
    manifest = build(args.input_dir, args.output, args.eval_metadata, count=args.count, seed=args.seed)
    print(json.dumps({"output": manifest["output"], "rows": manifest["rows"],
                      "domain_counts": manifest["domain_counts"], "audit": manifest["audit"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
