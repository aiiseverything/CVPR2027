"""Stream six pinned Parquet sources into content-addressed SQLite and JSONL.

Image payloads are copied byte-for-byte, never resized or re-encoded. A completed
shard and its questions commit together; the final manifest is a commit marker.
"""
from __future__ import annotations

import argparse
import base64
import collections
import fcntl
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import time
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

from sources import BENCHMARKS, DEFAULT_ROOT, atomic_json, file_sha256, load_manifest, utc_now
try:
    from .annotation_issues import known_annotation_issue, known_missing_gold, known_non_latin_option_source
except ImportError:
    from annotation_issues import known_annotation_issue, known_missing_gold, known_non_latin_option_source

SCHEMA_VERSION = "highres-perception-questions-v1"
# Header inspection only: the trusted benchmark's original large images must
# not be rejected or implicitly downsampled by Pillow's default pixel limit.
Image.MAX_IMAGE_PIXELS = None
OPTION_LINE = re.compile(r"^\s*(?:[（(]([A-Z])[)）]|([A-Z])[.:：)])\s*(.*)$", re.DOTALL)
EMBEDDED_OPTION = re.compile(r"(?m)^[ \t]*(?:[（(]([A-Z])[)）]|([A-Z])[.:：)])[ \t]+")
VSTAR_SUFFIX = "Answer with the option's letter from the given choices directly."


def json_text(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def log(event, **values):
    print(json_text({"time": utc_now(), "event": event, **values}), flush=True)


def read_selection(path):
    raw = Path(path).read_bytes()
    selection = json.loads(raw)
    if selection.get("schema_version") != "highres-selection-v1" or selection.get("ready") is not True:
        raise ValueError("Perception selection is not ready")
    if set(selection.get("benchmarks", {})) != set(BENCHMARKS):
        raise ValueError("Selection must specify all six benchmark subsets")
    for benchmark, spec in selection["benchmarks"].items():
        count = spec.get("expected_count")
        if not isinstance(count, int) or not 0 < count <= BENCHMARKS[benchmark]["expected_questions"]:
            raise ValueError(f"Invalid expected selected count for {benchmark}")
        if not spec.get("filters") and "include_source_ids" not in spec:
            raise ValueError(f"Explicit perception filter or source ID inclusion list required: {benchmark}")
        for condition in spec.get("filters", []):
            if set(condition) != {"field", "op", "value"} or condition["op"] not in {"eq", "startswith", "in"}:
                raise ValueError("Invalid selection predicate")
            if condition["op"] == "startswith" and not isinstance(condition["value"], str):
                raise ValueError("startswith expects a string")
            if condition["op"] == "in" and not isinstance(condition["value"], list):
                raise ValueError("in expects a JSON list")
        for key in ("include_source_ids", "exclude_source_ids"):
            if key in spec and (not isinstance(spec[key], list) or any(not isinstance(i, str) for i in spec[key])
                                or len(set(spec[key])) != len(spec[key])):
                raise ValueError(f"{key} must contain unique string source IDs")
    return selection, hashlib.sha256(raw).hexdigest()


def selected_row(row, spec):
    source_id = str(row.get("question_id", row.get("id", row.get("index"))))
    if "include_source_ids" in spec and source_id not in spec["include_source_ids"]:
        return False
    if source_id in spec.get("exclude_source_ids", []):
        return False
    for condition in spec.get("filters", []):
        field, operator, wanted = condition["field"], condition["op"], condition["value"]
        if field not in row:
            raise ValueError(f"Selection references an absent source column: {field}")
        value = row[field]
        if operator == "eq" and value != wanted:
            return False
        if operator == "startswith" and (not isinstance(value, str) or not value.startswith(wanted)):
            return False
        if operator == "in" and value not in wanted:
            return False
    return True


def options_from_lines(lines, *, source_row=None):
    options, duplicates = {}, []
    allow_non_latin_one = (known_non_latin_option_source(source_row)
                          and lines == source_row["metadata"]["option_lines"])
    for line in lines:
        match = OPTION_LINE.fullmatch(str(line))
        if not match:
            if not allow_non_latin_one or not str(line).startswith("（一）"):
                raise ValueError(f"Unrecognized option label: {str(line)[:80]!r}")
            # Eight exact pinned source rows display this literal label. Keep
            # it as-is; neither the source prompt nor gold A is corrected.
            letter, value = "一", str(line)[3:].strip()
        else:
            letter, value = match.group(1) or match.group(2), match.group(3).strip()
        if letter in options:
            duplicates.append(letter)
        else:
            options[letter] = value
    return options, duplicates


def embedded_options(text):
    matches = list(EMBEDDED_OPTION.finditer(text))
    if not matches:
        return {}, [], []
    lines = []
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        option = text[match.start():end].strip()
        if option.endswith(VSTAR_SUFFIX):
            option = option[:-len(VSTAR_SUFFIX)].rstrip()
        lines.append(option)
    options, duplicates = options_from_lines(lines)
    return options, lines, duplicates


def image_payload(row, benchmark):
    value = row["bytes"] if benchmark.startswith("mme_rw_") else row["image"]
    if isinstance(value, dict):
        value = value.get("bytes")
    if isinstance(value, str):
        # Whitespace is permitted in base64; decoded compressed image bytes are
        # unchanged. No PIL save/convert/thumbnail operation is performed.
        value = base64.b64decode(re.sub(r"\s+", "", value), validate=True)
    if not isinstance(value, (bytes, bytearray, memoryview)) or not value:
        raise ValueError(f"Missing embedded original image bytes: {benchmark}")
    return bytes(value)


def normalize_row(row, entry, ordinal, row_index):
    benchmark = entry["benchmark"]
    excluded = {"image", "bytes", "crop_image"}
    metadata = {key: value for key, value in row.items() if key not in excluded}
    metadata["source"] = {"repo": entry["repo"], "revision": entry["revision"], "path": entry["path"],
                          "sha256": entry["sha256"], "row_index": row_index}
    metadata["original_id"] = row.get("question_id", row.get("id", row.get("index")))
    metadata["options_embedded"] = benchmark in {"vstar", "zoom"}
    if benchmark == "vstar":
        question, answer = row["text"], row["label"]
        options, lines, duplicates = embedded_options(question)
    elif benchmark == "zoom":
        question, answer = row["query"], row["response"]
        options, lines, duplicates = embedded_options(question)
        metadata["image_variant"] = "full"
        metadata["crop_image_used"] = False
        if str(row.get("question_type", "")).lower() == "mcq" and not options:
            raise ValueError(f"Zoom MCQ has no parsed options: {ordinal}")
    elif benchmark.startswith("hrbench_"):
        question, answer = row["question"], row["answer"]
        options = {letter: str(row[letter]) for letter in "ABCD" if row.get(letter) is not None}
        lines = [f"({letter}) {value}" for letter, value in options.items()]
        duplicates = []
        if "cycle_category" not in row:
            raise ValueError("HRBench source lost cycle_category")
    elif benchmark.startswith("mme_rw_"):
        question, answer = row["question"], row["answer"]
        lines = list(row["multi-choice options"])
        metadata["option_lines"] = lines
        source_row = {"id": f"{benchmark}:{ordinal:06d}", "benchmark": benchmark,
                      "question": question, "answer": answer, "metadata": metadata}
        options, duplicates = options_from_lines(lines, source_row=source_row)
        if "category" not in row or "l2-category" not in row:
            raise ValueError("MME-RealWorld source lost native grouping columns")
    else:
        raise ValueError(f"Unknown benchmark {benchmark}")
    if not isinstance(question, str) or not isinstance(answer, str):
        raise ValueError("Question and answer must be original source strings")
    metadata["option_lines"] = lines
    metadata["option_labels_duplicated"] = duplicates
    if duplicates:
        metadata["options_dict_policy"] = "first occurrence; option_lines is lossless and authoritative"
    if benchmark != "zoom" and not options:
        raise ValueError(f"Multiple-choice example has no parsed options: {benchmark}/{ordinal}")
    text_row = {"id": f"{benchmark}:{ordinal:06d}", "benchmark": benchmark,
                "question": question, "answer": answer, "options": options, "metadata": metadata}
    issue = known_annotation_issue(text_row)
    if issue:
        metadata["annotation_issue_type"] = issue
        metadata["annotation_anomaly"] = "Exact pinned upstream annotation issue; original question, option lines and gold preserved without correction or exclusion"
    if options and answer not in options:
        if not known_missing_gold(text_row):
            raise ValueError(f"Answer {answer!r} is absent from option labels: {benchmark}/{ordinal}")
        metadata["answer_label_missing_from_options"] = True
    payload = image_payload(row, benchmark)
    digest = hashlib.sha256(payload).hexdigest()
    with Image.open(io.BytesIO(payload)) as img:
        width, height = img.size
        mime = Image.MIME.get(img.format)
    if not mime or width <= 0 or height <= 0:
        raise ValueError(f"Unknown image format/dimensions: {benchmark}/{ordinal}")
    image = {"sha256": digest, "width": width, "height": height, "mime": mime, "size": len(payload)}
    question = {"id": f"{benchmark}:{ordinal:06d}", "benchmark": benchmark, "question": question,
                "options": options, "answer": answer, "image": image, "metadata": metadata}
    return question, payload


def open_database(path, source_manifest_sha, selection_sha):
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute("PRAGMA temp_store=FILE")
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS images (
            sha256 TEXT PRIMARY KEY, data BLOB NOT NULL, width INTEGER NOT NULL,
            height INTEGER NOT NULL, mime TEXT NOT NULL, size INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS questions (
            id TEXT PRIMARY KEY, benchmark TEXT NOT NULL, ordinal INTEGER NOT NULL,
            source_key TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(benchmark, ordinal)
        );
        CREATE TABLE IF NOT EXISTS prepared_shards (
            source_key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, ordinal_start INTEGER NOT NULL,
            benchmark TEXT NOT NULL, rows INTEGER NOT NULL, selected_rows INTEGER NOT NULL, completed_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS preparation_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
    """)
    expected = {"source_manifest_sha256": source_manifest_sha, "schema_version": SCHEMA_VERSION,
                "selection_sha256": selection_sha}
    for key, value in expected.items():
        previous = connection.execute("SELECT value FROM preparation_meta WHERE key=?", (key,)).fetchone()
        if previous is not None and previous[0] != value:
            raise ValueError(f"Refusing to mix different preparation inputs: {key}")
        connection.execute("INSERT OR IGNORE INTO preparation_meta VALUES (?,?)", (key, value))
    connection.commit()
    return connection


def prepare_shard(connection, root, entry, ordinal_start, batch_size, selection_spec, minimum_free):
    source_key = entry["local_path"]
    previous = connection.execute("SELECT sha256,ordinal_start,rows,selected_rows FROM prepared_shards WHERE source_key=?", (source_key,)).fetchone()
    if previous:
        if previous[:2] != (entry["sha256"], ordinal_start):
            raise ValueError(f"Previously prepared shard does not match pinned order: {source_key}")
        log("shard_already_prepared", file=source_key, rows=previous[2], selected_rows=previous[3])
        return previous[2], previous[3]
    path = root / source_key
    if shutil.disk_usage(root).free < minimum_free:
        raise OSError("Persistent disk is below reserved headroom before preparing a shard")
    if path.stat().st_size != entry["size"] or file_sha256(path) != entry["sha256"]:
        raise ValueError(f"Source integrity check failed: {source_key}")
    parquet = pq.ParquetFile(path)
    columns = [name for name in parquet.schema_arrow.names if name != "crop_image"]
    metadata_columns = [name for name in columns if name not in {"image", "bytes"}]
    # Shards contain at most hundreds of small text records. Read text first so
    # an entirely excluded row group never loads its large image column.
    metadata_rows = parquet.read(columns=metadata_columns, use_threads=False).to_pylist()
    chosen = {i for i, row in enumerate(metadata_rows) if selected_row(row, selection_spec)}
    source_count = len(metadata_rows)
    if source_count != parquet.metadata.num_rows:
        raise ValueError("Source text row count differs from Parquet footer")
    selected_count = 0
    log("shard_prepare_start", file=source_key, rows=source_count, selected_rows=len(chosen), ordinal_start=ordinal_start)
    # The transaction covers questions and the shard receipt together. A crash
    # rolls back the incomplete shard, leaving completed shards reusable.
    with connection:
        group_start = 0
        for group_index in range(parquet.metadata.num_row_groups):
            group_size = parquet.metadata.row_group(group_index).num_rows
            group_end = group_start + group_size
            if any(group_start <= i < group_end for i in chosen):
                row_index = group_start
                for batch in parquet.iter_batches(batch_size=batch_size, row_groups=[group_index], columns=columns, use_threads=False):
                    for row in batch.to_pylist():
                        if row_index % 100 == 0 and shutil.disk_usage(root).free < minimum_free:
                            raise OSError("Persistent disk reached reserved headroom during preparation")
                        if row_index in chosen:
                            record, payload = normalize_row(row, entry, ordinal_start + row_index, row_index)
                            image = record["image"]
                            exists = connection.execute("SELECT 1 FROM images WHERE sha256=?", (image["sha256"],)).fetchone()
                            if not exists:
                                if shutil.disk_usage(root).free < minimum_free + len(payload):
                                    raise OSError("Persistent disk reached reserved free-space floor")
                                connection.execute("INSERT INTO images VALUES (?,?,?,?,?,?)",
                                                   (image["sha256"], payload, image["width"], image["height"], image["mime"], image["size"]))
                            connection.execute("INSERT INTO questions VALUES (?,?,?,?,?)",
                                               (record["id"], entry["benchmark"], ordinal_start + row_index, source_key, json_text(record)))
                            selected_count += 1
                        row_index += 1
                if row_index != group_end:
                    raise ValueError("Parquet row-group count mismatch")
            group_start = group_end
        if selected_count != len(chosen):
            raise ValueError("Selected row count mismatch")
        connection.execute("INSERT INTO prepared_shards VALUES (?,?,?,?,?,?,?)",
                           (source_key, entry["sha256"], ordinal_start, entry["benchmark"], source_count, selected_count, utc_now()))
    log("shard_prepared", file=source_key, rows=source_count, selected_rows=selected_count)
    return source_count, selected_count


def freeze_outputs(connection, root, source_manifest_sha, manifest, state, selection, selection_sha, minimum_free):
    counts = dict(connection.execute("SELECT benchmark,COUNT(*) FROM questions GROUP BY benchmark"))
    expected = {key: value["expected_count"] for key, value in selection["benchmarks"].items()}
    if counts != expected:
        raise ValueError(f"Selected counts differ from the frozen contract: {counts!r}")
    source_counts = dict(connection.execute("SELECT benchmark,SUM(rows) FROM prepared_shards GROUP BY benchmark"))
    if source_counts != {key: spec["expected_questions"] for key, spec in BENCHMARKS.items()}:
        raise ValueError(f"Full source counts do not match: {source_counts!r}")
    if file_sha256(root / "selection.json") != selection_sha:
        raise ValueError("Selection changed during preparation")
    for benchmark, spec in selection["benchmarks"].items():
        if "include_source_ids" in spec and not spec.get("filters"):
            actual_ids = {str(json.loads(row[0])["metadata"]["original_id"]) for row in connection.execute(
                "SELECT payload FROM questions WHERE benchmark=?", (benchmark,))}
            expected_ids = set(spec["include_source_ids"]) - set(spec.get("exclude_source_ids", []))
            if actual_ids != expected_ids:
                raise ValueError(f"Selected source ID set is incomplete for {benchmark}")
    export_bytes = connection.execute("SELECT SUM(length(CAST(payload AS BLOB))+1) FROM questions").fetchone()[0] or 0
    if shutil.disk_usage(root).free < minimum_free + export_bytes + 16 * 1024 * 1024:
        raise OSError("Persistent disk lacks reserved headroom plus JSONL export space before freezing")
    checks = connection.execute("PRAGMA quick_check").fetchall()
    if checks != [("ok",)]:
        raise ValueError(f"SQLite integrity failure: {checks!r}")
    output = root / "questions.jsonl"
    temporary = root / "questions.jsonl.tmp"
    digest = hashlib.sha256()
    grouped = {}
    with temporary.open("wb") as stream:
        for benchmark in BENCHMARKS:
            groups = collections.Counter()
            for (payload,) in connection.execute("SELECT payload FROM questions WHERE benchmark=? ORDER BY ordinal", (benchmark,)):
                data = (payload + "\n").encode("utf-8")
                stream.write(data)
                digest.update(data)
                record = json.loads(payload)
                meta = record["metadata"]
                groups[json_text({k: meta.get(k) for k in ("category", "l2-category", "cycle_category", "question_type") if k in meta})] += 1
            grouped[benchmark] = dict(groups)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, output)
    image_count, image_bytes = connection.execute("SELECT COUNT(*),SUM(size) FROM images").fetchone()
    connection.commit()
    database = root / "images.sqlite"
    code = {name: file_sha256(Path(__file__).with_name(name)) for name in ("sources.py", "download.py", "prepare.py")}
    frozen = {"schema_version": SCHEMA_VERSION, "frozen_at": utc_now(), "source_manifest_sha256": source_manifest_sha,
              "selection_sha256": selection_sha, "source_counts": source_counts,
              "questions": {"path": "questions.jsonl", "sha256": digest.hexdigest(), "size": output.stat().st_size,
                            "count": sum(counts.values()), "counts": counts},
              "images": {"path": "images.sqlite", "sha256": file_sha256(database), "size": database.stat().st_size,
                         "count": image_count, "unique_payload_bytes": image_bytes, "table": "images", "key": "sha256", "blob_column": "data"},
              "source_files": len(manifest["files"]), "group_counts": grouped, "preparation_code_sha256": code,
              "policies": {"ordinal": "zero-based within benchmark; source_manifest file order, then original Parquet row order",
                           "image": "original encoded bytes; SHA256 deduplication; full Zoom image only; no resize or re-encode",
                           "question": "original question/text/query field preserved verbatim, including any embedded choices",
                           "options": "metadata.option_lines retains original order and duplicate labels; options dictionary keeps first label occurrence",
                           "duplicates": "all selected source rows retained; no Question_id deduplication; original full-source ordinals preserved",
                           "selection": "frozen selection.json only; excluded rows counted but images not stored",
                           "freeze": "dataset_manifest.json is the commit marker; prepare_state complete is written last"}}
    # Consumers must wait for this manifest and the complete state, rather than
    # treating a visible intermediate questions.jsonl as a completed freeze.
    atomic_json(root / "dataset_manifest.json", frozen)
    state.update(status="complete", completed_at=utc_now(), counts=counts, source_counts=source_counts,
                 selection_sha256=selection_sha, questions_sha256=digest.hexdigest(), images=image_count)
    atomic_json(root / "prepare_state.json", state)
    log("prepare_complete", counts=counts, images=image_count, unique_payload_bytes=image_bytes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--wait-selection", action="store_true")
    parser.add_argument("--minimum-free-gib", type=float, default=15.0)
    parser.add_argument("--available-only", action="store_true", help="Prepare verified source prefixes now; freeze only when all 87 shards exist")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    args.root.mkdir(parents=True, exist_ok=True)
    if args.root.resolve().name == "highres_v1":
        parser.error("The cancelled full-suite root must not be prepared as a perception subset")
    selection_path = args.selection or args.root / "selection.json"
    if selection_path.resolve() != (args.root / "selection.json").resolve():
        parser.error("Selection must be the new root's frozen selection.json")
    lock = (args.root / "prepare.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    while True:
        ready = selection_path.exists() and json.loads(selection_path.read_text()).get("ready") is True
        if ready:
            break
        atomic_json(args.root / "prepare_state.json", {"pid": os.getpid(), "status": "waiting_for_selection", "updated_at": utc_now()})
        if not args.wait_selection:
            raise SystemExit(75)
        time.sleep(15)
    selection, selection_sha = read_selection(selection_path)
    manifest = load_manifest(args.root)
    source_sha = file_sha256(args.root / "source_manifest.json")
    frozen_path = args.root / "dataset_manifest.json"
    if frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if (frozen.get("selection_sha256") != selection_sha or frozen["source_manifest_sha256"] != source_sha
                or file_sha256(args.root / "questions.jsonl") != frozen["questions"]["sha256"]):
            raise ValueError("Existing frozen dataset is inconsistent")
        if (args.root / "images.sqlite").stat().st_size != frozen["images"]["size"]:
            raise ValueError("Existing frozen image database size changed")
        atomic_json(args.root / "prepare_state.json", {"pid": os.getpid(), "status": "complete", "verified_at": utc_now(),
                    "counts": frozen["questions"]["counts"], "source_counts": frozen["source_counts"], "selection_sha256": selection_sha,
                    "questions_sha256": frozen["questions"]["sha256"], "images": frozen["images"]["count"]})
        log("already_frozen", questions=frozen["questions"]["count"])
        return
    state = {"pid": os.getpid(), "started_at": utc_now(), "status": "preparing", "completed_shards": 0,
             "current_source": None, "selection_sha256": selection_sha}
    atomic_json(args.root / "prepare_state.json", state)
    connection = None
    try:
        connection = open_database(args.root / "images.sqlite", source_sha, selection_sha)
        ordinals = collections.Counter()
        selected_counts = collections.Counter()
        blocked = set()
        for entry in manifest["files"]:
            benchmark = entry["benchmark"]
            path = args.root / entry["local_path"]
            if benchmark in blocked:
                continue
            if not path.exists():
                if not args.available_only:
                    raise FileNotFoundError(f"Missing verified source {entry['local_path']}")
                blocked.add(benchmark)
                continue
            state["current_source"] = entry["local_path"]
            atomic_json(args.root / "prepare_state.json", state)
            source_count, selected_count = prepare_shard(connection, args.root, entry, ordinals[benchmark], args.batch_size,
                                                        selection["benchmarks"][benchmark], int(args.minimum_free_gib * 2 ** 30))
            ordinals[benchmark] += source_count
            selected_counts[benchmark] += selected_count
            state["completed_shards"] += 1
            state["source_counts"] = dict(ordinals)
            state["counts"] = dict(selected_counts)
            state["updated_at"] = utc_now()
            atomic_json(args.root / "prepare_state.json", state)
        if blocked:
            state.update(status="waiting_for_sources", missing_benchmarks=sorted(blocked), current_source=None)
            atomic_json(args.root / "prepare_state.json", state)
            log("waiting_for_sources", counts=dict(selected_counts), source_counts=dict(ordinals), benchmarks=sorted(blocked))
        else:
            state.update(status="freezing", current_source=None)
            atomic_json(args.root / "prepare_state.json", state)
            freeze_outputs(connection, args.root, source_sha, manifest, state, selection, selection_sha,
                           int(args.minimum_free_gib * 2 ** 30))
    except Exception as error:
        state.update(status="failed", error_type=type(error).__name__, error=str(error), updated_at=utc_now())
        atomic_json(args.root / "prepare_state.json", state)
        raise
    finally:
        if connection is not None:
            connection.close()


if __name__ == "__main__":
    main()
