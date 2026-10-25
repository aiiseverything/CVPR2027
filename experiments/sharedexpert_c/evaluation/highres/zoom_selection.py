#!/usr/bin/env python3
"""Acquire small ZoomBench text annotations and build an auditable selection.

No model, GPU, image payload, or prediction is read. Selection is based solely on
question and option text; source answers are retained only for provenance.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import urllib.request

HERE = Path(__file__).resolve().parent
DEFAULT = HERE.parent / "data/highres_perception_v1/zoom_selection.json"
REVISION = "b788097e57d30510c6877824833234a73bf80d25"
REPOSITORY = "inclusionAI/ZoomBench"


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode()).hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def fetch_json(url):
    error = None
    for attempt in range(5):
        try:
            opener = urllib.request.build_opener() if attempt % 2 == 0 else urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(url, timeout=25) as response:
                return json.load(response)
        except Exception as exc:
            error = exc
            time.sleep(min(attempt + 1, 5))
    raise RuntimeError(f"Annotation request failed: {type(error).__name__}")


def fetch(path):
    document = json.loads(path.read_text()) if path.exists() else {
        "status": "draft_pending_review", "source": {"repository": REPOSITORY,
        "revision": REVISION, "split": "test", "n": 845,
        "retrieval": "Hugging Face datasets-server /rows; image payloads not fetched"},
        "annotations": []}
    if document["status"] == "complete":
        raise ValueError("Selection is frozen; refusing to refresh annotations")
    if document["source"]["revision"] != REVISION:
        raise ValueError("Existing annotation revision differs")
    api = f"https://huggingface.co/api/datasets/{REPOSITORY}"
    if fetch_json(api)["sha"] != REVISION:
        raise ValueError("The public viewer is no longer at the pinned revision")
    retained = {r["row_index"]: r for r in document["annotations"]}
    def page(offset):
        url = f"https://datasets-server.huggingface.co/rows?dataset=inclusionAI%2FZoomBench&config=default&split=test&offset={offset}&length=100"
        data = fetch_json(url)
        if data["num_rows_total"] != 845 or data.get("partial") is not False:
            raise ValueError("Incomplete public annotations")
        rows = [{"row_index": item["row_idx"], **{k: v for k, v in item["row"].items()
                 if k not in {"image", "crop_image"}}} for item in data["rows"]]
        if {r["row_index"] for r in rows} != set(range(offset, min(offset + 100, 845))):
            raise ValueError("Annotation page does not match requested indices")
        return rows
    missing = [offset for offset in range(0, 845, 100)
               if not set(range(offset, min(offset + 100, 845))) <= set(retained)]
    errors = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(page, offset): offset for offset in missing}
        for future in as_completed(futures):
            offset = futures[future]
            try:
                for r in future.result():
                    retained[r["row_index"]] = r
            except Exception as exc:
                errors.append({"offset": offset, "error": str(exc)})
            document["annotations"] = [retained[i] for i in sorted(retained)]
            document["source"]["retrieved_at"] = datetime.now(timezone.utc).isoformat()
            document["source"]["annotations_sha256"] = canonical_sha(document["annotations"])
            save(path, document)
            print(json.dumps({"offset": offset, "annotations_saved": len(retained), "page_failed": bool(errors and errors[-1]["offset"] == offset)}), flush=True)
    if errors:
        raise RuntimeError(f"Incomplete text annotations; retry --fetch: {errors}")
    if set(retained) != set(range(845)) or len({r["id"] for r in retained.values()}) != 845:
        raise ValueError("Annotation index or ID set is incomplete")
    if fetch_json(api)["sha"] != REVISION:
        raise ValueError("Viewer source revision changed during retrieval")
    document["source"]["revision_verified_before_and_after_fetch"] = True
    save(path, document)
    print(json.dumps({"text_fetch_complete": True, "rows": 845, "path": str(path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    if args.fetch:
        fetch(args.output)
    else:
        parser.error("Choose --fetch; reviewed selection builder is not yet enabled")


if __name__ == "__main__":
    main()
