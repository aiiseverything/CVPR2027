#!/usr/bin/env python3
"""Evaluate a frozen suite on an already-running LOCAL vLLM endpoint; resumable."""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import fcntl
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

from common import HERE, PROTOCOL, SCORING_PROTOCOL, digest, file_sha, load_suite, message_content, read_jsonl, score_prediction, image_size_for_count, local_lock_path


def request_json(url, payload=None, timeout=60):
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def infer(row, args):
    images = []
    for path, expected in zip(row["images"], row["image_file_sha256"]):
        if file_sha(path) != expected:
            raise ValueError(f"Evaluation image changed: {path}")
        blob = base64.b64encode(Path(path).read_bytes()).decode()
        mime = mimetypes.guess_type(path)[0] or "image/jpeg"
        images.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{blob}"}})
    payload = {
        "model": args.model, "messages": [{"role": "user", "content": message_content(row["prompt"], images)}],
        "temperature": PROTOCOL["temperature"], "top_p": PROTOCOL["top_p"], "max_tokens": PROTOCOL["max_tokens"],
        "seed": PROTOCOL["seed"], "chat_template_kwargs": {"enable_thinking": PROTOCOL["enable_thinking"]},
        "mm_processor_kwargs": {"size": image_size_for_count(len(images))},
    }
    t0 = time.monotonic()
    error = None
    for attempt in range(args.retries):
        try:
            result = request_json(args.base_url.rstrip("/") + "/chat/completions", payload, args.timeout)
            choice = result["choices"][0]
            message = choice["message"]
            content = message.get("content") or ""
            reasoning = message.get("reasoning_content") or message.get("reasoning")
            # Some vLLM parsers return thinking separately; judge final content only.
            usage = result.get("usage") or {}
            prompt_n = usage.get("prompt_tokens")
            if prompt_n is None or prompt_n != row["prompt_tokens"]:
                raise ValueError(f"Prompt-token mismatch: server={prompt_n}, prepared={row['prompt_tokens']}; stop instead of changing protocol")
            if prompt_n > PROTOCOL["max_prompt_tokens"]:
                raise ValueError("Prompt exceeds fixed budget")
            return score_prediction(row, {
                "id": row["id"], "row_sha256": digest(row), "model": args.name,
                "raw_response": content, "reasoning": reasoning,
                "finish_reason": choice.get("finish_reason"), "usage": usage,
                "elapsed_seconds": time.monotonic() - t0, "error": None,
            })
        except urllib.error.HTTPError as exc:
            try:
                detail = json.loads(exc.read()).get("error", {}).get("message", "")
            except (ValueError, AttributeError):
                detail = ""
            error = f"HTTP {exc.code}: {str(detail)[:400]}"
            if 400 <= exc.code < 500 and exc.code != 429:
                break
            if attempt + 1 < args.retries:
                time.sleep(min(2 ** attempt, 10))
        except (urllib.error.URLError, TimeoutError, ConnectionError, KeyError, json.JSONDecodeError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
            if attempt + 1 < args.retries:
                time.sleep(min(2 ** attempt, 10))
    return {"id": row["id"], "row_sha256": digest(row), "model": args.name, "error": error, "correct": False, "parse_success": False, "elapsed_seconds": time.monotonic() - t0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", type=Path, default=HERE / "suite.jsonl")
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True, help="Served model name, checked against /models")
    ap.add_argument("--name", required=True, choices=["baseline", "generalist", "perception", "reasoning"])
    ap.add_argument("--output-root", type=Path, default=HERE / "results")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="Smoke only: writes to a separate smoke directory")
    ap.add_argument("--import-results", type=Path, help="Import only same-model/protocol cached rows that match this expanded suite")
    args = ap.parse_args()
    url = urllib.parse.urlparse(args.base_url)
    if url.scheme != "http" or url.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise SystemExit("Only local HTTP inference endpoints are accepted; no external judge or service")
    if args.concurrency < 1 or args.retries < 1:
        raise SystemExit("concurrency and retries must be positive")
    rows, suite_meta = load_suite(args.suite)
    served = request_json(args.base_url.rstrip("/") + "/models")
    if args.model not in {m["id"] for m in served["data"]}:
        raise SystemExit(f"Expected model {args.model} is not served at endpoint")
    if args.limit:
        # Cover multiple benchmarks and a multi-image example before a long run.
        representatives = {}
        for row in rows:
            key = (row["benchmark"], min(len(row["images"]), 2))
            representatives.setdefault(key, row)
        chosen = list(representatives.values())[:args.limit]
        chosen_ids = {r["id"] for r in chosen}
        rows = (chosen + [r for r in rows if r["id"] not in chosen_ids])[:args.limit]
    out_dir = args.output_root / ("smoke" if args.limit else "full") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    inference_lock = local_lock_path(out_dir, 'inference').open("a")
    try:
        fcntl.flock(inference_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"Another inference client already owns {out_dir}")
    out_file = out_dir / "predictions.jsonl"
    run_info = {
        "suite_sha256": suite_meta["suite_sha256"], "protocol_sha256": suite_meta["protocol_sha256"],
        "protocol": PROTOCOL, "name": args.name, "served_model": args.model,
        "suite": str(args.suite.resolve()), "n_requested": len(rows), "smoke": bool(args.limit),
        "scorer_source_sha256": file_sha(HERE / "common.py"),
        "scoring_protocol": SCORING_PROTOCOL, "scoring_protocol_sha256": digest(SCORING_PROTOCOL),
    }
    info_file = out_dir / "run.json"
    if info_file.exists():
        old_info = json.loads(info_file.read_text())
        scoring_keys = {"scorer_source_sha256", "scoring_protocol", "scoring_protocol_sha256"}
        if {k: v for k, v in old_info.items() if k not in scoring_keys} != {k: v for k, v in run_info.items() if k not in scoring_keys}:
            raise SystemExit("Output directory already contains a different inference protocol/model/suite")
    info_file.write_text(json.dumps(run_info, ensure_ascii=False, indent=2) + "\n")
    if args.import_results and not out_file.exists():
        old_info = json.loads((args.import_results / "run.json").read_text())
        for key in ["protocol_sha256", "name", "served_model"]:
            if old_info[key] != run_info[key]:
                raise SystemExit(f"Cannot import predictions with differing {key}")
        row_map = {r["id"]: r for r in rows}
        compatible = {}
        for r in read_jsonl(args.import_results / "predictions.jsonl"):
            if r["id"] in row_map and r.get("row_sha256") == digest(row_map[r["id"]]) and not r.get("error"):
                compatible[r["id"]] = score_prediction(row_map[r["id"]], r)
        out_file.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in compatible.values()))
        print(f"Imported {len(compatible)} unchanged predictions from {args.import_results}", flush=True)
    previous = {r["id"]: r for r in read_jsonl(out_file)}
    for row in rows:
        if row["id"] in previous and previous[row["id"]].get("row_sha256") != digest(row):
            raise SystemExit(f"Cached prediction does not match suite row: {row['id']}")
        if row["id"] in previous:
            previous[row["id"]] = score_prediction(row, previous[row["id"]])
    todo = [r for r in rows if r["id"] not in previous or previous[r["id"]].get("error")]
    print(f"{args.name}: {len(rows)} samples, {len(todo)} pending, concurrency={args.concurrency}", flush=True)
    # One synchronous sample validates the exact processor/chat-template protocol before parallel work.
    if todo:
        first = infer(todo[0], args)
        with out_file.open("a") as f:
            f.write(json.dumps(first, ensure_ascii=False) + "\n")
        if first.get("error"):
            print("First sample had an API error; cached successes are preserved for retry", flush=True)
            raise SystemExit(2)
        previous[first["id"]] = first
        todo = todo[1:]
    with out_file.open("a") as f, concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {pool.submit(infer, r, args): r["id"] for r in todo}
        completed = len(rows) - len(todo)
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
            f.flush()
            previous[result["id"]] = result
            completed += 1
            if completed % 10 == 0 or completed == len(rows):
                print(f"{args.name}: {completed}/{len(rows)}", flush=True)
    results = [previous.get(r["id"], {}) for r in rows]
    errors = sum(bool(r.get("error")) or not r for r in results)
    print(json.dumps({"name": args.name, "n": len(rows), "api_errors": errors, "extraction_failures": sum(not r.get("parse_success", False) for r in results), "finish_reasons": dict(Counter(r.get("finish_reason") for r in results))}), flush=True)
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
