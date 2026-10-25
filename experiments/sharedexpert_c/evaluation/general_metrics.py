"""OCRBench v1 matching and pinned official IFEval strict/loose checks."""
import functools
import hashlib
import json
from pathlib import Path
import random
import sys
import threading

from common import SCORING_PROTOCOL

HERE = Path(__file__).resolve().parent
_IFEVAL_RANDOM_LOCK = threading.RLock()


def ifeval_seed(row):
    """Derive a model/response-independent seed from frozen instruction input."""
    identity = {"base_seed": SCORING_PROTOCOL["ifeval_randomness"]["base_seed"],
                "ifeval": row["ifeval"]}
    canonical = json.dumps(identity, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(canonical).digest(), "big")


def metric_name(row):
    if row['task'] == 'ocrbench':
        return 'ocrbench_normalized_substring_accuracy'
    if row['task'] == 'ifeval':
        return 'ifeval_strict_prompt_accuracy'
    if row['benchmark'] == 'MMBench_DEV_EN_V11':
        return 'circular_group_accuracy'
    if row['task'] == 'grounding':
        return 'accuracy_at_iou_0.5'
    return 'strict_accuracy'


@functools.lru_cache(maxsize=1)
def ifeval_library():
    manifest = json.loads((HERE / "general_sources_manifest.json").read_text())
    root = HERE / "general_sources"
    for name, expected in manifest["files"].items():
        path = root / "instruction_following_eval" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"IFEval pinned source changed: {name}")
    sys.path.insert(0, str(root))
    import nltk
    nltk.data.path.insert(0, str(root/'nltk_data'))
    import langdetect
    langdetect.DetectorFactory.seed = 0
    from instruction_following_eval import evaluation_lib
    return evaluation_lib


def final_content(response):
    if "</think>" in response:
        return response.rsplit("</think>", 1)[1].strip()
    if "<think>" in response:
        return ""
    return response.strip()


def score_general(row, response):
    content = final_content(response)
    result = {"extracted": content, "extraction": "whole_final_content",
              "parse_success": bool(content), "correct": False}
    if row["task"] == "ocrbench":
        math = row["category"] == "Handwritten Mathematical Expression Recognition"
        def normalize(text):
            text = str(text).strip().replace("\n", " ")
            return text.replace(" ", "") if math else text.lower()
        prediction = normalize(content)
        result["correct"] = bool(content) and any(
            normalize(answer) and normalize(answer) in prediction for answer in row["answers"])
    else:
        seed = ifeval_seed(row)
        # The pinned upstream checker can randomly replace unsupported symbols
        # in its instruction parameters. Preserve that implementation, but make
        # each row's fallback identical across models and strict/loose scoring.
        # run_api also scores concurrently, so saving global RNG state alone is
        # insufficient: all checker calls must hold the same lock.
        with _IFEVAL_RANDOM_LOCK:
            previous = random.getstate()
            try:
                library = ifeval_library()
                example = library.InputExample(**row["ifeval"])
                random.seed(seed)
                strict = library.test_instruction_following_strict(example, {example.prompt: content})
                random.seed(seed)
                loose = library.test_instruction_following_loose(example, {example.prompt: content})
            finally:
                random.setstate(previous)
        result.update(correct=bool(strict.follow_all_instructions),
                      loose_correct=bool(loose.follow_all_instructions),
                      strict_instructions=list(strict.follow_instruction_list),
                      loose_instructions=list(loose.follow_instruction_list),
                      ifeval_random_seed=seed)
    return result


def aggregate_general(rows, predictions):
    """Secondary metrics, reported only for complete successful benchmark runs."""
    result = {}
    for model, pred in predictions.items():
        metrics = {}
        for task in ('ifeval', 'ocrbench'):
            samples = [r for r in rows if r['task'] == task]
            if not samples or any(r['id'] not in pred or pred[r['id']].get('error') for r in samples):
                continue
            scores = [pred[r['id']] for r in samples]
            if task == 'ifeval':
                metrics['ifeval'] = {'n_prompts':len(samples)}
                for mode in ('strict','loose'):
                    values = [v for s in scores for v in s[mode+'_instructions']]
                    metrics['ifeval'][mode+'_instruction_accuracy'] = sum(values)/len(values)
                    metrics['ifeval'][mode+'_prompt_accuracy'] = sum(
                        s['correct' if mode == 'strict' else 'loose_correct'] for s in scores)/len(scores)
            else:
                metrics['ocr_categories'] = {}
                for category in sorted({r['category'] for r in samples}):
                    subset = [pred[r['id']]['correct'] for r in samples if r['category']==category]
                    metrics['ocr_categories'][category] = {'n':len(subset),'accuracy':sum(subset)/len(subset)}
        if metrics:
            result[model] = metrics
    return result
