"""Deterministic, content-dependent RL rewards for the three-teacher experiment.

No model judge, KL, entropy or format-only bonus. The final answer must exist;
truncated rollouts receive zero. Gold JSON is produced by data_pipeline.
VQA consensus follows the leave-one-annotator-out evaluation definition.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from fractions import Fraction
from functools import lru_cache

NUMBER_WORDS = dict(zip(
    ('none','zero','one','two','three','four','five','six','seven','eight','nine','ten'),
    ('0','0','1','2','3','4','5','6','7','8','9','10')))
SPECIAL = re.compile(r'<\|[^>]+\|>')


def final_region(text: str, require_think_close: bool = False) -> str:
    text = SPECIAL.sub('', str(text)).strip()
    if '</think>' in text:
        return text.rsplit('</think>', 1)[1].strip()
    if '<think>' in text or require_think_close:
        return ''
    return text


def last_boxed(text: str) -> str | None:
    result = None
    for match in re.finditer(r'\\boxed\s*\{', text):
        # An unfinished later answer must not fall back to an earlier guess.
        result = None
        start, depth = match.end(), 1
        for pos in range(start, len(text)):
            char = text[pos]
            if char in '{}':
                # Escaped literal braces do not delimit the boxed expression.
                slashes, cursor = 0, pos - 1
                while cursor >= 0 and text[cursor] == '\\':
                    slashes += 1; cursor -= 1
                if slashes % 2:
                    continue
                depth += 1 if char == '{' else -1
                if depth == 0:
                    result = text[start:pos].strip()
                    break
    return result


def normalize_answer(value) -> str:
    value = str(value).strip().lower().replace('\n', ' ').replace('\t', ' ')
    value = re.sub(r'\\(?:text|mathrm)\{([^{}]*)\}', r'\1', value)
    value = value.strip('$ ')
    value = re.sub(r'(?<=\d),(?=\d)', '', value)
    value = re.sub(r'(?<!\d)\.|\.(?!\d)', '', value)
    value = re.sub(r'[;/:\[\]"{}()=+\\_<>@`,?!-]', ' ', value)
    words = [NUMBER_WORDS.get(w, w) for w in value.split() if w not in {'a','an','the'}]
    # Equivalent apostrophe spellings, without changing arbitrary words.
    return ' '.join(words).replace("'", '').replace('’', '')


def number(value) -> float | None:
    value = str(value).strip().strip('$')
    if ',' in value:
        # A comma may group thousands; it must not collapse a list like 1,2.
        if not re.fullmatch(r'[+-]?\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\\?%)?', value):
            return None
        value = value.replace(',', '')
    percent = value.endswith('%') or value.endswith(r'\%')
    if percent:
        value = value.removesuffix(r'\%').removesuffix('%').strip()
    try:
        result = float(Fraction(value))
        if percent:
            result /= 100
        return result if math.isfinite(result) else None
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def exact_canonical(value) -> str:
    value = unicodedata.normalize('NFKC', str(value)).strip().casefold()
    value = re.sub(r'^\\(?:text|mathrm)\{([^{}]*)\}$', r'\1', value)
    value = ' '.join(value.split())
    return NUMBER_WORDS.get(value, value)


def exact_equal(pred: str, gold: str) -> bool:
    pred, gold = exact_canonical(pred), exact_canonical(gold)
    if not pred or not gold:
        return False
    a, b = number(pred), number(gold)
    if a is not None and b is not None:
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return pred == gold


@lru_cache(maxsize=4096)
def _parse_math(value: str):
    from math_verify import LatexExtractionConfig, parse
    return parse('$' + value + '$',
                 extraction_config=[LatexExtractionConfig(boxed_match_priority=-1)],
                 fallback_mode='no_fallback', extraction_mode='first_match')


def single_math_expression(value: str) -> str | None:
    value = str(value).strip()
    for left, right in [('$$', '$$'), ('$', '$'), (r'\(', r'\)'), (r'\[', r'\]')]:
        if value.startswith(left) and value.endswith(right):
            value = value[len(left):-len(right)].strip()
            break
    # The verifier can accept only the prefix before a LaTeX row break. These
    # tasks request a single boxed answer, so reject segmented math rather than
    # crediting its first segment. Also prevent nested extraction anchors.
    if not value or len(value) > 4096 or any(x in value for x in (
            '$', r'\\', r'\[', r'\]', r'\(', r'\)', r'\boxed', r'\begin', r'\end')):
        return None
    return value


def math_equal(pred: str, gold: str) -> bool:
    # ViRL's trusted answer field itself contains a boxed reference. Unwrap
    # references before comparison; predictions are already extracted by the
    # final-answer gate, and must never gain partial-expression extraction here.
    gold = str(gold)
    boxed_gold = last_boxed(gold)
    if boxed_gold is not None:
        gold = boxed_gold
    pred = str(pred).replace('°', r'^\circ')
    gold = gold.replace('°', r'^\circ')
    # Exact complete reference matches remain valid for structured answers
    # (e.g. matrices or multi-equation gold) outside the symbolic parser subset.
    raw_pred, raw_gold = str(pred).strip(), str(gold).strip()
    if raw_pred and raw_pred.casefold() == raw_gold.casefold():
        return True
    pred, gold = single_math_expression(pred), single_math_expression(gold)
    if pred is None or gold is None:
        return False
    if pred.casefold() == gold.casefold():
        return True
    a, b = number(pred), number(gold)
    if a is not None and b is not None:
        return math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-8)
    # Missing verifier is a configuration error, never silently a wrong answer.
    from math_verify import verify
    try:
        return bool(verify(_parse_math(gold), _parse_math(pred)))
    except (ValueError, TypeError, TimeoutError, ArithmeticError):
        return False


def vqa_consensus(pred: str, answers: list[str]) -> float:
    pred = normalize_answer(pred)
    refs = [normalize_answer(a) for a in answers]
    if not refs:
        raise ValueError('VQA gold has no answers')
    if len(refs) == 1:
        return float(pred == refs[0])
    # Keep repeated annotator answers: their multiplicity is the consensus.
    return sum(min(sum(a == pred for j, a in enumerate(refs) if j != k) / 3, 1)
               for k in range(len(refs))) / len(refs)


def anls(pred: str, answers: list[str]) -> float:
    pred = pred.strip().lower()
    result = 0.0
    for gold in answers:
        gold = str(gold).strip().lower()
        previous = list(range(len(gold) + 1))
        for row, ca in enumerate(pred, 1):
            current = [row]
            for col, cb in enumerate(gold, 1):
                current.append(min(current[-1]+1, previous[col]+1,
                                   previous[col-1]+int(ca != cb)))
            previous = current
        similarity = 1 - previous[-1] / max(len(pred), len(gold), 1)
        result = max(result, similarity if similarity > .5 else 0.0)
    return result


def relaxed_numeric(pred: str, answers: list[str]) -> float:
    for gold in answers:
        if str(pred).strip().lower() == str(gold).strip().lower():
            return 1.0
        a, b = number(pred), number(gold)
        if a is not None and b is not None:
            if abs(a-b) <= max(.05 * abs(b), 1e-8):
                return 1.0
    return 0.0


def bbox_iou(pred: str, gold: list[float]) -> float:
    try:
        box = json.loads(pred)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    if not isinstance(box, list) or len(box) != 4:
        return 0.0
    if any(isinstance(x, bool) or not isinstance(x, (float, int)) or
           not math.isfinite(x) or not 0 <= x <= 1000 for x in box):
        return 0.0
    x1,y1,x2,y2 = box
    gx1,gy1,gx2,gy2 = gold
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = max(0, min(x2,gx2)-max(x1,gx1))*max(0, min(y2,gy2)-max(y1,gy1))
    union = (x2-x1)*(y2-y1)+(gx2-gx1)*(gy2-gy1)-inter
    return inter/union if union > 0 else 0.0


def strict_json(text: str):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def reject_constant(value):
        raise ValueError('Non-JSON constant')
    return json.loads(text, object_pairs_hook=unique_pairs, parse_constant=reject_constant)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    gold = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if not isinstance(gold, dict) or not gold.get('task'):
        raise ValueError(f'Invalid ground-truth schema for {data_source}')
    info = extra_info or {}
    result = {'score': 0.0, 'accuracy': 0.0, 'format_valid': 0.0,
              'truncated': float(bool(info.get('truncated', False)))}
    if result['truncated']:
        return result
    final = final_region(solution_str, bool(info.get('require_think_close', False)))
    if not final:
        return result
    task, answers = gold['task'], gold.get('answers', [])
    if gold.get('format') == 'json' or task == 'json':
        try:
            value = strict_json(final)
        except (ValueError, TypeError):
            return result
        key = gold.get('json_key', 'answer')
        if not isinstance(value, dict) or set(value) != {key}:
            return result
        pred = value[key]
        if not isinstance(pred, str):
            return result
    else:
        pred = last_boxed(final)
        if pred is None or not pred:
            return result
    result['format_valid'] = 1.0
    if task == 'json':
        task = gold.get('content_task', 'exact')
    if task == 'bbox':
        score = bbox_iou(pred, gold['bbox'])
        accuracy = float(score >= .5)
    elif task == 'vqa':
        score = vqa_consensus(pred, answers); accuracy = score
    elif task == 'math':
        score = float(any(math_equal(pred, str(a)) for a in answers)); accuracy = score
    elif task == 'anls':
        score = anls(pred, answers); accuracy = score
    elif task == 'relaxed_numeric':
        score = relaxed_numeric(pred, answers); accuracy = score
    elif task in {'exact', 'json'}:
        score = float(any(exact_equal(pred, a) for a in answers))
        accuracy = score
    else:
        raise ValueError(f'Unknown reward task {task!r} for {data_source}')
    result.update(score=float(score), accuracy=float(accuracy))
    return result


def validate_runtime():
    import math_verify
    if not math_equal(r'\frac{1}{2}', '0.5'):
        raise RuntimeError('Math-Verify failed fraction equivalence preflight')
    return {'math_verify': getattr(math_verify, '__version__', 'installed')}
