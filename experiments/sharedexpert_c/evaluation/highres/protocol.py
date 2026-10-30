"""Frozen perception subsets under a separate full-image high-resolution protocol."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVALUATION = HERE.parent
EXPERIMENT = EVALUATION.parent
DATA = EVALUATION / 'data' / 'highres_perception_v1'
RESULTS = EVALUATION / 'results_highres_perception_v1'
PROTOCOL = {
    'version': 'sharedexpert-highres-perception-subsets-v1',
    'enable_thinking': True, 'temperature': 0, 'top_p': 1,
    'seed': 20261019, 'max_tokens': 8192,
    'max_prompt_tokens': 40960, 'max_model_len': 49152,
    'max_images': 1,
    'image_size': {'shortest_edge': 4096, 'longest_edge': 33554432},
    'image_view': 'full original image only; no privileged crops',
    'answer_instruction': 'Put only your final {kind} inside <answer>...</answer>.',
}
EXPECTED = {'vstar': 191, 'zoom': 845, 'hrbench_4k': 800, 'hrbench_8k': 800,
            'mme_rw_en': 23609, 'mme_rw_cn': 5917}
MODEL_PATHS = {
    'baseline': EXPERIMENT / 'models' / 'Qwen3.5-4B',
    'perception-step100': EXPERIMENT / 'models' / 'Qwen3.5-4B-perception-step100',
    'reasoning-step100': EXPERIMENT / 'models' / 'Qwen3.5-4B-reasoning-step100',
    'perception-v2-step100': EXPERIMENT / 'runs' / 'perception_v2' / 'hf' / 'global_step_100',
}
# New exports are registered against a private copy of the frozen metadata.
# Keep MODEL_PATHS unchanged: freeze.py and the historical queue own four models.
REGISTERED_MODEL_NAMES = ('perception-v3-step100',)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def prompt_for(row):
    prompt = row['question'].strip()
    options = row['options']
    if options and not row.get('metadata', {}).get('options_embedded'):
        lines = row.get('metadata', {}).get('option_lines')
        prompt += '\n' + '\n'.join(lines or [f'{key}. {value}' for key, value in options.items()])
    kind = 'option letter' if options else 'numeric answer'
    return prompt + '\n' + PROTOCOL['answer_instruction'].format(kind=kind)


def content_for(prompt, image):
    return [image, {'type': 'text', 'text': prompt}]


def prompt_tokens(processor, row):
    messages = [{'role': 'user', 'content': content_for(row['prompt'], {'type': 'image'})}]
    text = processor.apply_chat_template(messages, tokenize=False,
        add_generation_prompt=True, enable_thinking=PROTOCOL['enable_thinking'])
    # This installed Qwen2VLImageProcessor helper ignores size=, unlike preprocessing.
    # Pass the equivalent min/max pixel bounds explicitly to the count helper.
    info = processor._get_num_multimodal_tokens(
        image_sizes=[(row['image']['height'], row['image']['width'])],
        min_pixels=PROTOCOL['image_size']['shortest_edge'],
        max_pixels=PROTOCOL['image_size']['longest_edge'])
    visual = int(info['num_image_tokens'][0])
    return len(processor.tokenizer.encode(text, add_special_tokens=False)) + visual - 1, visual


def load_selection(root=DATA):
    selection = json.loads((Path(root) / 'selection.json').read_text())
    if selection.get('schema_version') != 'highres-selection-v1' or selection.get('ready') is not True:
        raise ValueError('Perception selection has not been frozen and approved for execution')
    specs = selection.get('benchmarks', {})
    if set(specs) != set(EXPECTED):
        raise ValueError('Selection must cover exactly the six authorized benchmarks')
    for name, spec in specs.items():
        count = spec.get('expected_count')
        if not isinstance(count, int) or not 0 < count <= EXPECTED[name]:
            raise ValueError(f'Invalid selected count for {name}')
        for condition in spec.get('filters', []):
            if set(condition) != {'field', 'op', 'value'} or condition['op'] not in {'eq', 'startswith', 'in'}:
                raise ValueError('Invalid perception subset predicate')
        for key in ('include_source_ids', 'exclude_source_ids'):
            if key in spec and (not isinstance(spec[key], list) or len(spec[key]) != len(set(spec[key]))):
                raise ValueError(f'{name}: {key} must contain unique IDs')
    return selection


def selected_counts(root=DATA):
    return {name: spec['expected_count'] for name, spec in load_selection(root)['benchmarks'].items()}


def load_suite(root=DATA):
    root = Path(root)
    meta = json.loads((root / 'suite.meta.json').read_text())
    if meta['protocol'] != PROTOCOL or meta['protocol_sha256'] != digest(PROTOCOL):
        raise ValueError('High-resolution inference protocol changed after freezing')
    path = root / 'suite.jsonl'
    if file_sha(path) != meta['suite_sha256']:
        raise ValueError('Frozen high-resolution suite checksum mismatch')
    rows = read_jsonl(path)
    from collections import Counter
    expected = selected_counts(root)
    if meta.get('selection_sha256') != file_sha(root / 'selection.json') or meta.get('counts') != expected:
        raise ValueError('Perception selection changed after freezing')
    if dict(Counter(r['benchmark'] for r in rows)) != expected:
        raise ValueError('Suite must contain every selected row from all six benchmarks')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate suite IDs')
    return rows, meta
