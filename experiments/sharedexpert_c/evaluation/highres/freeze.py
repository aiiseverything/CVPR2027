#!/usr/bin/env python3
"""CPU-only checks of six perception subsets and four processors, then freeze prompts."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import sqlite3

from protocol import (DATA, EXPECTED, MODEL_PATHS, PROTOCOL, HERE, atomic_json,
                      content_for, digest, file_sha, load_suite, prompt_for,
                      prompt_tokens, read_jsonl, load_selection, selected_counts)


def freeze(root=DATA):
    root = Path(root)
    state_file = root / 'freeze_state.json'
    preference_path = root / 'skip_preflight.json'
    preference = json.loads(preference_path.read_text()) if preference_path.exists() else {}
    skip_preflight = preference.get('enabled') is True
    selection = load_selection(root)
    expected = selected_counts(root)

    def state(stage, **details):
        value = {'status': stage, 'time': datetime.now(timezone.utc).isoformat(),
                 'pid': os.getpid(), **details}
        atomic_json(state_file, value)
        print(json.dumps(value), flush=True)

    if (root / 'suite.meta.json').is_file():
        rows, meta = load_suite(root)
        if meta['questions_sha256'] != file_sha(root / 'questions.jsonl'):
            raise ValueError('Prepared questions changed after freezing')
        state('complete', rows=len(rows), suite_sha256=meta['suite_sha256'], reused=True)
        return
    preparation = json.loads((root / 'prepare_state.json').read_text())
    if preparation.get('status') != 'complete':
        raise ValueError('All six selected subsets must be prepared first')
    dataset_manifest = json.loads((root / 'dataset_manifest.json').read_text())
    if dataset_manifest.get('selection_sha256') != file_sha(root / 'selection.json'):
        raise ValueError('Prepared data belongs to a different perception selection')
    state('building_inputs' if skip_preflight else 'validating_questions', preflight_skipped=skip_preflight)
    rows = read_jsonl(root / 'questions.jsonl')
    if dict(Counter(r['benchmark'] for r in rows)) != expected:
        raise ValueError('Prepared question counts differ from frozen perception selection')
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate prepared question IDs')
    from scoring import validate_row, SCORING_PROTOCOL
    image_meta = {}
    for row in rows:
        if not skip_preflight:
            validate_row(row)
        im = row['image']
        if not row['question'] or not str(row['answer']).strip():
            raise ValueError(f'Missing question/answer: {row["id"]}')
        if im['width'] <= 0 or im['height'] <= 0 or im['size'] <= 0:
            raise ValueError(f'Invalid image: {row["id"]}')
        if im['sha256'] in image_meta and image_meta[im['sha256']] != im:
            raise ValueError('Inconsistent image metadata')
        image_meta[im['sha256']] = im
        row['prompt'] = prompt_for(row)
    db = None
    if not skip_preflight:
        db = sqlite3.connect(f'file:{root / "images.sqlite"}?mode=ro', uri=True)
        stored = {sha: size for sha, size in db.execute('SELECT sha256,length(data) FROM images')}
        if any(stored.get(sha) != im['size'] for sha, im in image_meta.items()):
            raise ValueError('Missing or truncated packed images')
    # Imports and all image checks below are CPU only. No model weights are loaded.
    from transformers import AutoConfig, AutoProcessor
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    models = {}
    baseline = None
    for name, model in MODEL_PATHS.items():
        state('building_prompt_metadata' if skip_preflight else 'validating_processor', model=name,
              preflight_skipped=skip_preflight)
        config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        context = getattr(getattr(config, 'text_config', config), 'max_position_embeddings')
        if context < PROTOCOL['max_model_len']:
            raise ValueError(f'{name} cannot support the frozen context length')
        processor = (baseline if skip_preflight and baseline is not None else
                     AutoProcessor.from_pretrained(model, local_files_only=True, trust_remote_code=False))
        weights = sorted(model.glob('*.safetensors'))
        if not weights or any(p.stat().st_size < 1024 for p in weights):
            raise ValueError(f'Missing model weights: {name}')
        weight_index = model / 'model.safetensors.index.json'
        if weight_index.is_file():
            listed = set(json.loads(weight_index.read_text())['weight_map'].values())
            if not listed <= {p.name for p in weights}:
                raise ValueError(f'Incomplete weight shards: {name}')
        for row in (rows if baseline is None or not skip_preflight else []):
            count, visual = prompt_tokens(processor, row)
            if count > PROTOCOL['max_prompt_tokens']:
                raise ValueError(f'{name}: prompt {row["id"]} exceeds budget ({count})')
            if 'prompt_tokens' in row and (row['prompt_tokens'], row['visual_tokens']) != (count, visual):
                raise ValueError(f'Processor token mismatch for {name}: {row["id"]}')
            row['prompt_tokens'], row['visual_tokens'] = count, visual
        metadata = {p.name: file_sha(p) for p in model.iterdir()
                    if p.suffix in {'.json', '.jinja'} and p.is_file()}
        models[name] = {'path': str(model), 'metadata_sha256': metadata,
                        'context_capacity': context, 'processor_class': type(processor).__name__,
                        'processor_verification': 'skipped_at_user_request' if skip_preflight else 'checked',
                        'weight_files': [{'name': p.name, 'size': p.stat().st_size,
                                          'mtime_ns': p.stat().st_mtime_ns} for p in weights]}
        if baseline is None:
            baseline = processor
    state('writing_suite' if skip_preflight else 'validating_actual_cpu_preprocessing', preflight_skipped=skip_preflight)
    # Compare the count-only path against the installed real processor, including
    # one 8K row. These are CPU preparations, not separate GPU inference runs.
    chosen = [] if skip_preflight else [min(rows, key=lambda r: r['visual_tokens']),
              max((r for r in rows if r['benchmark'] == 'hrbench_4k'), key=lambda r: r['visual_tokens']),
              max((r for r in rows if r['benchmark'] == 'hrbench_8k'), key=lambda r: r['visual_tokens'])]
    checked = []
    for row in chosen:
        raw = db.execute('SELECT data FROM images WHERE sha256=?', (row['image']['sha256'],)).fetchone()[0]
        if hashlib.sha256(raw).hexdigest() != row['image']['sha256']:
            raise ValueError('Packed image checksum mismatch')
        with Image.open(io.BytesIO(raw)) as image:
            image = image.convert('RGB')
            messages = [{'role': 'user', 'content': content_for(row['prompt'], {'type': 'image'})}]
            text = baseline.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=PROTOCOL['enable_thinking'])
            actual = baseline(text=[text], images=[image], size=PROTOCOL['image_size'],
                              return_tensors='pt')
            count = int(actual['input_ids'].shape[-1])
            if count != row['prompt_tokens']:
                raise ValueError(f'Actual CPU processor count differs for {row["id"]}: {count}')
            checked.append({'id': row['id'], 'actual_prompt_tokens': count,
                            'image_grid_thw': actual['image_grid_thw'].tolist()})
            del actual, image, raw
            gc.collect()
    if db is not None:
        db.close()
    state('freezing_suite')
    tmp = root / 'suite.jsonl.tmp'
    with tmp.open('w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(root / 'suite.jsonl')
    stats = {}
    for benchmark in expected:
        group = [r for r in rows if r['benchmark'] == benchmark]
        stats[benchmark] = {'n': len(group),
            'max_prompt_tokens': max(r['prompt_tokens'] for r in group),
            'max_visual_tokens': max(r['visual_tokens'] for r in group),
            'images_over_pixel_budget': sum(r['image']['width'] * r['image']['height'] >
                                            PROTOCOL['image_size']['longest_edge'] for r in group)}
    meta = {'protocol': PROTOCOL, 'protocol_sha256': digest(PROTOCOL),
            'suite_sha256': file_sha(root / 'suite.jsonl'),
            'questions_sha256': file_sha(root / 'questions.jsonl'),
            'source_manifest_sha256': file_sha(root / 'source_manifest.json'),
            'dataset_manifest_sha256': file_sha(root / 'dataset_manifest.json'),
            'selection_sha256': file_sha(root / 'selection.json'), 'selection': selection,
            'counts': expected, 'stats': stats, 'models': models,
            'scoring_protocol_at_freeze': SCORING_PROTOCOL,
            'cpu_actual_processor_checks': checked,
            'preflight_policy': {'skipped': skip_preflight, 'user_request': preference if skip_preflight else None,
                                 'prompt_construction': 'baseline processor once' if skip_preflight else 'all four processors',
                                 'image_database_scan': 'skipped' if skip_preflight else 'completed',
                                 'actual_cpu_image_preprocessing': 'skipped' if skip_preflight else 'completed'},
            'source_sha256': {n: file_sha(HERE / n) for n in ['protocol.py', 'freeze.py', 'prepare.py', 'sources.py', 'scoring.py', 'annotation_issues.py', 'run.py', 'throughput.py', 'launch_queue.py']},
            'created_at': datetime.now(timezone.utc).isoformat(),
            'note': ('Offline preflight skipped at explicit user request; prompts constructed once using baseline processor. Formal server prompt-token usage is checked on actual evaluation responses. ' if skip_preflight else 'Four processors produced identical prompt token counts. ')
                    + 'Selected perception questions only; original full images capped at 32 MP; separate from previous low-resolution results.'}
    atomic_json(root / 'suite.meta.json', meta)
    state('complete', rows=len(rows), suite_sha256=meta['suite_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA)
    args = parser.parse_args()
    try:
        freeze(args.root)
    except Exception as exc:
        atomic_json(args.root / 'freeze_state.json', {'status': 'failed', 'error': repr(exc)})
        raise
