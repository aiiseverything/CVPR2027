#!/usr/bin/env python3
"""Validate question-only selection audits and commit the six perception subsets."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from protocol import DATA, atomic_json, file_sha
from sources import load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA)
    args = parser.parse_args()
    root = args.root
    target = root / 'selection.json'
    if target.exists():
        raise ValueError('Selection already committed; refusing to change frozen inputs')
    manifest = load_manifest(root)
    draft_path = root / 'zoom_selection.json'
    draft = json.loads(draft_path.read_text())
    annotations = draft['annotations']
    if len(annotations) != 845:
        raise ValueError('Zoom annotation set is incomplete')
    source = next(f for f in manifest['files'] if f['benchmark'] == 'zoom')
    if draft['source']['revision'] != source['revision']:
        raise ValueError('Zoom annotation revision differs from pinned source')
    # Read no image/crop or answer columns. Verify every reviewed query against
    # the pinned, SHA-verified local parquet before selecting original IDs.
    source_path = root / source['local_path']
    if source_path.stat().st_size != source['size'] or file_sha(source_path) != source['sha256']:
        raise ValueError('Zoom parquet checksum mismatch')
    actual = pq.read_table(source_path, columns=['id', 'query']).to_pylist()
    if len(actual) != len(annotations):
        raise ValueError('Zoom source row count differs')
    for index, (row, annotation) in enumerate(zip(actual, annotations)):
        if annotation['row_index'] != index or any(row[k] != annotation[k] for k in ('id', 'query')):
            raise ValueError(f'Zoom annotation differs at original index {index}')
    decisions, part_hashes = [], {}
    for name, lower, upper in [('zoom_decisions_part1.json', 0, 440),
                               ('zoom_decisions_part2.json', 440, 845)]:
        path = root / name
        part = json.loads(path.read_text())
        if part['source_sha256'] != file_sha(draft_path):
            raise ValueError(f'{name}: decisions refer to changed annotations')
        rows = part['decisions']
        if sorted(r['index'] for r in rows) != list(range(lower, upper)):
            raise ValueError(f'{name}: missing or duplicate reviewed index')
        decisions.extend(rows)
        part_hashes[name] = file_sha(path)
    decisions.sort(key=lambda row: row['index'])
    if len({r['id'] for r in decisions}) != 845:
        raise ValueError('Duplicate Zoom source IDs')
    for decision, source_row in zip(decisions, actual):
        query_sha = hashlib.sha256(source_row['query'].encode()).hexdigest()
        if decision['id'] != source_row['id'] or decision['query_sha256'] != query_sha:
            raise ValueError(f"Decision does not match query: {decision['index']}")
        if type(decision['include']) is not bool or not all(decision.get(k) for k in ('category', 'reason', 'evidence_mode')):
            raise ValueError('Incomplete Zoom selection decision')
    kept = [r['id'] for r in decisions if r['include']]
    audit = {
        'schema_version': 'zoom-perception-review-v1',
        'source_revision': source['revision'], 'source_parquet_sha256': source['sha256'],
        'annotation_file_sha256': file_sha(draft_path), 'decision_file_sha256': part_hashes,
        'review_method': 'AI review of question and option text; no image review or human annotation; no gold labels or model predictions used for selection',
        'selected_count': len(kept), 'excluded_count': len(decisions) - len(kept),
        'category_counts': dict(Counter(r['category'] for r in decisions if r['include'])),
        'excluded_category_counts': dict(Counter(r['category'] for r in decisions if not r['include'])),
        'decisions': decisions,
    }
    audit_path = root / 'zoom_selection_audit.json'
    atomic_json(audit_path, audit)
    benchmarks = {
        'vstar': {'filters': [{'field': 'category', 'op': 'eq', 'value': 'direct_attributes'}],
                  'expected_count': 115, 'scope': 'Direct attributes; excludes relative position'},
        'zoom': {'filters': [], 'include_source_ids': kept, 'expected_count': len(kept),
                 'scope': 'Derived direct static perception selection, not an official split',
                 'audit_file': audit_path.name, 'audit_sha256': file_sha(audit_path)},
        'hrbench_4k': {'filters': [{'field': 'category', 'op': 'eq', 'value': 'single'}],
                       'expected_count': 400, 'scope': 'Fine-grained single-instance perception; 100 base questions with four cycles'},
        'hrbench_8k': {'filters': [{'field': 'category', 'op': 'eq', 'value': 'single'}],
                       'expected_count': 400, 'scope': 'Fine-grained single-instance perception; 100 base questions with four cycles'},
    }
    annotation_sources = {}
    for name, filename, expected_sha, expected_count in [
        ('mme_rw_en', 'MME_RealWorld.json', '502eae251a7a11257b94d36fde51e9171144184a522017637be99eea03393037', 19129),
        ('mme_rw_cn', 'MME_RealWorld_CN.json', '1f2902d386d49ddf536932bd0042b816f2131e14a089b52654f132f2dff51572', 3610),
    ]:
        path = root / filename
        if file_sha(path) != expected_sha:
            raise ValueError(f'Original MME annotations changed: {filename}')
        rows = json.loads(path.read_text())
        selected = [r for r in rows if r['Task'] == 'Perception' and not r['Category'].startswith('Attribute_Motion_')]
        if len(selected) != expected_count:
            raise ValueError(f'{name}: unexpected selected annotation count')
        categories = sorted({r['Category'] for r in selected})
        benchmarks[name] = {
            'filters': [{'field': 'category', 'op': 'startswith', 'value': 'Perception/'},
                        {'field': 'l2-category', 'op': 'in', 'value': categories}],
            'expected_count': expected_count,
            'scope': 'Official Perception task excluding Attribute_Motion categories; no Reasoning rows',
        }
        annotation_sources[filename] = {'sha256': expected_sha, 'selected_category_counts': dict(Counter(r['Category'] for r in selected))}
    selection = {
        'schema_version': 'highres-selection-v1', 'ready': True,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'scope': 'User-authorized perception subsets only; report separately from full benchmark scores',
        'authorized_models': ['baseline', 'reasoning-step100', 'perception-step100', 'perception-v2-step100'],
        'source_manifest_sha256': file_sha(root / 'source_manifest.json'),
        'benchmarks': benchmarks,
        'selection_provenance': {'zoom_audit_sha256': file_sha(audit_path),
                                 'mme_original_repo': 'yifanzhang114/MME-RealWorld',
                                 'mme_original_revision': '741cb8831ac86085bd54f678d13ca193e2334114',
                                 'mme_original_annotations': annotation_sources},
        'limitations': ['Zoom subset is derived from question text by AI review; not an official or image-audited pure-perception split.',
                       'Zoom uses deterministic local answer scoring, not its official LLM judge.',
                       'All models receive original full images without supplied boxes or crops.',
                       'Selection uses task scope rather than model predictions or correctness.'],
    }
    # Commit last: the detached queue starts preparing as soon as this appears.
    atomic_json(target, selection)
    print(json.dumps({'selection': str(target), 'counts': {k:v['expected_count'] for k,v in benchmarks.items()},
                      'total': sum(v['expected_count'] for v in benchmarks.values()), 'sha256': file_sha(target)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
