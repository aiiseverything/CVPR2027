#!/usr/bin/env python3
"""Audited one-time migration of unchanged generations to the total-image budget.

Only the sole changed image grid is invalidated. Connection errors are retried.
Original suites, run metadata and every generation (including errors) are retained.
"""
import json
import shutil
from pathlib import Path
from common import HERE, PROTOCOL, digest, file_sha, read_jsonl, score_prediction


def main():
    audit = json.loads((HERE / 'multimage_budget_audit.json').read_text())
    changed = {r['id']: r for r in audit['affected_rows']}
    assert set(changed) == {'MMMU_DEV_VAL:10719'}
    archive = HERE / 'old_protocol' / 'migration_v2_to_v3_total_image_budget'
    evidence_path = archive / 'migration.json'
    if evidence_path.exists():
        print(evidence_path.read_text())
        return
    archive.mkdir(parents=True, exist_ok=True)
    evidence = {'audit_sha256': file_sha(HERE / 'multimage_budget_audit.json'),
                'new_protocol': PROTOCOL, 'new_protocol_sha256': digest(PROTOCOL),
                'effective_image_grid_unchanged_rows': audit['total_rows'] - len(changed),
                'changed_rows': list(changed), 'suites': {}, 'caches': {}}
    suite_rows = {}
    for name in ['core_suite', 'suite']:
        path = HERE / f'{name}.jsonl'
        meta_path = path.with_suffix('.meta.json')
        meta = json.loads(meta_path.read_text())
        assert meta['protocol']['version'] == 'sharedexpert-teacher-eval-v2'
        assert file_sha(path) == meta['suite_sha256']
        shutil.copy2(path, archive / path.name)
        shutil.copy2(meta_path, archive / meta_path.name)
        old_rows = read_jsonl(path)
        rows = []
        for old in old_rows:
            row = dict(old)
            if row['id'] in changed:
                assert row['prompt_tokens'] == changed[row['id']]['prompt_old']
                row['prompt_tokens'] = changed[row['id']]['prompt_new']
            elif digest(old) != digest(row):
                raise AssertionError('An unchanged row was modified')
            rows.append(row)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))
        old_hash = meta['suite_sha256']
        meta.update(protocol=PROTOCOL, protocol_sha256=digest(PROTOCOL), suite_sha256=file_sha(path),
                    image_budget_migration={'previous_suite_sha256': old_hash,
                        'audit_file': str(HERE / 'multimage_budget_audit.json'),
                        'changed_row_ids': list(changed),
                        'cache_policy': 'Only unchanged effective image grids and identical row hashes can migrate; errors retried'})
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2)+'\n')
        suite_rows[name] = {r['id']: r for r in rows}
        evidence['suites'][name] = {'old_sha256': old_hash, 'new_sha256': meta['suite_sha256'], 'n': len(rows)}
    for rel, suite_name in [('core_results/full/baseline','core_suite'), ('results/full/baseline','suite')]:
        directory = HERE / rel
        if not (directory / 'run.json').exists():
            continue
        destination = archive / rel
        shutil.copytree(directory, destination)
        info = json.loads((directory / 'run.json').read_text())
        assert info['protocol']['version'] == 'sharedexpert-teacher-eval-v2'
        assert info['suite_sha256'] == evidence['suites'][suite_name]['old_sha256']
        old_map = {r['id']: r for r in read_jsonl(archive / f'{suite_name}.jsonl')}
        row_map = suite_rows[suite_name]
        previous = {r['id']: r for r in read_jsonl(directory / 'predictions.jsonl')}
        migrated, errors, changed_cached = [], [], []
        for sample_id, record in previous.items():
            assert sample_id in old_map and record['row_sha256'] == digest(old_map[sample_id])
            if record.get('error'):
                errors.append(sample_id)
                continue
            if sample_id in changed:
                changed_cached.append(sample_id)
                continue
            assert record['row_sha256'] == digest(row_map[sample_id])
            migrated.append(score_prediction(row_map[sample_id], record))
        info.update(protocol=PROTOCOL,protocol_sha256=digest(PROTOCOL),suite_sha256=evidence['suites'][suite_name]['new_sha256'], scorer_source_sha256=file_sha(HERE / 'common.py'))
        (directory / 'run.json').write_text(json.dumps(info,ensure_ascii=False,indent=2)+'\n')
        (directory / 'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in migrated))
        for stale in ['predictions.scored.jsonl', 'scoring.json']:
            (directory / stale).unlink(missing_ok=True)
        evidence['caches'][rel] = {'original_records': len(previous), 'migrated_successes': len(migrated), 'errors_to_retry': errors,
                                  'changed_image_grid_to_retry': changed_cached, 'raw_archive_sha256': file_sha(destination / 'predictions.jsonl')}
    for name in ['summary.json','summary.md','per_sample.csv']:
        path = HERE / 'results' / name
        if path.exists():
            shutil.move(str(path), str(archive / name))
    evidence_path.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'suite_sha256': evidence['suites']['suite']['new_sha256'],
        'unchanged_suite_rows': evidence['effective_image_grid_unchanged_rows'],
        'migrated_successes': {k:v['migrated_successes'] for k,v in evidence['caches'].items()},
        'errors_to_retry': {k:len(v['errors_to_retry']) for k,v in evidence['caches'].items()},
        'evidence': str(evidence_path)},indent=2))

if __name__ == '__main__':
    main()
