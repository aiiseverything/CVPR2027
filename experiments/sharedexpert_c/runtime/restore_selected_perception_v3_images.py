#!/usr/bin/env python3
"""CPU-only recovery of the frozen perception-v3 training images after migration.

Only absent images selected in the final 3,000-row dataset are restored. Raw
shards, image blobs, and decoded pixels are checked against existing evidence.
Frozen records, parquet, manifests, and existing image files are never changed.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

sys.dont_write_bytecode = True
E = Path(__file__).resolve().parents[1]
D = E / 'data_pipeline/prepared/perception_v3_filter/dataset'
S = E / 'data_pipeline/perception_v3_sources'
R = E / 'runtime'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pixel_hash(value):
    from PIL import Image
    with Image.open(value) as image:
        rgb = image.convert('RGB')
        return hashlib.sha256(struct.pack('>II', *rgb.size) + rgb.tobytes()).hexdigest()


def write_json(path, obj):
    tmp = path.with_name(path.name + '.writing')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def main():
    import pyarrow.parquet as pq
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    records = [json.loads(line) for line in (D / 'perception_v3.records.jsonl').read_text().splitlines()]
    manifest = json.loads((D / 'manifest.json').read_text())
    frozen_hashes = {name: sha(D / name) for name in ('manifest.json', 'perception_v3.records.jsonl', 'perception_v3.parquet')}
    assert frozen_hashes['perception_v3.records.jsonl'] == manifest['records_sha256']
    assert frozen_hashes['perception_v3.parquet'] == manifest['parquet_sha256']
    rows = pq.read_table(D / 'perception_v3.parquet', columns=['images', 'extra_info']).to_pylist()
    assert len(rows) == len(records) == 3000
    expected = {}
    jobs = {}
    missing_paths = set()
    for row, record in zip(rows, records, strict=True):
        assert row['extra_info']['record_id'] == record['id']
        assert [x['path'] for x in row['images']] == record['images']
        assert json.loads(row['extra_info']['image_sha256']) == record['image_sha256']
        for path, pixels in zip(record['images'], record['image_sha256'], strict=True):
            if path in expected:
                assert expected[path] == pixels
            expected[path] = pixels
            if Path(path).is_file():
                continue
            missing_paths.add(path)
            prov = record['source_provenance']
            raw = S / 'raw' / record['source'] / prov['file']
            if str(raw) not in jobs:
                m = json.loads((S / 'normalized_shards' / record['source'] / (raw.stem + '.manifest.json')).read_text())
                assert m['source_sha256'] == prov['file_sha256']
                jobs[str(raw)] = {'path': raw, 'sha256': prov['file_sha256'], 'bytes': m['source_bytes'], 'wanted': {}}
            job = jobs[str(raw)]
            assert job['sha256'] == prov['file_sha256']
            job['wanted'].setdefault(Path(path).stem, {})[path] = pixels
    for job in jobs.values():
        assert job['path'].is_file(), f"Missing local raw shard: {job['path']}"
        assert job['path'].stat().st_size == job['bytes'], f"Raw shard size mismatch: {job['path']}"
    # Keep the original temporary path functional, but store recovered images on
    # the persistent shared filesystem so another host migration retains them.
    live = Path('/root/autodl-tmp/sharedexpert/perception_v3_sources_images')
    persistent = S / 'selected_training_images'
    persistent.mkdir(exist_ok=True)
    live.parent.mkdir(parents=True, exist_ok=True)
    if not live.exists() and not live.is_symlink():
        live.symlink_to(persistent, target_is_directory=True)
    if live.is_symlink():
        assert live.resolve() == persistent.resolve(), 'Unexpected existing recovery destination'
    receipt = {'status': 'restoring', 'started_at': started, 'rows': len(rows), 'unique_image_paths': len(expected),
               'missing_image_paths_before': len(missing_paths), 'raw_shards_required': len(jobs),
               'persistent_destination': str(persistent), 'compatibility_path': str(live),
               'frozen_dataset_hashes': frozen_hashes, 'gpu_used': False, 'completed_shards': []}
    receipt_path = R / 'perception_v3_selected_images_recovery_20261030.json'
    write_json(receipt_path, receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'frozen_dataset_hashes'}), flush=True)

    def restore(job):
        actual = sha(job['path'])
        assert actual == job['sha256'], f"Raw shard hash mismatch: {job['path']}"
        wanted = job['wanted']
        found = set()
        written = 0
        for batch in pq.ParquetFile(job['path']).iter_batches(batch_size=32, columns=['image']):
            for row in batch.to_pylist():
                item = row.get('image')
                blob = item.get('bytes') if isinstance(item, dict) else None
                if not blob:
                    continue
                digest = hashlib.sha256(blob).hexdigest()
                if digest not in wanted or digest in found:
                    continue
                pixels = pixel_hash(io.BytesIO(blob))
                for name, expected_pixels in wanted[digest].items():
                    assert pixels == expected_pixels, f'Decoded pixel mismatch: {name}'
                    path = Path(name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    if path.is_file():
                        assert sha(path) == digest, f'Existing image blob mismatch: {name}'
                        continue
                    # Publish complete files atomically without replacing an image
                    # another shard worker may have restored concurrently.
                    temporary = None
                    try:
                        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.restoring-', delete=False) as output:
                            temporary = Path(output.name)
                            output.write(blob)
                            output.flush()
                            os.fsync(output.fileno())
                        os.link(temporary, path)
                        written += 1
                    except FileExistsError:
                        assert sha(path) == digest
                    finally:
                        if temporary is not None:
                            temporary.unlink(missing_ok=True)
                found.add(digest)
            if found == set(wanted):
                break
        assert found == set(wanted), f"Images absent from pinned raw shard: {job['path']}"
        return {'path': str(job['path']), 'sha256': actual, 'blobs_found': len(found), 'files_written': written}

    with ThreadPoolExecutor(max_workers=4) as pool:
        for completed in pool.map(restore, jobs.values()):
            receipt['completed_shards'].append(completed)
            write_json(receipt_path, receipt)
            print(json.dumps({'shards_done': len(receipt['completed_shards']), **completed}), flush=True)

    def verify(item):
        path, expected_pixels = item
        if not Path(path).is_file():
            return {'missing_image': path}
        actual = pixel_hash(path)
        if actual != expected_pixels:
            return {'pixel_hash_mismatch': path, 'actual': actual, 'expected': expected_pixels}
        return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        errors = [error for error in pool.map(verify, expected.items()) if error]
    assert all(sha(D / name) == digest for name, digest in frozen_hashes.items())
    receipt.update(status='complete' if not errors else 'failed', completed_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                   verified_image_paths=len(expected), errors=errors, dataset_unchanged=True)
    write_json(receipt_path, receipt)
    cpu_path = R / 'perception_v3_resume_cpu_20261030.json'
    if cpu_path.exists():
        previous = json.loads(cpu_path.read_text())
        backup = cpu_path.with_name(cpu_path.stem + '.before_pixel_hash_correction.json')
        if not backup.exists():
            backup.write_bytes(cpu_path.read_bytes())
        previous.update(data_errors=errors, verified_images=len(expected),
                        image_hash_basis='SHA256(big-endian width,height + decoded RGB pixels); initial file-byte comparison was invalid',
                        image_recovery_receipt=str(receipt_path), data_checked_at=receipt['completed_at'])
        write_json(cpu_path, previous)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'completed_shards'}), flush=True)
    assert not errors


if __name__ == '__main__':
    main()
