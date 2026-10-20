#!/usr/bin/env python3
"""Recheck local source payloads against their pinned upstream LFS digests."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def sha256(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    raw = ROOT / 'raw'
    sources = []
    for item in json.loads((raw / 'primary_lfs_verification.json').read_text()):
        sources.append((Path(item['file']), item['upstream_lfs_sha256'], item['bytes'], item['repo'], item['revision']))
    for item in json.loads((raw / 'reused_source_verification.json').read_text()):
        sources.append((Path(item['path']), item['expected_etag'], item['bytes'], item['repo'], item['revision']))
    for item in json.loads((raw / 'cauldron_lfs_manifest.json').read_text()):
        sources.append((raw / 'cauldron' / item['path'], item['lfs']['oid'], item['size'], 'HuggingFaceM4/the_cauldron', '847a98a779b1652d65111daf20c972dfcd333605'))
    item = json.loads((raw / 'virl39k/images_expected.json').read_text())
    sources.append((raw / 'virl39k/images.zip', item['lfs']['oid'], item['size'], 'TIGER-Lab/ViRL39K', '812ec617dea4bc8a4e751663b88e4ebb7de4d00e'))
    records = []
    for path, expected, expected_bytes, repo, revision in sources:
        assert path.is_file(), path
        assert path.stat().st_size == expected_bytes, path
        actual = sha256(path)
        assert actual == expected, (path, expected, actual)
        records.append({'path': str(path.resolve()), 'repo': repo, 'revision': revision, 'bytes': expected_bytes, 'sha256': actual, 'upstream_sha256_match': True})
        print('Verified', path.name, flush=True)
    # Official non-LFS annotations are recorded separately: these local hashes
    # provide reproducibility, without claiming an unavailable upstream checksum.
    annotations = []
    for name in ('tallyqa_annotations.zip', 'clevr_no_images.zip'):
        path = raw / name
        assert path.is_file(), path
        annotations.append({'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': sha256(path), 'upstream_checksum_available': False})
    result = {'pinned_lfs_files_verified': len(records), 'files': records, 'official_annotation_archives': annotations}
    target = ROOT / 'prepared/source_validation.json'
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print('All pinned source payloads verified:', len(records), flush=True)


if __name__ == '__main__':
    main()
