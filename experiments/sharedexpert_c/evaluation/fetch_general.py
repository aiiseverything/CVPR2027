#!/usr/bin/env python3
"""Restore pinned OCRBench/IFEval sources; never modifies model environments."""
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

HERE = Path(__file__).resolve().parent


def fetch(url, path, expected=None):
    if path.exists() and expected and hashlib.sha256(path.read_bytes()).hexdigest() == expected:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.download')
    urllib.request.urlretrieve(url, temporary)
    if expected and hashlib.sha256(temporary.read_bytes()).hexdigest() != expected:
        temporary.unlink()
        raise ValueError(f'Source checksum mismatch: {url}')
    temporary.replace(path)


def main():
    manifest = json.loads((HERE/'general_sources_manifest.json').read_text())
    root = HERE/'general_sources'
    base = 'https://raw.githubusercontent.com/google-research/google-research/' + manifest['commit'] + '/'
    for name, sha in manifest['files'].items():
        fetch(base+'instruction_following_eval/'+name, root/'instruction_following_eval'/name, sha)
    fetch(base+'LICENSE',root/'instruction_following_eval/LICENSE')
    (root/'instruction_following_eval/__init__.py').touch()
    ocr = manifest['ocrbench']
    for index, name in enumerate(ocr['files']):
        fetch('https://huggingface.co/datasets/'+ocr['repository']+'/resolve/'+ocr['revision']+'/'+name,
              root/f'ocrbench-{index}.parquet', ocr.get('sha256',{}).get(name))
    resource = manifest['nltk_data']
    fetch(resource['url'],root/'punkt_tab.zip',resource['sha256'])
    with zipfile.ZipFile(root/'punkt_tab.zip') as archive:
        if any(Path(name).is_absolute() or '..' in Path(name).parts for name in archive.namelist()):
            raise ValueError('Invalid tokenizer resource archive')
        archive.extractall(root/'nltk_data/tokenizers')


if __name__ == '__main__':
    main()
