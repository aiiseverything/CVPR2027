#!/usr/bin/env python3
"""Freeze OCRBench and original IFEval prompts, excluding local training overlap."""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path

from common import HERE, ROOT, PROTOCOL, SCORING_PROTOCOL, digest, file_sha, read_jsonl, question_sha
from prepare_suite import save_images, prompt_tokens


def exclusions_from_training():
    import pyarrow.parquet as pq
    images, questions, sources = set(), set(), {}
    for name in ("perception", "reasoning", "generalist", "domain_opd_6400"):
        path = HERE.parent / "data_pipeline/prepared" / (name + ".parquet")
        sources[name] = file_sha(path)
        for batch in pq.ParquetFile(path).iter_batches(columns=["extra_info"]):
            for row in batch.to_pylist():
                info = row["extra_info"]
                values = info["image_sha256"]
                images.update(json.loads(values) if isinstance(values, str) else values)
                questions.add(info["question_sha256"])
    return images, questions, sources


def build(processor_path, output):
    import pyarrow.parquet as pq
    from transformers import AutoProcessor
    if output.exists() or output.with_suffix(".meta.json").exists():
        raise ValueError("Frozen output exists; choose a new --output")
    sources = HERE / "general_sources"
    processor = AutoProcessor.from_pretrained(processor_path, local_files_only=True)
    train_images, train_questions, training = exclusions_from_training()
    existing_images = {h for path in (HERE/'suite.jsonl', HERE/'suite_blink_mmvp.jsonl')
                       for row in read_jsonl(path) for h in row['image_sha256']}
    assets = HERE / "assets_general"
    assets.mkdir(exist_ok=True)
    rows, excluded = [], []
    raw = pq.read_table(sources / "ocrbench-0.parquet").to_pylist()
    for index, item in enumerate(raw):
        paths, hashes, files, sizes = save_images(
            [base64.b64encode(item['image']['bytes']).decode()], assets)
        reason = ('training_image_overlap' if set(hashes) & train_images else
                  'existing_evaluation_image' if set(hashes) & existing_images else None)
        prompt = item['question']  # Preserve benchmark prompt, no answer wrapper.
        count = prompt_tokens(processor, prompt, sizes)
        if count > PROTOCOL['max_prompt_tokens']:
            reason = 'prompt_budget'
        if reason:
            excluded.append({'id':f'OCRBench:{index}','reason':reason})
            continue
        rows.append({'id':f'OCRBench:{index}', 'benchmark':'OCRBench_heldout', 'dimension':'general',
                     'general_domain':'ocr', 'category':item['question_type'], 'group':hashes[0],
                     'source_index':str(index), 'source_dataset':item['dataset'], 'split':'test',
                     'question':prompt, 'question_sha256':question_sha(prompt), 'prompt':prompt,
                     'prompt_tokens':count, 'images':paths, 'image_sha256':hashes,
                     'image_file_sha256':files, 'image_sizes':sizes, 'task':'ocrbench',
                     'answers':item['answer'], 'answer':item['answer'],
                     'scoring_version':'ocrbench-v1-final-content-length-zero'})
    for item in read_jsonl(sources / 'instruction_following_eval/data/input_data.jsonl'):
        prompt = item['prompt']
        count = prompt_tokens(processor, prompt, [])
        reason = ('training_question_overlap' if question_sha(prompt) in train_questions else
                  'prompt_budget' if count > PROTOCOL['max_prompt_tokens'] else None)
        if reason:
            excluded.append({'id':f"IFEval:{item['key']}",'reason':reason})
            continue
        rows.append({'id':f"IFEval:{item['key']}", 'benchmark':'IFEval_strict', 'dimension':'general',
                     'general_domain':'instruction_following', 'category':'|'.join(item['instruction_id_list']),
                     'group':question_sha(prompt), 'source_index':str(item['key']), 'split':'official-evaluation',
                     'question':prompt, 'question_sha256':question_sha(prompt), 'prompt':prompt,
                     'prompt_tokens':count, 'images':[], 'image_sha256':[], 'image_file_sha256':[],
                     'image_sizes':[], 'task':'ifeval', 'answer':None, 'ifeval':item,
                     'scoring_version':'google-e6890f8-strict-and-loose-final-content-length-zero'})
    rows.sort(key=lambda row: row['id'])
    output.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows))
    meta = {'protocol':PROTOCOL, 'protocol_sha256':digest(PROTOCOL),
            'suite_sha256':file_sha(output), 'scoring_protocol':SCORING_PROTOCOL,
            'scoring_protocol_sha256':digest(SCORING_PROTOCOL),
            'counts':dict(Counter(row['benchmark'] for row in rows)),
            'ocr_categories':dict(Counter(row['category'] for row in rows if row['task']=='ocrbench')),
            'training_sha256':training, 'excluded':excluded,
            'source_manifest':json.loads((HERE/'general_sources_manifest.json').read_text()),
            'source_sha256':{'ocr_parquet':file_sha(sources/'ocrbench-0.parquet')},
            'limitations':['Post-teacher-training expansion; no retrospective preregistration claim',
                          'Exact RGB overlap removal cannot detect every crop or re-encoding',
                          'OCRBench filtered subset, not the official 1000-point total',
                          'IFEval tests text instruction following, not multimodal instruction following',
                          'Thinking enabled; final content scored; length termination counts zero']}
    output.with_suffix('.meta.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'counts':meta['counts'],'excluded':len(excluded),'output':str(output)},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--processor',type=Path,default=HERE.parent/'models/Qwen3.5-4B')
    parser.add_argument('--output',type=Path,default=HERE/'suite_general.jsonl')
    args = parser.parse_args()
    build(args.processor,args.output)
