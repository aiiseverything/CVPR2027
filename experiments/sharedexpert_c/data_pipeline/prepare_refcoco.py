#!/usr/bin/env python3
"""Prepare genuine UNC RefCOCO train and val examples from pinned annotations."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import struct
import unicodedata
import pyarrow.parquet as pq
from PIL import Image
from download_sources import ROOT, download

def qhash(text):
    return hashlib.sha256(' '.join(unicodedata.normalize('NFKC', text).casefold().split()).encode()).hexdigest()

def ihash(path):
    with Image.open(path) as im:
        im = im.convert('RGB')
        return hashlib.sha256(struct.pack('>II', *im.size) + im.tobytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,default=20261019)
    ap.add_argument('--train-images',type=int,default=3500)
    ap.add_argument('--eval-rows',type=int,default=500)
    ap.add_argument('--workers',type=int,default=16)
    a=ap.parse_args()
    raw=ROOT/'raw/refcoco/data'; rng=random.Random(a.seed)
    train=pq.read_table(next(raw.glob('train-*.parquet'))).to_pylist()
    val=pq.read_table(next(raw.glob('validation-*.parquet'))).to_pylist()
    assert {r['split'] for r in train}=={'train'}
    assert {r['split'] for r in val}=={'val'}
    train_ids=sorted({r['image_id'] for r in train});rng.shuffle(train_ids)
    train_ids=set(train_ids[:a.train_images])
    train=[r for r in train if r['image_id'] in train_ids]
    rng.shuffle(val); val=val[:a.eval_rows]
    assert train_ids.isdisjoint({r['image_id'] for r in val})
    output=ROOT/'normalized';output.mkdir(exist_ok=True)
    (ROOT/'raw/refcoco_selection.json').write_text(json.dumps({'seed':a.seed,'train_images':len(train_ids),'heldout_rows':len(val),'heldout_image_ids':sorted({r['image_id'] for r in val}),'all_original_val_image_ids':sorted({r['image_id'] for r in pq.read_table(next(raw.glob('validation-*.parquet'))).to_pylist()}),'source':'jxu124/refcoco','revision':'9fba8200c5326e996f789191f095bd464ef1d09e'},indent=2))
    for split,rows in [('val',val),('train',train)]:
        ids=sorted({r['image_id'] for r in rows})
        jobs=[]
        for image_id in ids:
            name=f'COCO_train2014_{image_id:012d}.jpg'
            jobs.append((f'http://images.cocodataset.org/train2014/{name}',ROOT/'images/coco'/name,0))
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            for n,rec in enumerate(pool.map(download,jobs)):
                if n%100==0:print(f'{split} COCO images {n}/{len(jobs)}',flush=True)
        image_hash={i:ihash(ROOT/'images/coco'/f'COCO_train2014_{i:012d}.jpg') for i in ids}
        outfile=output/('refcoco.jsonl' if split=='train' else 'refcoco_heldout.jsonl')
        with open(outfile,'w') as f:
            for row in rows:
                info=json.loads(row['raw_image_info']);w,h=info['width'],info['height']
                anns=json.loads(row['raw_anns']);x,y,bw,bh=anns['bbox']
                assert all(abs(a-b)<1e-5 for a,b in zip(row['bbox'],[x,y,x+bw,y+bh]))
                bbox=[1000*x/w,1000*y/h,1000*(x+bw)/w,1000*(y+bh)/h]
                if split=='val':sentences=row['sentences'][:1]
                else:sentences=row['sentences']
                for sentence in sentences:
                    q=f'Locate the object described as "{sentence["sent"]}". Return its bounding box as [x1, y1, x2, y2] in coordinates normalized to 0-1000.'
                    rec={'id':f'refcoco:{row["ref_id"]}:{sentence["sent_id"]}','source':'refcoco','split':split,'source_id':str(sentence['sent_id']),'image_id':str(row['image_id']),'images':[str(ROOT/'images/coco'/f'COCO_train2014_{row["image_id"]:012d}.jpg')],'image_sha256':[image_hash[row['image_id']]],'question':q,'question_sha256':qhash(q),'task':'bbox','answers':[], 'bbox':bbox,'bbox_xyxy':bbox,'bbox_xyxy_1000':bbox,'reward_type':'bbox'}
                    f.write(json.dumps(rec,ensure_ascii=False)+'\n')
        print(outfile,flush=True)

if __name__=='__main__':main()
