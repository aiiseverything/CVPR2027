#!/usr/bin/env python3
"""Fail closed on mixture counts, train/eval overlap, schema, and missing images."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from build import ROOT, WEIGHTS, pixel_hash, read_jsonl

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--eval',nargs='+',required=True);ap.add_argument('--n',type=int,default=12000);ap.add_argument('--full-image-hash',action='store_true');a=ap.parse_args()
 ev=[r for p in a.eval for r in read_jsonl(p)]
 blocked={h for r in ev for h in r['image_sha256']}
 prepared_audit=json.loads((ROOT/'prepared'/'audit.json').read_text())
 actual_eval={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in a.eval}
 assert {r['path']:r['sha256'] for r in prepared_audit['eval_manifests']}==actual_eval
 split_audit=json.loads((ROOT/'raw/virl39k_original_split_audit.json').read_text())
 banned_virl_ids=set(split_audit['excluded_original_test_qids'])
 paths={};report={};ids_by_teacher={}
 for teacher,weights in WEIGHTS.items():
  file=ROOT/'prepared'/(teacher+'.parquet');rows=pq.read_table(file).to_pylist()
  assert len(rows)==a.n,(teacher,len(rows))
  counts=Counter(r['extra_info']['source'] for r in rows);source_ids=defaultdict(set);source_images=defaultdict(set)
  assert dict(counts)=={s:a.n*w//100 for s,w in weights.items()},(teacher,counts)
  for i,r in enumerate(rows):
   info=r['extra_info'];assert info['split']=='train'
   if info['source']=='virl39k':assert info['source_id'] not in banned_virl_ids
   hashes=json.loads(info['image_sha256']);assert not blocked.intersection(hashes)
   assert len(hashes)==len(r['images'])
   assert 1<=len(hashes)<=8
   source_ids[info['source']].add(info['record_id']);source_images[info['source']].update(hashes)
   assert r['prompt'][0]['content'].count('<image>')==len(r['images'])
   gt=json.loads(r['reward_model']['ground_truth']);task=gt['task']
   assert task in {'math','exact','vqa','anls','relaxed_numeric','bbox','json'}
   assert gt['answers'] or task=='bbox'
   if task=='vqa' or (task=='json' and gt.get('content_task')=='vqa'):assert len(gt['answers'])==10
   if task=='bbox':
    x1,y1,x2,y2=gt['bbox'];assert 0<=x1<x2<=1000 and 0<=y1<y2<=1000,(info,gt)
   for im,h in zip(r['images'],hashes):
    assert im['min_pixels']==4096 and im['max_pixels']==min(786432,1572864//max(1,len(r['images'])))
    p=Path(im['path']);assert p.is_file() and p.stat().st_size>0,p
    if str(p) in paths:assert paths[str(p)]==h
    paths[str(p)]=h
  # The exact ratios hold in each aligned 200-prompt supercycle.
  for start in range(0,a.n,200):
   c=Counter(r['extra_info']['source'] for r in rows[start:start+200])
   assert dict(c)=={s:2*w for s,w in weights.items()}
  ids_by_teacher[teacher]={r['extra_info']['record_id'] for r in rows}
  teacher_audit=prepared_audit['teachers'][teacher]
  file_hash=hashlib.sha256(file.read_bytes()).hexdigest()
  assert teacher_audit['sha256']==file_hash
  assert teacher_audit['exposures']==dict(counts)
  assert teacher_audit['unique_records']=={s:len(ids) for s,ids in source_ids.items()}
  assert teacher_audit['unique_images']=={s:len(hashes) for s,hashes in source_images.items()}
  assert teacher_audit['repeated_exposures']=={s:counts[s]-len(ids) for s,ids in source_ids.items()}
  if teacher=='generalist':
   for source,hashes in source_images.items():
    if source!='if_format':assert not hashes.intersection(source_images['if_format'])
  report[teacher]={'rows':len(rows),'source_counts':dict(counts),'unique_ids':len(ids_by_teacher[teacher]),'unique_records_by_source':{s:len(ids) for s,ids in source_ids.items()},'known_virl_original_test_leakage':0,'eval_image_overlap':0,'max_images':max(len(r['images']) for r in rows),'sha256':file_hash}
 # Repeated source items across specialists/generalist are intentional; heldout images are never reused.
 sample=list(sorted(paths.items())) if a.full_image_hash else list(sorted(paths.items()))[::max(1,len(paths)//500)]
 def verify_image(item):
  path,h=item;assert pixel_hash(path)==h,path
 with ThreadPoolExecutor(max_workers=4) as pool:
  for i,_ in enumerate(pool.map(verify_image,sample),1):
   if i%2000==0:print('Image RGB hashes verified',i,'/',len(sample),flush=True)
 report['images']={'distinct_paths':len(paths),'hashes_recomputed':len(sample),'all_exist':True}
 report['eval_manifests']=actual_eval
 report['if_image_overlap_with_other_generalist_sources']=0
 (ROOT/'prepared'/'validation.json').write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2))

if __name__=='__main__':main()
