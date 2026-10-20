#!/usr/bin/env python3
"""Normalize genuine train sources, audit leakage, and emit exact-exposure RL mixtures."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import re
import struct
import unicodedata
import zipfile
import zlib
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

ROOT=Path(__file__).resolve().parent
OLD=Path('/home/ml-user/workdir/project-2/data/raw/rlvr')
SEED=20261019
WEIGHTS={
 'perception':{'tallyqa':40,'clevr':40,'refcoco':20},
 'reasoning':{'geometry3k':40,'mmk12':20,'virl39k':40},
 'generalist':{'vqav2':40,'docvqa':5,'chartqa':5,'geometry3k':8,'mmk12':4,'virl39k':8,'tallyqa':8,'clevr':8,'refcoco':4,'if_format':10},
}

def norm(s):return ' '.join(unicodedata.normalize('NFKC',str(s)).casefold().split())
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def qhash(s):return digest(norm(s))
def pixel_hash(path):
 with Image.open(path) as im:
  im=im.convert('RGB');return hashlib.sha256(struct.pack('>II',*im.size)+im.tobytes()).hexdigest()

def materialize(img,source):
 if isinstance(img,(str,Path)):
  p=Path(img);assert p.is_file(),p
 else:
  data=img.get('bytes')
  if not data:return materialize(img['path'],source)
  h=hashlib.sha256(data).hexdigest();p=ROOT/'images'/source/(h+'.img')
  p.parent.mkdir(parents=True,exist_ok=True)
  if not p.exists():p.write_bytes(data)
 return str(p.resolve()),pixel_hash(p)

def verified_virl_image(archive,relative_path,stats):
 """Resume an interrupted extraction; check every selected image's ZIP CRC."""
 root=(ROOT/'raw/virl39k').resolve();path=(root/relative_path).resolve()
 if not path.is_relative_to(root):raise ValueError('Unsafe archive image path')
 member=archive.getinfo(relative_path)
 if path.is_file() and path.stat().st_size==member.file_size and (zlib.crc32(path.read_bytes())&0xffffffff)==member.CRC:
  stats['archive_images_verified_existing']+=1;return path
 # ZipFile.read validates the uncompressed CRC before returning. The archive
 # itself is independently checked against the pinned upstream LFS SHA256.
 data=archive.read(member);path.parent.mkdir(parents=True,exist_ok=True)
 tmp=path.with_name(path.name+'.extracting');tmp.write_bytes(data);tmp.replace(path)
 stats['archive_images_extracted']+=1
 return path

def make(source,sid,q,answers,images,task,**extra):
 pairs=[materialize(im,source) for im in images]
 q=q.replace('<image>','').strip()
 return {'id':f'{source}:{sid}','source':source,'source_id':str(sid),'split':'train','question':q,'question_sha256':qhash(q),'images':[x[0] for x in pairs],'image_sha256':[x[1] for x in pairs],'task':task,'answers':[str(a) for a in answers],**extra}

def rows(files):
 for file in sorted(files):
  pf=pq.ParquetFile(file)
  for batch in pf.iter_batches(batch_size=32):
   for i,row in enumerate(batch.to_pylist()):yield row

def selected_rows(files,limit,seed=SEED):
 """Uniform reservoir over source records without loading image bytes all at once."""
 rng=random.Random(seed);chosen=[]
 for i,row in enumerate(rows(files)):
  if i<limit:chosen.append(row)
  else:
   j=rng.randrange(i+1)
   if j<limit:chosen[j]=row
 rng.shuffle(chosen)
 return chosen

def json_array_items(stream,key):
 """Stream an array in an original JSON document without retaining CLEVR programs."""
 decoder=json.JSONDecoder();buffer='';position=0;ended=False
 pattern=re.compile(r'"'+re.escape(key)+r'"\s*:\s*\[')
 while True:
  chunk=stream.read(65536)
  if not chunk:raise ValueError('Missing original JSON array: '+key)
  buffer+=chunk;match=pattern.search(buffer)
  if match:position=match.end();break
 while True:
  while position<len(buffer) and buffer[position] in ' \n\r\t,':position+=1
  if position<len(buffer) and buffer[position]==']':return
  try:
   item,end=decoder.raw_decode(buffer,position)
  except json.JSONDecodeError:
   chunk=stream.read(65536)
   if not chunk:raise
   buffer=buffer[position:]+chunk;position=0;continue
  yield item;position=end
  if position>65536:buffer=buffer[position:];position=0

def canonical_annotations(source):
 if source=='tallyqa':
  with zipfile.ZipFile(ROOT/'raw/tallyqa_annotations.zip') as z:items=json.loads(z.read('train.json'))
  return [{'question':x['question'],'answer':str(x['answer']),'image':x['image'],'qid':x['question_id']} for x in items]
 if source=='clevr':
  def iterator():
   with zipfile.ZipFile(ROOT/'raw/clevr_no_images.zip') as z:
    names=[x for x in z.namelist() if x.endswith('CLEVR_train_questions.json')]
    assert len(names)==1,names
    with z.open(names[0]) as raw,io.TextIOWrapper(raw,encoding='utf-8') as text:
     for x in json_array_items(text,'questions'):
      yield {'question':x['question'],'answer':str(x['answer']),'image':x['image_filename'],'qid':x['question_index']}
  return iterator()
 raise ValueError(source)

def strip_instruction(q):
 # Original TallyQA/CLEVR questions are single-line. The Cauldron mirror appends
 # varied short-answer instructions after a newline; match against the official
 # original annotations and subsequently use the untouched original question.
 return q.split('\n',1)[0].strip()

def cauldron_records(source,limit):
 original=canonical_annotations(source)
 image_candidates=defaultdict(set);lookup={}
 for r in original:
  key=(norm(r['question']),norm(r['answer']).rstrip('.'))
  image_candidates[key].add(r['image']);lookup[(r['image'],*key)]=r
 files=list((ROOT/'raw/cauldron'/source).glob('*.parquet'))
 if not files:raise FileNotFoundError(source+' images not downloaded')
 # Reproduce reservoir selection/order using only integer positions. Image bytes
 # are processed in bounded Arrow batches rather than retained in a reservoir.
 total=sum(pq.ParquetFile(f).metadata.num_rows for f in sorted(files));rng=random.Random(SEED);indices=[]
 for i in range(total):
  if i<limit:indices.append(i)
  else:
   j=rng.randrange(i+1)
   if j<limit:indices[j]=i
 rng.shuffle(indices);ranks={index:rank for rank,index in enumerate(indices)}
 stats=Counter();records=[]
 for index,row in enumerate(rows(files)):
  if index not in ranks:continue
  texts=[(strip_instruction(x['user']),x['assistant'].strip().rstrip('.')) for x in row['texts']]
  possible=None
  for q,a in texts:
   ids=image_candidates.get((norm(q),norm(a)))
   if ids:possible=set(ids) if possible is None else possible.intersection(ids)
  if not possible or len(possible)!=1:
   stats['ambiguous_or_nontrain_source_image']+=1;continue
  original_image=next(iter(possible))
  # One question per image in the initial source pool controls repeated-image weighting.
  valid=[lookup[(original_image,norm(q),norm(a))] for q,a in texts if (original_image,norm(q),norm(a)) in lookup]
  if not valid:stats['no_train_question_match']+=1;continue
  chosen=valid[int(digest(original_image)[:8],16)%len(valid)]
  rec=make(source,chosen['qid'],chosen['question'],[chosen['answer']],row['images'],'exact',original_image=original_image)
  records.append((ranks[index],rec))
 records=[rec for rank,rec in sorted(records,key=lambda pair:pair[0])]
 stats['source_images_read']=len(indices);stats['verified_records']=len(records)
 return records,dict(stats)

def normalize(source,limit):
 if source in ('tallyqa','clevr'):return cauldron_records(source,limit)
 records=[];stats=Counter()
 if source=='geometry3k':
  for i,r in enumerate(rows((OLD/'geometry3k/data').glob('train-*.parquet'))):
   records.append(make(source,i,r['problem'],[r['answer']],r['images'],'math'))
 elif source=='mmk12':
  for r in selected_rows((OLD/'mmk12/data').glob('train-*.parquet'),limit):
   records.append(make(source,r['id'],r['question'],[r['answer']],[r['image']],'math',subject=r.get('subject','')))
 elif source=='virl39k':
  heldout_mmk_ids={r['id'] for p in (OLD/'mmk12/data').glob('test-*.parquet') for r in pq.read_table(p,columns=['id']).to_pylist()}
  rr=selected_rows([ROOT/'raw/virl39k/39Krelease.parquet'],limit)
  eligible=[]
  for r in rr:
   origin_id=r['qid'][len(r['source'])+1:] if r['qid'].startswith(r['source']+'-') else r['qid']
   if r['source'] in ('MMK12','MMK12Converted') and origin_id in heldout_mmk_ids:
    stats['known_original_mmk12_test_derivative_removed']+=1;continue
   eligible.append((r,origin_id))
  wanted={p for r,_ in eligible for p in r['image']}
  with zipfile.ZipFile(ROOT/'raw/virl39k/images.zip') as archive:
   names=[member.filename for member in archive.infolist() if member.filename in wanted]
   assert len(names)==len(wanted),'Selected ViRL image absent from original archive'
   def extract_image(name):
    local_stats=Counter();verified_virl_image(archive,name,local_stats);return local_stats
   # One source at a time; four bounded IO workers avoid serial shared-filesystem
   # latency, without keeping thousands of image byte arrays in RAM.
   with ThreadPoolExecutor(max_workers=4) as pool:
    for i,local_stats in enumerate(pool.map(extract_image,names),1):
     stats.update(local_stats)
     if i%1000==0:print('ViRL images verified',i,'/',len(names),flush=True)
  def convert_record(item):
   r,origin_id=item;images=[ROOT/'raw/virl39k'/p for p in r['image']]
   return make(source,r['qid'],r['question'],[r['answer']],images,'math',original_source=r['source'],original_source_id=origin_id,category=r['category'])
  with ThreadPoolExecutor(max_workers=4) as pool:
   for i,rec in enumerate(pool.map(convert_record,eligible),1):
    records.append(rec)
    if i%1000==0:print('ViRL normalized',i,'/',len(eligible),flush=True)
 elif source=='vqav2':
  for r in selected_rows((ROOT/'raw/vqav2/data').glob('train-*.parquet'),limit):
   records.append(make(source,r['question_id'],r['question'],[x['answer'] for x in r['answers']],[r['image']],'vqa',image_id=str(r['image_id'])))
 elif source=='docvqa':
  for r in selected_rows((ROOT/'raw/docvqa/DocVQA').glob('train-*.parquet'),limit):
   assert r['data_split']=='train'
   records.append(make(source,r['questionId'],r['question'],r['answers'],[r['image']],'anls'))
 elif source=='chartqa':
  for i,r in enumerate(selected_rows((ROOT/'raw/chartqa/data').glob('train-*.parquet'),limit)):
   image_id=hashlib.sha256(r['image']['bytes']).hexdigest()
   records.append(make(source,digest(r['query']+str(r['label'])+image_id)[:24],r['query'],r['label'],[r['image']],'relaxed_numeric'))
 else:raise ValueError(source)
 stats['verified_records']=len(records)
 return records,dict(stats)

def read_jsonl(path):
 with open(path) as f:return [json.loads(line) for line in f if line.strip()]

def coco_id(rec):
 value=rec.get('image_id')
 if value:return int(value)
 original=rec.get('original_image','')
 if 'COCO_' in original:
  match=re.search(r'(\d+)\.jpg$',original)
  if match:return int(match.group(1))
 return None

def finalize(args):
 assert args.n%100==0 and args.n%args.batch==0
 suites=[Path(p) for p in args.eval]
 if not suites or any(not p.is_file() for p in suites):raise RuntimeError('A completed evaluation manifest is required for decontamination')
 eval_rows=[r for p in suites for r in read_jsonl(p)]
 blocked_images={h for r in eval_rows for h in r['image_sha256']}
 blocked_questions={r['question_sha256'] for r in eval_rows if len(r.get('question',''))>=40}
 ref_sel=ROOT/'raw/refcoco_selection.json'
 blocked_coco=set(json.loads(ref_sel.read_text())['all_original_val_image_ids']) if ref_sel.exists() else set()
 pools={};audit={'n':args.n,'batch':args.batch,'seed':args.seed,'steps':args.n//args.batch,'eval_manifests':[{ 'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in suites],'sources':{},'teachers':{}}
 for source in sorted(set(k for w in WEIGHTS.values() for k in w)-{'if_format'}):
  path=ROOT/'normalized'/(source+'.jsonl')
  if not path.exists():raise FileNotFoundError(path)
  original=read_jsonl(path);seen=set();kept=[];stat=Counter()
  bbox_labels=defaultdict(set)
  for rec in original:
   if rec['task']=='bbox':bbox_labels[(tuple(rec['image_sha256']),rec['question_sha256'])].add(tuple(round(x,6) for x in rec['bbox']))
  ambiguous_bbox_keys={key for key,targets in bbox_labels.items() if len(targets)>1}
  if ambiguous_bbox_keys:stat['ambiguous_bbox_pairs_excluded']=len(ambiguous_bbox_keys)
  for rec in original:
   assert rec['split']=='train'
   key=(tuple(rec['image_sha256']),rec['question_sha256'])
   coco=coco_id(rec)
   if blocked_images.intersection(rec['image_sha256']):stat['eval_image_removed']+=1;continue
   if rec['question_sha256'] in blocked_questions:
    if source in ('geometry3k','mmk12','virl39k'):
     stat['eval_long_math_question_removed']+=1;continue
    stat['same_question_different_image_audit_only']+=1
   if coco is not None and coco in blocked_coco:stat['refcoco_val_image_removed']+=1;continue
   if key in ambiguous_bbox_keys:stat['ambiguous_bbox_records_removed']+=1;continue
   if key in seen:stat['duplicate_image_question_removed']+=1;continue
   seen.add(key);kept.append(rec)
  pools[source]=kept;stat['input']=len(original);stat['kept_unique_pairs']=len(kept);audit['sources'][source]=dict(stat)
  audit['sources'][source]['normalized_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
  norm_audit=path.with_suffix('.audit.json')
  if norm_audit.exists():audit['sources'][source]['normalization']=json.loads(norm_audit.read_text())
 # ViRL39K incorporates MMK12: remove actual shared images/questions before quotas.
 prior_images={h for name in ['geometry3k','mmk12'] for r in pools[name] for h in r['image_sha256']}
 prior_questions={r['question_sha256'] for name in ['geometry3k','mmk12'] for r in pools[name] if len(r['question'])>=80 and any(c.isdigit() for c in r['question'])}
 prior_mmk_ids={r['source_id'] for r in pools['mmk12']}
 old=len(pools['virl39k'])
 pools['virl39k']=[r for r in pools['virl39k'] if not (prior_images.intersection(r['image_sha256']) or r['question_sha256'] in prior_questions or (r.get('original_source') in ('MMK12','MMK12Converted') and r.get('original_source_id') in prior_mmk_ids))]
 audit['sources']['virl39k']['cross_reasoning_duplicate_removed']=old-len(pools['virl39k'])
 audit['sources']['virl39k']['after_cross_reasoning_deduplication']=len(pools['virl39k'])
 # Deterministic image-level partition: an IF item never shares an image with general VQA.
 if_pool=[];vqa=[]
 for r in pools['vqav2']:
  key=r['image_sha256'][0]
  if int(key[:8],16)%5==0:
   rr=dict(r);rr['source']='if_format';rr['id']='if_format:'+r['source_id'];rr['task']='json';rr['content_task']='vqa';if_pool.append(rr)
  else:vqa.append(r)
 pools['vqav2']=vqa;pools['if_format']=if_pool
 # Keep IF source images disjoint from every other source in the generalist too.
 if_images={h for r in if_pool for h in r['image_sha256']}
 if_coco={coco_id(r) for r in if_pool}
 general_pools={s:([r for r in rows_ if not if_images.intersection(r['image_sha256']) and (coco_id(r) is None or coco_id(r) not in if_coco)] if s!='if_format' else rows_) for s,rows_ in pools.items()}
 audit['if_partition']={'if_pool_size':len(if_pool),'vqa_pool_size':len(vqa),'image_partition':'int(pixel_sha256[:8],16) % 5 == 0','removed_from_other_generalist_pools':{s:len(pools[s])-len(general_pools[s]) for s in pools if s!='if_format'}}
 out=ROOT/'prepared';out.mkdir(exist_ok=True)
 audit['provenance_artifacts']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'raw/primary_lfs_verification.json',ROOT/'raw/reused_source_verification.json',ROOT/'raw/cauldron_lfs_manifest.json',ROOT/'raw/virl39k/images_expected.json',ROOT/'raw/virl39k_original_split_audit.json',ROOT/'raw/refcoco_selection.json']}
 source_validation=out/'source_validation.json'
 if source_validation.exists():audit['provenance_artifacts']['prepared/source_validation.json']=hashlib.sha256(source_validation.read_bytes()).hexdigest()
 for teacher,weights in WEIGHTS.items():
  rng=random.Random(args.seed+sum(map(ord,teacher)));active=general_pools if teacher=='generalist' else pools
  counters=Counter();uniques=defaultdict(set);image_uniques=defaultdict(set);decks={};ptr={}
  for source in weights:
   if not active[source]:raise RuntimeError('Empty source pool: '+source)
   decks[source]=list(active[source]);rng.shuffle(decks[source]);ptr[source]=0
  def draw(source):
   if ptr[source]>=len(decks[source]):rng.shuffle(decks[source]);ptr[source]=0
   r=decks[source][ptr[source]];ptr[source]+=1;counters[source]+=1;uniques[source].add(r['id']);image_uniques[source].update(r['image_sha256']);return r
  result=[]
  # 200-prompt supercycles preserve all source ratios and align to five 40-prompt steps.
  block_size=200 if args.n%200==0 else 100
  for block in range(args.n//block_size):
   labels=[s for s,w in weights.items() for _ in range(w*block_size//100)]
   rng.shuffle(labels)
   for source in labels:
    r=draw(source);payload={'task':r['task'],'answers':r['answers'],'format':'json' if source=='if_format' else 'boxed'}
    if r['task']=='bbox':payload['bbox']=r['bbox']
    if r['task']=='json':payload.update(json_key='answer',content_task=r.get('content_task','exact'))
    instruction='Return only a JSON object with exactly one key "answer" and a string value containing the correct answer. Do not include markdown fences or any other text.' if source=='if_format' else 'Give the final answer inside \\boxed{...}.'
    prompt=('<image>\n'*len(r['images']))+r['question']+'\n'+instruction
    image_cap=min(786432,1572864//max(1,len(r['images'])))
    result.append({'data_source':'sharedexpert_'+source,'prompt':[{'role':'user','content':prompt}],'images':[{'path':p,'min_pixels':4096,'max_pixels':image_cap} for p in r['images']],'ability':teacher,'reward_model':{'style':'rule','ground_truth':json.dumps(payload,ensure_ascii=False)},'extra_info':{'source':source,'source_id':r['source_id'],'split':'train','task':r['task'],'question_sha256':r['question_sha256'],'image_sha256':json.dumps(r['image_sha256']),'record_id':r['id'],'index':len(result)}})
  assert len(result)==args.n
  assert dict(counters)=={s:args.n*w//100 for s,w in weights.items()}
  path=out/(teacher+'.parquet');pq.write_table(pa.Table.from_pylist(result),path,compression='zstd')
  audit['teachers'][teacher]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'exposures':dict(counters),'unique_records':{s:len(x) for s,x in uniques.items()},'unique_images':{s:len(x) for s,x in image_uniques.items()},'repeated_exposures':{s:counters[s]-len(uniques[s]) for s in counters},'available_pool_sizes':{s:len(active[s]) for s in weights},'block_size':block_size}
 (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2))
 print(json.dumps(audit['teachers'],indent=2),flush=True)

def main():
 ap=argparse.ArgumentParser();sub=ap.add_subparsers(dest='command',required=True)
 n=sub.add_parser('normalize');n.add_argument('source',choices=['geometry3k','mmk12','virl39k','vqav2','docvqa','chartqa','tallyqa','clevr']);n.add_argument('--limit',type=int,default=10000)
 f=sub.add_parser('finalize');f.add_argument('--n',type=int,default=12000);f.add_argument('--batch',type=int,default=40);f.add_argument('--seed',type=int,default=SEED);f.add_argument('--eval',nargs='+',required=True)
 a=ap.parse_args()
 if a.command=='finalize':return finalize(a)
 records,stats=normalize(a.source,a.limit)
 out=ROOT/'normalized';out.mkdir(exist_ok=True)
 with open(out/(a.source+'.jsonl'),'w') as fp:
  for r in records:fp.write(json.dumps(r,ensure_ascii=False)+'\n')
 (out/(a.source+'.audit.json')).write_text(json.dumps(stats,indent=2));print(a.source,stats,flush=True)

if __name__=='__main__':main()
