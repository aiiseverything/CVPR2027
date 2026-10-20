#!/usr/bin/env python3
"""Verify all downloaded primary parquets against pinned upstream LFS objects."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import requests
from download_sources import ROOT, HF_REPOS, sha256

def verify_group(group):
 repo,rev,_=HF_REPOS[group]
 local=list((ROOT/'raw'/group).rglob('*.parquet'))
 directories=sorted({str(p.relative_to(ROOT/'raw'/group).parent) for p in local})
 tree={}
 for directory in directories:
  suffix='' if directory=='.' else '/'+directory
  response=requests.get(f'https://hf-mirror.com/api/datasets/{repo}/tree/{rev}{suffix}',params={'limit':1000},timeout=(10,60))
  response.raise_for_status()
  tree.update({x['path']:x for x in response.json()})
 records=[]
 for p in local:
  rel=str(p.relative_to(ROOT/'raw'/group));x=tree[rel]
  expected=x['lfs']['oid'];actual=sha256(p)
  if actual!=expected:raise RuntimeError(f'Upstream SHA256 mismatch: {p}')
  records.append({'source':group,'repo':repo,'revision':rev,'file':str(p),'bytes':p.stat().st_size,'sha256':actual,'upstream_lfs_sha256':expected})
 print('Verified',group,len(records),flush=True)
 return records

if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=5) as pool:
  result=[r for chunk in pool.map(verify_group,['refcoco','vqav2','docvqa','chartqa','virl39k']) for r in chunk]
 (ROOT/'raw/primary_lfs_verification.json').write_text(json.dumps(result,indent=2))
