#!/usr/bin/env python3
"""Resumable HTTP ranges with final upstream LFS SHA256 validation."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import time
import requests
import socket
import urllib3.util.connection
urllib3.util.connection.allowed_gai_family=lambda:socket.AF_INET

def run(url,path,size,sha,workers=12,chunk=16<<20):
 path=Path(path);parts=path.with_name(path.name+'.ranges');parts.mkdir(exist_ok=True,parents=True)
 def one(start):
  end=min(start+chunk,size)-1;p=parts/f'{start:012d}.part'
  if p.exists() and p.stat().st_size==end-start+1:return p
  for attempt in range(8):
   try:
    pieces=[]
    for substart in range(start,end+1,1<<20):
     subend=min(substart+(1<<20)-1,end)
     request_url=url+('&' if '?' in url else '?')+f'fresh={substart}-{time.time_ns()}'
     r=requests.get(request_url,headers={'Range':f'bytes={substart}-{subend}'},timeout=(10,30))
     r.raise_for_status()
     if r.status_code!=206 or len(r.content)!=subend-substart+1 or not r.headers.get('Content-Range','').startswith(f'bytes {substart}-{subend}/'):raise ValueError('Bad HTTP range response')
     pieces.append(r.content)
    tmp=p.with_suffix('.tmp');tmp.write_bytes(b''.join(pieces));tmp.replace(p);return p
   except Exception as e:
    print('range retry',start,attempt,type(e).__name__,flush=True)
    if attempt==7:raise
    time.sleep(min(15,attempt+1))
 with ThreadPoolExecutor(max_workers=workers) as pool:
  futures=[pool.submit(one,start) for start in range(0,size,chunk)]
  for i,f in enumerate(as_completed(futures)):
   f.result()
   if i%10==0:print('ranges',i+1,'/',len(futures),flush=True)
 temporary=path.with_name(path.name+'.verified.tmp');h=hashlib.sha256()
 with open(temporary,'wb') as f:
  for start in range(0,size,chunk):
   b=(parts/f'{start:012d}.part').read_bytes();h.update(b);f.write(b)
 if h.hexdigest()!=sha:raise RuntimeError('Upstream SHA256 mismatch: '+h.hexdigest())
 temporary.replace(path);print('Verified',path,sha,flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('url');ap.add_argument('path');ap.add_argument('size',type=int);ap.add_argument('sha256');ap.add_argument('--workers',type=int,default=12)
 a=ap.parse_args();run(a.url,a.path,a.size,a.sha256,a.workers)
