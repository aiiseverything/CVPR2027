#!/usr/bin/env python3
"""校验已下载模型文件，sha256 不符的删除重下（等主下载进程退出后执行）"""
import os, time, hashlib, subprocess, importlib.util, urllib.parse
from concurrent.futures import ThreadPoolExecutor

R = "/root/autodl-fs/CVPR2027/pre-lab"
spec = importlib.util.spec_from_file_location("msd", os.path.join(R, "ms_download.py"))
msd = importlib.util.module_from_spec(spec); spec.loader.exec_module(msd)

REPOS = [("Qwen/Qwen3.5-4B",  os.path.join(R, "models", "Qwen3.5-4B")),
         ("Qwen/Qwen3.5-9B",  os.path.join(R, "models", "Qwen3.5-9B"))]

def sha(fp):
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def fetch_file(repo, p, size, fp):
    url = msd.BASE.format(repo=repo, path=urllib.parse.quote(p))
    chunk = 16 << 20
    ranges = [(s, min(s + chunk - 1, size - 1)) for s in range(0, size, chunk)]
    with open(fp, "wb") as f:
        f.truncate(size)
    with ThreadPoolExecutor(max_workers=8) as ex:
        for fu in [ex.submit(msd.fetch, url, s, e, fp) for s, e in ranges]:
            fu.result()

print("[repair] 等待主下载进程退出 ...", flush=True)
while subprocess.run(["pgrep", "-f", "[m]s_download.py"], capture_output=True).returncode == 0:
    time.sleep(30)

print("[repair] 开始校验", flush=True)
for repo, local in REPOS:
    files = msd.list_files(repo)
    for p, size, want in files:
        fp = os.path.join(local, p)
        if not os.path.exists(fp):
            print("  [repair] 缺失，重下 %s" % p, flush=True)
            fetch_file(repo, p, size, fp)
            continue
        if os.path.getsize(fp) != size:
            print("  [repair] 大小不符，重下 %s" % p, flush=True)
            os.remove(fp); fetch_file(repo, p, size, fp); continue
        if not want:
            continue
        got = sha(fp)
        if got != want:
            print("  [repair] sha 不符，重下 %s" % p, flush=True)
            os.remove(fp)
            fetch_file(repo, p, size, fp)
            got = sha(fp)
            print("  [repair] 重下%s: %s" % ("成功" if got == want else "仍然失败", p), flush=True)

print("[repair] === 最终校验 ===", flush=True)
bad = 0
for repo, local in REPOS:
    for p, size, want in msd.list_files(repo):
        fp = os.path.join(local, p)
        ok = os.path.exists(fp) and os.path.getsize(fp) == size and (not want or sha(fp) == want)
        print("  %s %s" % ("OK  " if ok else "FAIL", p), flush=True)
        bad += 0 if ok else 1
print("[repair] ALL DONE, 失败 %d 个" % bad, flush=True)
