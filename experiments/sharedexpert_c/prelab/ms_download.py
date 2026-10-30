#!/usr/bin/env python3
"""从 ModelScope 并发下载模型快照（Range 分块多线程 + sha256 校验）"""
import os, sys, json, time, threading, urllib.request, urllib.parse
from concurrent.futures import ThreadPoolExecutor

API  = "https://modelscope.cn/api/v1/models/{repo}/repo/files"
BASE = "https://modelscope.cn/models/{repo}/resolve/master/{path}"

_lock = threading.Lock()
_done = 0
_total = 0
_t0 = 0.0

def log(*a):
    print(*a, flush=True)

def api_get(repo, root=""):
    url = API.format(repo=repo) + "?" + urllib.parse.urlencode({"Revision": "master", "Root": root})
    last = None
    for i in range(6):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError("API failed: %s" % last)

def list_files(repo, root=""):
    out = []
    for f in api_get(repo, root)["Data"]["Files"]:
        p = f.get("Path") or f.get("Name")
        if f.get("Type") in ("tree", "dir"):
            out += list_files(repo, p)
        else:
            out.append((p, int(f.get("Size") or 0), f.get("Sha256")))
    return out

def fetch(url, start, end, path, retries=6):
    global _done
    n = end - start + 1
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start, end)})
            with urllib.request.urlopen(req, timeout=120) as r:
                buf = r.read()
            if len(buf) != n:
                raise IOError("short read %d/%d" % (len(buf), n))
            fd = os.open(path, os.O_WRONLY)
            try:
                os.pwrite(fd, buf, start)
            finally:
                os.close(fd)
            with _lock:
                _done += n
            return
        except Exception as e:
            if a == retries - 1:
                raise IOError("%s [%d-%d] failed: %s" % (path, start, end, e))
            time.sleep(1.5 * (a + 1))

def monitor():
    while True:
        time.sleep(10)
        with _lock:
            d = _done
        el = max(time.time() - _t0, 1e-9)
        sp = d / el / 1e6
        pct = 100.0 * d / _total if _total else 0
        eta = (_total - d) / max(d / el, 1) / 60 if d else 0
        log("  [进度] %.2f/%.2f GB (%.1f%%) %.1f MB/s ETA %.0f 分钟" % (d/1e9, _total/1e9, pct, sp, eta))

def download_one(repo, local, threads=16, mb=32):
    global _done, _total, _t0
    files = list_files(repo)
    total = sum(s for _, s, _ in files)
    log("=== %s -> %s | %d 个文件 | %.2f GB ===" % (repo, local, len(files), total/1e9))
    _done, _total, _t0 = 0, total, time.time()
    threading.Thread(target=monitor, daemon=True).start()
    for p, size, sha in sorted(files, key=lambda x: -x[1]):
        dst = os.path.join(local, p)
        d = os.path.dirname(dst)
        if d:
            os.makedirs(d, exist_ok=True)
        if os.path.exists(dst) and os.path.getsize(dst) == size:
            log("  跳过（已完整） %s" % p)
            with _lock:
                _done += size
            continue
        if size == 0:
            open(dst, "wb").close()
            continue
        url = BASE.format(repo=repo, path=urllib.parse.quote(p))
        chunk = mb * 1024 * 1024
        ranges = [(s, min(s + chunk - 1, size - 1)) for s in range(0, size, chunk)]
        with open(dst, "wb") as f:
            f.truncate(size)
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=threads) as ex:
            for fu in [ex.submit(fetch, url, s, e, dst) for s, e in ranges]:
                fu.result()
        log("  ✓ %s  %.0f MB  %.0f s" % (p, size/1e6, time.time()-t1))
    return files

def verify(repo, local, files):
    import hashlib
    bad = 0
    for p, size, sha in files:
        if not sha:
            continue
        fp = os.path.join(local, p)
        if not os.path.exists(fp) or os.path.getsize(fp) != size:
            log("  ✗ 缺失/大小不符 %s" % p); bad += 1; continue
        h = hashlib.sha256()
        with open(fp, "rb") as f:
            for blk in iter(lambda: f.read(8 << 20), b""):
                h.update(blk)
        if h.hexdigest() == sha:
            log("  ✓ sha256 %s" % p)
        else:
            log("  ✗ sha256 不符 %s\n    期望 %s\n    实际 %s" % (p, sha, h.hexdigest())); bad += 1
    return bad

if __name__ == "__main__":
    ROOT = "/root/autodl-fs/CVPR2027/pre-lab/models"
    bad = 0
    for repo in ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B"]:
        local = os.path.join(ROOT, repo.split("/")[-1])
        files = download_one(repo, local)
        bad += verify(repo, local, files)
    log("ALL DONE, 校验失败 %d 个" % bad)
