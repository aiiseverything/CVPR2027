#!/usr/bin/env python3
"""Snapshot/release only this project's idle 9B servers on GPUs 3--6.

No broad pkill, process-name kill, or unrelated GPU process termination.
The saved commands can be restored with the restore subcommand.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
STATE=HERE/'runtime/previous_9b_servers.json'
PORT_GPU={8113:'3',8114:'4',8115:'5',8116:'6'}
SAFE_ENV={'CUDA_VISIBLE_DEVICES','CUDA_HOME','PATH','LD_LIBRARY_PATH',
          'VLLM_USE_FLASHINFER_SAMPLER','VLLM_USE_V1','VLLM_WORKER_MULTIPROC_METHOD'}


def save_json(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def matches_process(record):
    try:
        return cmdline(record['pid'])==record['argv'] and starttime(record['pid'])==record['starttime']
    except (FileNotFoundError,ProcessLookupError):
        return False


def healthy(record):
    if not matches_process(record): return False
    try:
        response=json.load(urllib.request.urlopen(f'http://127.0.0.1:{record["port"]}/v1/models',timeout=3))
        expected=record['argv'][record['argv'].index('--served-model-name')+1]
        return any(model.get('id')==expected for model in response.get('data',[]))
    except (OSError,ValueError):
        return False


def wait_healthy(record):
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        if not matches_process(record):
            raise RuntimeError(f'Restored server {record["port"]} exited; see restore_PORT.log')
        if healthy(record): return
        time.sleep(2)
    raise RuntimeError(f'Server {record["port"]} readiness timed out; partial state retained')


def cmdline(pid):
    return [x.decode() for x in (Path('/proc')/str(pid)/'cmdline').read_bytes().split(b'\0') if x]


def starttime(pid):
    return (Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19]


def idle(port):
    data=urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=5).read().decode()
    values=[]
    for line in data.splitlines():
        if line.startswith(('vllm:num_requests_running{','vllm:num_requests_waiting{')):
            values.append(float(line.rsplit(' ',1)[1]))
    if len(values)<2 or any(values):
        raise RuntimeError(f'Port {port} has active work or missing queue metrics; leave it running')


def snapshot():
    records=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit(): continue
        pid=int(process.name)
        try:
            args=cmdline(pid)
            if str(ROOT/'pre-lab/venv/bin/vllm') not in args or '--port' not in args: continue
            port=int(args[args.index('--port')+1])
            if port not in PORT_GPU: continue
            if str(ROOT/'pre-lab/models/Qwen3.5-9B') not in args:
                raise RuntimeError(f'Port {port} is not the expected project 9B server')
            # Persist only a narrow non-secret runtime allowlist, never the full environment.
            runtime_env={key.decode():value.decode() for item in (process/'environ').read_bytes().split(b'\0')
                         if b'=' in item for key,value in [item.split(b'=',1)] if key.decode() in SAFE_ENV}
            device=runtime_env['CUDA_VISIBLE_DEVICES']
            if device!=PORT_GPU[port]: raise RuntimeError(f'Unexpected GPU mapping on port {port}')
            idle(port)
            records.append({'pid':pid,'starttime':starttime(pid),'port':port,'gpu':device,
                            'argv':args,'cwd':str((process/'cwd').resolve()),'environment':runtime_env})
        except (FileNotFoundError,ProcessLookupError): continue
    if set(r['port'] for r in records)!=set(PORT_GPU):
        raise RuntimeError(f'Expected exactly four servers, found {[r["port"] for r in records]}')
    records.sort(key=lambda r:r['port'])
    STATE.parent.mkdir(parents=True,exist_ok=True)
    if STATE.exists():
        previous=json.loads(STATE.read_text())
        # Upgrade an old snapshot only when every process identity is unchanged.
        strip_env=lambda rows:[{k:v for k,v in r.items() if k not in {'environment','environment_source'}} for r in rows]
        if strip_env(previous)!=strip_env(records):
            raise RuntimeError(f'Existing snapshot differs: {STATE}')
    save_json(STATE,records)
    print(json.dumps({'snapshot':str(STATE),'servers':[{k:r[k] for k in ('pid','port','gpu')} for r in records]}))


def release():
    records=json.loads(STATE.read_text())
    live=[]
    # Recheck every server before changing any process.
    for r in records:
        if not (Path('/proc')/str(r['pid'])).exists():
            # A service already lost to the recorded cgroup OOM needs no signal.
            continue
        if not matches_process(r):
            raise RuntimeError('PID/command changed after snapshot; aborting release')
        idle(r['port'])
        live.append(r)
    time.sleep(2)
    for r in live: idle(r['port'])
    for r in live: os.kill(r['pid'],signal.SIGTERM)
    save_json(STATE.parent/'released_9b_servers.json',{'ports':list(PORT_GPU),'signalled_pids':[r['pid'] for r in live],'time':time.time()})
    print(f'Sent SIGTERM only to {len(live)} surviving recorded idle project servers.')


def restore():
    records=json.loads(STATE.read_text())
    progress_path=STATE.parent/'restoring_9b_servers.json'
    progress=json.loads(progress_path.read_text()) if progress_path.exists() else {}
    pending=[]
    for original in records:
        saved=progress.get(str(original['port']))
        if saved and matches_process(saved):
            continue
        pending.append(original)
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    used={a.strip():int(b) for a,b in (line.split(',') for line in memory.splitlines())}
    if any(used[r['gpu']]>1024 for r in pending):
        raise RuntimeError('A requested GPU is busy; no restart performed')
    for r in pending:
        with open(STATE.parent/f'restore_{r["port"]}.log','ab') as log:
            env=dict(os.environ)
            env.update(r.get('environment',{}))
            env['CUDA_VISIBLE_DEVICES']=r['gpu']
            p=subprocess.Popen(r['argv'],cwd=r['cwd'],env=env,
                               stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'port':r['port'],'new_pid':p.pid}))
        progress[str(r['port'])]=dict(r,pid=p.pid,starttime=starttime(p.pid))
        save_json(progress_path,progress)
        # Sequential startup prevents four simultaneous 9B loading/warmup peaks
        # from exceeding this container's 64 GiB limit.
        wait_healthy(progress[str(r['port'])])
    deadline=time.monotonic()+600
    while time.monotonic()<deadline:
        failed=[r['port'] for r in progress.values() if not matches_process(r)]
        if failed:
            raise RuntimeError(f'Restored servers exited before readiness: {failed}; see restore_PORT.log')
        if all(healthy(r) for r in progress.values()) and len(progress)==len(records):
            save_json(STATE.parent/'restored_9b_servers.json',{'servers':list(progress.values()),'time':time.time(),'healthy':True})
            print('All four previous project services are healthy.')
            return
        time.sleep(2)
    raise RuntimeError('Timed out waiting for restored services; partial state retained for retry')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['snapshot','release','restore'])
    globals()[parser.parse_args().action]()
