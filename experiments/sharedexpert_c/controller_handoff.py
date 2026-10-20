#!/usr/bin/env python3
"""One-shot, identity-guarded controller handoff after perception export; never trains.

prepare binds the old live controller and the on-disk replacement/check scripts.
launch starts a detached watcher. No watcher is started by prepare or self-test.
"""
from __future__ import annotations
import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import select
import signal
import subprocess
import sys
import tempfile
import time
import uuid

HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
TERMINAL = {'complete', 'aborted', 'failed', 'cancelled'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(directory_fd)
    finally: os.close(directory_fd)


def read(path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(pid):
    try:
        p = Path('/proc') / str(int(pid))
        st = p.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': int(pid), 'start_ticks': int(st[19]), 'state': st[0],
                'ppid': int(st[1]), 'pgrp': int(st[2]), 'session': int(st[3]),
                'argv': [os.fsdecode(x) for x in p.joinpath('cmdline').read_bytes().split(b'\0') if x],
                'cwd': str(p.joinpath('cwd').resolve(strict=True))}
    except (OSError, ValueError, IndexError):
        return None


def same(actual, expected):
    return bool(actual and actual['state'] not in ('Z', 'X') and
                all(actual[k] == expected[k] for k in ('pid', 'start_ticks', 'argv', 'cwd')))


def alive(expected):
    return same(identity(expected['pid']), expected)


@contextlib.contextmanager
def pid_handle(expected):
    if not hasattr(os, 'pidfd_open') or not hasattr(signal, 'pidfd_send_signal'):
        raise RuntimeError('Linux pidfd support is required; no unsafe numeric-PID fallback')
    fd = os.pidfd_open(expected['pid'])
    try:
        if not same(identity(expected['pid']), expected):
            raise RuntimeError('Process identity mismatch')
        yield fd
    finally:
        os.close(fd)


def send_exact(expected, sig):
    with pid_handle(expected) as fd:
        signal.pidfd_send_signal(fd, sig)


def processes():
    result = {}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            v = identity(int(p.name))
            if v and v['state'] not in ('Z', 'X'):
                result[v['pid']] = v
    return result


class Context:
    def __init__(self, here=HERE):
        self.here = Path(here).resolve()
        self.root = self.here.parents[1]
        self.runtime = self.here / 'runtime'
        self.home = self.runtime / 'controller_handoff'
        self.plan = self.home / 'plan.json'
        self.state = self.home / 'status.json'
        self.events = self.home / 'events.jsonl'
        self.driver = self.here / 'run_experiment.py'
        self.launcher = self.here / 'training/launch.py'
        self.check = self.here / 'training/multimodal_memory_check.py'
        self.cpu_receipt = self.here / 'logs/multimodal_memory_check/cpu_report.json'
        digest = hashlib.sha256(str(self.runtime).encode()).hexdigest()
        self.driver_lock = Path(tempfile.gettempdir()) / ('sharedexpert-driver-' + digest + '.lock')
        self.lock = Path(tempfile.gettempdir()) / ('sharedexpert-handoff-' + digest + '.lock')

    def hashes(self):
        return {str(p): sha(p) for p in (SELF, self.driver, self.launcher, self.check, self.cpu_receipt)}

    def event(self, event, **fields):
        self.home.mkdir(parents=True, exist_ok=True)
        with self.events.open('a') as f:
            f.write(json.dumps({'at': now(), 'event': event, **fields}, ensure_ascii=False) + '\n')
            f.flush(); os.fsync(f.fileno())


@contextlib.contextmanager
def lock_file(path):
    with path.open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield f


def lock_free(path):
    try:
        with lock_file(path):
            return True
    except BlockingIOError:
        return False


def verify_hashes(ctx, plan):
    if plan['bound_sha256'] != ctx.hashes():
        raise RuntimeError('Bound handoff/driver/launcher/check script changed; refusing handoff')


def expected_driver(ctx, proc):
    return (proc and proc['cwd'] == str(ctx.root) and len(proc['argv']) >= 3 and
            proc['argv'][1:3] == [str(ctx.driver), 'run'] and
            proc['argv'][3:] in ([], ['--wait-for-smoke']))


def prepare(ctx, old_pid, old_ticks):
    with lock_file(ctx.lock):
        prev = read(ctx.state, {})
        if prev.get('irreversible_started') or prev.get('launch_requested') or prev.get('stage') == 'complete':
            raise RuntimeError('An irreversible/complete handoff already exists; do not reset it automatically')
        old = identity(old_pid)
        if not expected_driver(ctx, old) or old['start_ticks'] != old_ticks:
            raise RuntimeError('Old driver PID/start_ticks/argv/cwd do not match the requested controller')
        if old['state'] in ('T', 't'):
            raise RuntimeError('Old driver was already stopped; refusing to adopt or resume it')
        state = read(ctx.runtime / 'experiment_state.json', {})
        if state.get('driver_pid') != old_pid or state.get('stage') in ('failed', 'interrupted', 'complete'):
            raise RuntimeError('Old driver experiment state is not active/matching')
        cpu = read(ctx.cpu_receipt, {})
        if cpu.get('status') != 'cpu_prepared' or cpu.get('gpu_started') is not False:
            raise RuntimeError('New memory-check CPU preparation is not complete')
        for source in (ctx.check, ctx.launcher):
            if cpu.get('source_sha256', {}).get(str(source)) != sha(source):
                raise RuntimeError('CPU preparation receipt does not match current check/launcher source')
        plan = {'version': 1, 'id': uuid.uuid4().hex, 'prepared_at': now(),
                'here': str(ctx.here), 'old_driver': old, 'bound_sha256': ctx.hashes(),
                'gpus': [3, 4, 5, 6], 'poll_seconds': 5,
                'old_driver_disk_sha256_at_start': read(ctx.here / 'logs/unused_topk_gate_validation.json', {}).get('driver_sha256'),
                'new_launch_argv': [sys.executable, str(ctx.driver), 'launch']}
        atomic(ctx.plan, plan)
        atomic(ctx.state, {'stage': 'prepared', 'plan_id': plan['id'], 'old_driver': old,
                           'heartbeat_epoch': time.time(), 'at': now()})
        ctx.event('prepared', plan_id=plan['id'], old_pid=old_pid)
        return plan


def perception_complete(ctx):
    run = ctx.here / 'runs/perception'
    latest = run / 'checkpoints/latest_checkpointed_iteration.txt'
    if not latest.exists() or latest.read_text().strip() != '300':
        return False
    source = run / 'checkpoints/global_step_300/actor'
    if not (source / 'fsdp_config.json').is_file():
        return False
    for kind in ('model', 'optim', 'extra_state'):
        for rank in range(4):
            p = source / f'{kind}_world_size_4_rank_{rank}.pt'
            if not p.is_file() or p.stat().st_size == 0:
                return False
    target = run / 'hf/global_step_300'
    marker = read(target / 'sharedexpert_export_complete.json', {})
    if marker.get('step') != 300 or marker.get('source') != str(source):
        return False
    return ((target / 'config.json').is_file() and
            any(p.stat().st_size > 0 for p in target.glob('*.safetensors')))


def reasoning_progress(ctx):
    """Any positive rollout/checkpoint/scalar/log step makes handoff ineligible."""
    run = ctx.here / 'runs/reasoning'
    evidence = []
    latest = run / 'checkpoints/latest_checkpointed_iteration.txt'
    if latest.exists() and int(latest.read_text()) > 0:
        evidence.append(str(latest))
    for parent, pattern in ((run / 'rollouts', '*.jsonl'), (run / 'checkpoints', 'global_step_*')):
        for p in parent.glob(pattern):
            value = p.stem if p.suffix == '.jsonl' else p.name.removeprefix('global_step_')
            if value.isdigit() and int(value) > 0:
                evidence.append(str(p))
    log = run / 'training.log'
    if log.exists():
        with log.open(errors='replace') as f:
            for line in f:
                if re.search(r'\bstep\s*[:=]\s*[1-9]\d*\s*(?:-|/|\b)', line):
                    evidence.append('training.log:' + line[:160]); break
    event_files = list((run / 'tensorboard').rglob('events.out.tfevents.*'))
    if event_files:
        from tensorboard.backend.event_processing.event_file_loader import EventFileLoader
        for p in event_files:
            for event in EventFileLoader(str(p)).Load():
                if event.step > 0 and event.summary.value:
                    evidence.append(str(p) + ':' + str(event.step)); break
    return evidence


def decision(ctx, plan):
    old = plan['old_driver']
    if not alive(old):
        return 'abort', 'Old driver is dead or identity changed; no automatic revival'
    state = read(ctx.runtime / 'experiment_state.json', {})
    if state.get('driver_pid') != old['pid'] or state.get('stage') in ('failed', 'interrupted', 'complete'):
        return 'abort', 'Old driver state is terminal or identity does not match'
    if state.get('driver_start_ticks') not in (None, old['start_ticks']):
        return 'abort', 'Recorded driver start_ticks mismatch'
    if reasoning_progress(ctx):
        return 'abort', 'Reasoning already has a logged/checkpoint/scalar step'
    if not perception_complete(ctx):
        return 'wait', 'Perception checkpoint 300 and validated HF export are not complete'
    if state.get('stage') == 'training' and state.get('teacher') == 'reasoning':
        return 'ready', 'Perception export complete; reasoning has no completed/logged step'
    if state.get('teacher') == 'generalist' or state.get('stage') in ('evaluation', 'complete'):
        return 'abort', 'Controller has passed the eligible handoff boundary'
    return 'wait', 'Waiting for old controller to enter training/reasoning'


def guardian_main(plan_path, read_fd, write_fd):
    """Own every STOP; pipe EOF resumes only our stops, including watcher SIGKILL."""
    plan = read(Path(plan_path)); old = plan['old_driver']; handles = {}
    def interrupted(*_):
        raise SystemExit(1)
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    reader = os.fdopen(read_fd, 'rb'); writer = os.fdopen(write_fd, 'wb', buffering=0)
    try:
        for line in reader:
            command = json.loads(line)
            p = old if command['op'] == 'driver' else command['identity']
            key = (p['pid'], p['start_ticks'])
            if key not in handles:
                fd = os.pidfd_open(p['pid'])
                current = identity(p['pid'])
                if not same(current, p):
                    os.close(fd); raise RuntimeError('Guardian identity mismatch')
                already = current['state'] in ('T', 't')
                if command['op'] == 'driver' and already:
                    os.close(fd); raise RuntimeError('Driver already stopped before handoff')
                handles[key] = (fd, not already)
                if not already:
                    signal.pidfd_send_signal(fd, signal.SIGSTOP)
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    current = identity(p['pid'])
                    if not same(current, p):
                        raise RuntimeError('Process vanished while stopping')
                    if current['state'] in ('T', 't'):
                        break
                    time.sleep(.01)
                else:
                    raise RuntimeError('Process did not stop')
            writer.write(json.dumps({'stopped_by_us': handles[key][1]}).encode() + b'\n')
    except BaseException as exc:
        with contextlib.suppress(BrokenPipeError, OSError):
            writer.write(json.dumps({'error': f'{type(exc).__name__}: {exc}'}).encode() + b'\n')
    finally:
        # Resume descendants before their controller. Never CONT an existing T state.
        for fd, owned_stop in reversed(list(handles.values())):
            if owned_stop:
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(fd, signal.SIGCONT)
            os.close(fd)
        reader.close(); writer.close()


class FreezeGuard:
    def __init__(self, plan_path):
        self.plan_path = plan_path
        self.child = None; self.control = None; self.reply = None

    def request(self, message):
        os.write(self.control, json.dumps(message).encode() + b'\n')
        if not select.select([self.reply], [], [], 8)[0]:
            raise RuntimeError('Freeze guardian acknowledgment timed out')
        data = b''
        while not data.endswith(b'\n'):
            chunk = os.read(self.reply, 1)
            if not chunk:
                raise RuntimeError('Freeze guardian exited without acknowledgment')
            data += chunk
        result = json.loads(data)
        if result.get('error'): raise RuntimeError(result['error'])
        return result

    def __enter__(self):
        cr, cw = os.pipe(); rr, rw = os.pipe()
        try:
            self.child = subprocess.Popen([sys.executable, str(SELF), '_guardian',
                '--plan', str(self.plan_path), '--read-fd', str(cr), '--write-fd', str(rw)],
                pass_fds=(cr, rw), stdin=subprocess.DEVNULL, start_new_session=True)
            self.control, self.reply = cw, rr
        except BaseException:
            os.close(cw); os.close(rr); raise
        finally:
            os.close(cr); os.close(rw)
        try:
            self.request({'op': 'driver'})
            return self
        except BaseException:
            self.release(); raise

    def freeze(self, process):
        return self.request({'op': 'owned_descendant', 'identity': process})['stopped_by_us']

    def release(self):
        if self.control is not None:
            os.close(self.control); self.control = None
        # Keep reply open until the guardian's finally block has resumed processes.
        if self.child:
            try:
                self.child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.child.terminate(); self.child.wait(timeout=5)
            self.child = None
        if self.reply is not None:
            os.close(self.reply); self.reply = None

    def __exit__(self, *_):
        self.release()


def reasoning_children(ctx, plan, cleaned=()):
    table = processes(); old = plan['old_driver']
    children = [p for p in table.values() if p['ppid'] == old['pid']]
    state = read(ctx.runtime / 'experiment_state.json', {})
    for p in children:
        if (p['cwd'] != str(ctx.root) or p['pgrp'] != p['pid'] or p['session'] != p['pid'] or
                len(p['argv']) != 6 or p['argv'][1:] !=
                [str(ctx.launcher), 'reasoning', '--gpus', '3,4,5,6', '--execute']):
            raise RuntimeError('Unexpected direct child; refusing to signal it: ' + json.dumps(p))
    if len(children) > 1:
        raise RuntimeError('Multiple reasoning launcher children; refusing ambiguous ownership')
    recorded = state.get('child_pid')
    if (recorded is not None and recorded not in [p['pid'] for p in children]
            and recorded not in {p['pid'] for p in cleaned if not alive(p)}):
        raise RuntimeError('State child PID disagrees with frozen direct-child enumeration')
    return children


def session_members(owner):
    """Only the launcher session and observed descendants are eligible."""
    if not alive(owner):
        raise RuntimeError('Reasoning launcher identity changed before session cleanup')
    table = processes()
    owned = {owner['pid']}
    # An escaped process group/session remains ours only with observed ancestry.
    while True:
        expanded = owned | {p['pid'] for p in table.values() if p['ppid'] in owned}
        if expanded == owned: break
        owned = expanded
    return [p for p in table.values() if p['session'] == owner['pid'] or p['pid'] in owned]


def stop_reasoning(ctx, owner, update, cancel_check, guardian):
    """Freeze the owned session, recheck progress, then terminate exact pidfds."""
    handles = {}; frozen = set(); committed = False
    try:
        # Freeze this owned session and descendants with observed parent ancestry.
        # Unprovable escaped/orphaned processes are never adopted; busy GPUs block launch.
        stable = 0
        for _ in range(20):
            cancel_check()
            before = len(handles)
            for p in session_members(owner):
                key = (p['pid'], p['start_ticks'])
                if key in handles:
                    continue
                fd = os.pidfd_open(p['pid'])
                if not same(identity(p['pid']), p):
                    os.close(fd); continue
                handles[key] = (fd, p)
                if p['state'] not in ('T', 't'):
                    if guardian.freeze(p): frozen.add(key)
            time.sleep(.05)
            stable = stable + 1 if len(handles) == before else 0
            if stable >= 2:
                break
        else:
            raise RuntimeError('Owned reasoning session did not stabilize')
        if reasoning_progress(ctx):
            raise RuntimeError('Reasoning progressed before frozen cleanup; leave the run intact')
        cancel_check()
        update('cleaning_reasoning', irreversible_started=True,
               reasoning_session=owner, owned_processes=[p for _, p in handles.values()])
        committed = True
        for fd, _ in handles.values():
            with contextlib.suppress(ProcessLookupError):
                signal.pidfd_send_signal(fd, signal.SIGTERM)
        for key, (fd, _) in handles.items():
            if key in frozen:
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(fd, signal.SIGCONT)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            if all(not alive(p) for _, p in handles.values()):
                return
            update('cleaning_reasoning', reason='Waiting for owned session processes to exit')
            time.sleep(.2)
        for fd, p in handles.values():
            if alive(p):
                with contextlib.suppress(ProcessLookupError):
                    signal.pidfd_send_signal(fd, signal.SIGKILL)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and any(alive(p) for _, p in handles.values()):
            time.sleep(.1)
        if any(alive(p) for _, p in handles.values()):
            raise RuntimeError('Owned reasoning session cleanup did not finish')
    finally:
        # On a reversible abort preserve already-stopped user processes; resume
        # only stops introduced here. On committed cleanup complete termination.
        for key, (fd, _) in handles.items():
            with contextlib.suppress(ProcessLookupError):
                if committed:
                    signal.pidfd_send_signal(fd, signal.SIGTERM)
                if key in frozen:
                    signal.pidfd_send_signal(fd, signal.SIGCONT)
            os.close(fd)


def gpu_idle(ctx):
    lines = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
              '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
    selected = {parts[1].strip() for line in lines if len(parts := line.split(',')) == 2
                and parts[0].strip() in {'3', '4', '5', '6'}}
    if len(selected) != 4:
        raise RuntimeError('Could not identify exactly physical GPUs 3-6')
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
               '--format=csv,noheader,nounits'], text=True, timeout=10).splitlines()
    busy = [line for line in rows if line.split(',')[0].strip() in selected]
    return not busy, busy


def execute_handoff(ctx, plan, update, cancelled, gpu_check=gpu_idle, after_freeze=None):
    verify_hashes(ctx, plan)
    outcome, why = decision(ctx, plan)
    if outcome != 'ready':
        raise RuntimeError(why)
    old = plan['old_driver']
    with FreezeGuard(ctx.plan) as guardian:
        update('driver_frozen', old_driver_stopped_by_handoff=True)
        cancelled()
        if after_freeze:
            after_freeze()
        outcome, why = decision(ctx, plan)
        if outcome != 'ready':
            raise RuntimeError('Frozen recheck: ' + why)
        verify_hashes(ctx, plan)
        children = reasoning_children(ctx, plan)
        update('children_verified', direct_children=children,
               state_child_pid=read(ctx.runtime / 'experiment_state.json', {}).get('child_pid'))
        for child in children:
            stop_reasoning(ctx, child, update, cancelled, guardian)
        cancelled()
        # Driver cannot create a child between this enumeration and termination.
        # A zombie child is expected after cleanup and excluded by processes().
        if reasoning_children(ctx, plan, cleaned=children):
            raise RuntimeError('Reasoning direct child remains alive after cleanup')
        if reasoning_progress(ctx):
            raise RuntimeError('Reasoning progress appeared; do not launch a replacement')
        verify_hashes(ctx, plan)
        update('terminating_old_driver', irreversible_started=True)
        send_exact(old, signal.SIGTERM)
    update('waiting_old_driver_exit', old_driver_stopped_by_handoff=False)
    deadline = time.monotonic() + 30
    while alive(old) and time.monotonic() < deadline:
        cancelled(); update('waiting_old_driver_exit'); time.sleep(.2)
    if alive(old):
        raise RuntimeError('Old driver did not exit after TERM+CONT; no new launch')
    deadline = time.monotonic() + 30
    while not lock_free(ctx.driver_lock) and time.monotonic() < deadline:
        cancelled(); update('waiting_driver_lock'); time.sleep(.2)
    if not lock_free(ctx.driver_lock):
        raise RuntimeError('Old driver flock still held; no new launch')
    deadline = time.monotonic() + 45
    while True:
        cancelled(); quiet, busy = gpu_check(ctx)
        update('waiting_gpu_idle', busy_training_gpus=busy)
        if quiet:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError('GPUs 3-6 remain busy; no processes were killed by GPU identity')
        time.sleep(.5)
    verify_hashes(ctx, plan)
    if reasoning_progress(ctx):
        raise RuntimeError('Reasoning progress appeared before new launch')
    if not lock_free(ctx.driver_lock):
        raise RuntimeError('Another controller acquired the driver lock; no new launch')
    cancelled()
    # Persist before Popen: a crash here is ambiguous and must not auto-retry.
    update('launch_requested', launch_requested=True, new_launch_argv=plan['new_launch_argv'])
    result = subprocess.run(plan['new_launch_argv'], cwd=ctx.root, capture_output=True,
                            text=True, check=True, timeout=30)
    data = json.loads(result.stdout.strip().splitlines()[-1])
    new_pid = int(data['pid']); new = identity(new_pid)
    exec_deadline = time.monotonic() + 5
    while not expected_driver(ctx, new) and time.monotonic() < exec_deadline:
        time.sleep(.05); new = identity(new_pid)
    if not expected_driver(ctx, new) or new_pid == old['pid']:
        raise RuntimeError('Replacement launch did not return a matching new driver identity')
    update('verifying_new_driver', new_driver=new, launch_output=data)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        state = read(ctx.runtime / 'experiment_state.json', {})
        if not alive(new):
            raise RuntimeError('New driver exited before takeover confirmation')
        if state.get('driver_pid') == new_pid:
            if state.get('stage') in ('failed', 'interrupted'):
                raise RuntimeError('Replacement controller reported failure')
            if not lock_free(ctx.driver_lock):
                update('complete', new_driver=new, completed_at=now(),
                       old_driver_stopped_by_handoff=False)
                return
        update('verifying_new_driver'); time.sleep(.2)
    raise RuntimeError('New controller state/flock confirmation timed out')


def run(ctx):
    deadline = time.monotonic() + 5
    while True:
        try:
            lock_context = lock_file(ctx.lock)
            lock_context.__enter__()
            break
        except BlockingIOError:
            if time.monotonic() >= deadline: raise
            time.sleep(.05)
    try:
        plan = read(ctx.plan)
        if not plan or plan['here'] != str(ctx.here):
            raise RuntimeError('Missing/mismatched prepared plan')
        previous = read(ctx.state, {})
        if previous.get('stage') in TERMINAL or previous.get('irreversible_started') or previous.get('launch_requested'):
            raise RuntimeError('Handoff is terminal or has irreversible history; never auto-repeat it')
        me = identity(os.getpid())
        state = dict(previous, pid=os.getpid(), start_ticks=me['start_ticks'], argv=me['argv'],
                     old_driver=plan['old_driver'], plan_id=plan['id'], started_at=now())
        cancelled_flag = False
        def on_signal(*_):
            nonlocal cancelled_flag
            cancelled_flag = True
        handlers = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGINT)}
        def cancelled():
            if cancelled_flag:
                raise InterruptedError('Handoff cancelled')
        def update(stage, **fields):
            previous_stage = state.get('stage')
            state.update(stage=stage, heartbeat_epoch=time.time(), at=now(), **fields)
            atomic(ctx.state, state)
            if previous_stage != stage:
                ctx.event('transition', from_stage=previous_stage, stage=stage, **fields)
        try:
            update('watching')
            while True:
                cancelled(); verify_hashes(ctx, plan)
                outcome, reason = decision(ctx, plan)
                update('watching', decision=outcome, reason=reason)
                if outcome == 'abort':
                    update('aborted'); ctx.event('aborted', reason=reason); return
                if outcome == 'ready':
                    execute_handoff(ctx, plan, update, cancelled)
                    ctx.event('complete', new_driver=state.get('new_driver')); return
                for _ in range(20):
                    cancelled(); time.sleep(.25)
        except InterruptedError as exc:
            update('cancelled', error=str(exc), old_driver_stopped_by_handoff=False)
            ctx.event('cancelled', error=str(exc))
        except BaseException as exc:
            update('failed', error=f'{type(exc).__name__}: {exc}', old_driver_stopped_by_handoff=False)
            ctx.event('failed', error=state['error']); raise
        finally:
            for s, handler in handlers.items(): signal.signal(s, handler)
    finally:
        lock_context.__exit__(None, None, None)


def launch(ctx):
    existing = status(ctx)
    if existing.get('watcher_running'):
        return {'already_running': True, 'pid': existing['pid']}
    with lock_file(ctx.lock):
        plan = read(ctx.plan)
        if not plan:
            raise RuntimeError('Run prepare first')
        verify_hashes(ctx, plan)
        state = read(ctx.state, {})
        if state.get('stage') in TERMINAL or state.get('irreversible_started') or state.get('launch_requested'):
            raise RuntimeError('One-shot watcher is terminal or has irreversible history')
        if state.get('pid'):
            p = identity(state['pid'])
            if p and p['start_ticks'] == state.get('start_ticks') and (str(SELF) in p['argv'] or state.get('stage') == 'launching'):
                return {'already_running': True, 'pid': p['pid']}
        log = ctx.home / 'watcher.log'
        with log.open('ab') as f:
            child = subprocess.Popen([sys.executable, str(SELF), 'run', '--plan', str(ctx.plan)], cwd=ctx.root,
                stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                start_new_session=True, env=dict(os.environ, PYTHONUNBUFFERED='1',
                    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        # Reserve the child identity before releasing the launch lock.
        child_identity = identity(child.pid)
        if not child_identity:
            raise RuntimeError('Detached watcher exited during launch')
        state.update(stage='launching', pid=child.pid, start_ticks=child_identity['start_ticks'],
                     argv=[sys.executable, str(SELF), 'run', '--plan', str(ctx.plan)], at=now(), heartbeat_epoch=time.time())
        atomic(ctx.state, state)
        # run() obtains the lock only after this function releases it below.
        return {'pid': child.pid, 'log': str(log), 'plan_id': plan['id']}


def status(ctx):
    state = read(ctx.state, {'stage': 'not_prepared'})
    p = identity(state.get('pid', -1))
    state['watcher_running'] = bool(p and p['start_ticks'] == state.get('start_ticks') and str(SELF) in p['argv'])
    state['heartbeat_age_seconds'] = max(0, time.time() - state.get('heartbeat_epoch', 0))
    return state


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=['prepare', 'launch', 'run', 'status', 'cancel', 'self-test', '_guardian'])
    ap.add_argument('--old-pid', type=int)
    ap.add_argument('--old-start-ticks', type=int)
    ap.add_argument('--plan', type=Path)
    ap.add_argument('--read-fd', type=int)
    ap.add_argument('--write-fd', type=int)
    args = ap.parse_args(); ctx = Context()
    if args.plan and args.action in ('run', 'launch', 'status', 'cancel'):
        ctx = Context(read(args.plan)['here'])
        if args.plan.resolve() != ctx.plan:
            ap.error('The supplied plan path does not match its workspace')
    if args.action == '_guardian':
        guardian_main(args.plan, args.read_fd, args.write_fd); return
    if args.action == 'prepare':
        if args.old_pid is None or args.old_start_ticks is None:
            ap.error('prepare requires --old-pid and --old-start-ticks')
        result = prepare(ctx, args.old_pid, args.old_start_ticks)
    elif args.action == 'launch': result = launch(ctx)
    elif args.action == 'run': run(ctx); return
    elif args.action == 'status': result = status(ctx)
    elif args.action == 'cancel':
        s = status(ctx)
        if not s['watcher_running']:
            result = {'cancelled': False, 'reason': 'No matching live watcher'}
        else:
            p = identity(s['pid'])
            if not p or p['start_ticks'] != s['start_ticks'] or str(SELF) not in p['argv']:
                raise RuntimeError('Watcher identity changed before cancel')
            send_exact(p, signal.SIGTERM)
            result = {'cancel_requested': True, 'pid': p['pid']}
    else:
        result = self_test()
    print(json.dumps(result, ensure_ascii=False, indent=2))


# CPU-only integration fixtures are below; they use temporary workspaces and
# fake driver/launcher scripts, with an injected GPU-idle predicate.

_FAKE_DRIVER = r'''
import fcntl, hashlib, json, os, signal, subprocess, sys, tempfile, time
from pathlib import Path
here=Path(__file__).resolve().parent; root=here.parents[1]; runtime=here/'runtime'
state=runtime/'experiment_state.json'; cfg=json.loads((here/'fixture.json').read_text())
def write(value):
    temp=state.with_suffix('.tmp'); temp.write_text(json.dumps(value)); temp.replace(state)
if sys.argv[1]=='launch':
    (here/'replacement').write_text('new')
    p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'run'],cwd=root,
        start_new_session=True,stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    deadline=time.monotonic()+5
    while time.monotonic()<deadline:
        try:
            if json.loads(state.read_text()).get('driver_pid')==p.pid: break
        except (FileNotFoundError,json.JSONDecodeError): pass
        time.sleep(.01)
    print(json.dumps({'pid':p.pid})); sys.exit(0)
lock=Path(tempfile.gettempdir())/('sharedexpert-driver-'+hashlib.sha256(str(runtime).encode()).hexdigest()+'.lock')
f=lock.open('a'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
child=None
value={'driver_pid':os.getpid(),'stage':'training','teacher':'reasoning','child_pid':None}
def stopped(*args):
    value['stage']='interrupted'; write(value); raise SystemExit(0)
signal.signal(signal.SIGTERM,stopped); signal.signal(signal.SIGINT,stopped)
if (here/'replacement').exists():
    value.update(stage='preflight',teacher='reasoning'); write(value)
else:
    value['stage']=cfg.get('stage','training'); value['teacher']=cfg.get('teacher','reasoning'); write(value)
    if cfg.get('child',True):
        argv=[sys.executable,str(here/'training/launch.py'),'reasoning','--gpus','3,4,5,6','--execute']
        if cfg.get('unexpected'): argv=[sys.executable,'-c','import time; time.sleep(300)']
        child=subprocess.Popen(argv,cwd=root,start_new_session=True)
        (here/'child_pid').write_text(str(child.pid))
        if not cfg.get('empty_child_state'): value['child_pid']=child.pid; write(value)
(here/'ready').write_text(str(os.getpid()))
while True: time.sleep(.05)
'''

_FAKE_LAUNCHER = r'''
import os, subprocess, sys, time
from pathlib import Path
here=Path(__file__).resolve().parent.parent
p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'],start_new_session=True)
(here/'descendant_pid').write_text(str(p.pid))
while True: time.sleep(.05)
'''


def wait_until(predicate, seconds=6, message='Condition timed out'):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.02)
    raise AssertionError(message)


class CpuFixture:
    def __init__(self, **options):
        self.temp = tempfile.TemporaryDirectory(prefix='sharedexpert-handoff-cpu-')
        self.here = Path(self.temp.name) / 'workspace/experiments/sharedexpert_c'
        (self.here / 'training').mkdir(parents=True)
        (self.here / 'runtime').mkdir()
        self.ctx = Context(self.here)
        self.ctx.driver.write_text(_FAKE_DRIVER)
        self.ctx.launcher.write_text(_FAKE_LAUNCHER)
        self.ctx.check.write_text('# fake CPU-only check\n')
        atomic(self.ctx.cpu_receipt, {'status':'cpu_prepared', 'gpu_started':False,
               'source_sha256':{str(p):sha(p) for p in (self.ctx.check,self.ctx.launcher)}})
        atomic(self.here / 'fixture.json', options)
        self.proc = subprocess.Popen([sys.executable, str(self.ctx.driver), 'run'], cwd=self.ctx.root,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        self.extra = []
        wait_until(lambda: (self.here / 'ready').exists())
        if options.get('child', True) and not options.get('unexpected'):
            wait_until(lambda: (self.here / 'descendant_pid').exists())
        self.old = identity(self.proc.pid)
        self.export()
        self.plan = prepare(self.ctx, self.old['pid'], self.old['start_ticks'])
        self.state = read(self.ctx.state)

    def export(self):
        run = self.here / 'runs/perception'
        source = run / 'checkpoints/global_step_300/actor'
        source.mkdir(parents=True, exist_ok=True)
        atomic(source / 'fsdp_config.json', {})
        for kind in ('model','optim','extra_state'):
            for rank in range(4):
                (source / f'{kind}_world_size_4_rank_{rank}.pt').write_bytes(b'fixture')
        (run / 'checkpoints/latest_checkpointed_iteration.txt').write_text('300')
        target = run / 'hf/global_step_300'; target.mkdir(parents=True, exist_ok=True)
        atomic(target / 'config.json', {})
        (target / 'model.safetensors').write_bytes(b'fixture')
        atomic(target / 'sharedexpert_export_complete.json', {'step':300,'source':str(source)})

    def update(self, stage, **fields):
        self.state.update(stage=stage, heartbeat_epoch=time.time(), **fields)
        atomic(self.ctx.state, self.state)

    def child(self):
        p = self.here / 'child_pid'
        return identity(int(p.read_text())) if p.exists() else None

    def descendant(self):
        p = self.here / 'descendant_pid'
        return identity(int(p.read_text())) if p.exists() else None

    def execute(self, **kwargs):
        execute_handoff(self.ctx, self.plan, self.update, lambda: None,
                        gpu_check=lambda _: (True, []), **kwargs)

    def close(self):
        owned = []
        for p in (self.old, self.child(), self.descendant(), self.state.get('new_driver'), *self.extra):
            if p and alive(p): owned.append(p)
        for p in owned:
            with contextlib.suppress(ProcessLookupError, RuntimeError):
                send_exact(p, signal.SIGTERM); send_exact(p, signal.SIGCONT)
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and any(alive(p) for p in owned): time.sleep(.02)
        for p in owned:
            if alive(p):
                with contextlib.suppress(ProcessLookupError, RuntimeError): send_exact(p, signal.SIGKILL)
        with contextlib.suppress(subprocess.TimeoutExpired): self.proc.wait(timeout=2)
        self.temp.cleanup()
        for p in (self.ctx.lock, self.ctx.driver_lock):
            with contextlib.suppress(FileNotFoundError): p.unlink()

    def __enter__(self): return self
    def __exit__(self, *_): self.close()


def must_raise(fn, expected=(RuntimeError, InterruptedError)):
    try: fn()
    except expected: return
    raise AssertionError('Expected safe refusal did not occur')


def self_test():
    """No real run paths, signals, GPU queries, or experiment state writes."""
    checks = []
    def checked(name, fn):
        started = time.monotonic(); fn()
        checks.append({'name':name,'passed':True,'seconds':round(time.monotonic()-started,3)})

    def eligibility():
        with CpuFixture() as f:
            assert decision(f.ctx, f.plan)[0] == 'ready'
            bad = dict(f.plan, old_driver=dict(f.old, start_ticks=f.old['start_ticks']+1))
            assert decision(f.ctx, bad)[0] == 'abort'
            must_raise(lambda: prepare(f.ctx, f.old['pid'], f.old['start_ticks']+1))
            latest=f.here/'runs/perception/checkpoints/latest_checkpointed_iteration.txt'
            latest.write_text('299'); assert decision(f.ctx, f.plan)[0]=='wait'
            f.export()
            marker=f.here/'runs/perception/hf/global_step_300/sharedexpert_export_complete.json'
            atomic(marker,{'step':300,'source':'/wrong/source'})
            assert decision(f.ctx,f.plan)[0]=='wait'
            f.export()
            state=read(f.ctx.runtime/'experiment_state.json'); state['stage']='failed'
            atomic(f.ctx.runtime/'experiment_state.json',state)
            assert decision(f.ctx,f.plan)[0]=='abort'
            assert alive(f.old) and alive(f.child())
    checked('identity_export_and_terminal_state_refusal',eligibility)

    def progress():
        for kind in ('rollout','checkpoint','log'):
            with CpuFixture() as f:
                run=f.here/'runs/reasoning'; run.mkdir(parents=True,exist_ok=True)
                if kind=='rollout':
                    (run/'rollouts').mkdir(); (run/'rollouts/1.jsonl').write_text('{}\n')
                elif kind=='checkpoint': (run/'checkpoints/global_step_1').mkdir(parents=True)
                else: (run/'training.log').write_text('step:1 - actor/pg_loss:0.0\n')
                assert decision(f.ctx,f.plan)[0]=='abort'
                must_raise(f.execute)
                assert alive(f.old) and alive(f.child())
    checked('reasoning_any_step_refuses_without_signals',progress)

    def parent_and_hash():
        with CpuFixture(teacher='perception') as f:
            assert decision(f.ctx,f.plan)[0]=='wait'
        with CpuFixture() as f:
            f.ctx.check.write_text('# mutated\n')
            must_raise(f.execute)
            assert identity(f.old['pid'])['state'] not in ('T','t')
        with CpuFixture(child=False) as f:
            send_exact(f.old,signal.SIGTERM); f.proc.wait(timeout=3)
            assert decision(f.ctx,f.plan)[0]=='abort'
    checked('wait_perception_child_and_hash_changed_or_dead_refusal',parent_and_hash)

    def pre_stopped():
        with CpuFixture() as f:
            send_exact(f.old,signal.SIGSTOP)
            wait_until(lambda: identity(f.old['pid'])['state']=='T')
            must_raise(lambda: prepare(f.ctx,f.old['pid'],f.old['start_ticks']))
            must_raise(f.execute)
            assert identity(f.old['pid'])['state']=='T'
            send_exact(f.old,signal.SIGCONT)
    checked('preexisting_driver_stop_is_never_resumed',pre_stopped)

    def cancel():
        with CpuFixture() as f:
            def abort(): raise InterruptedError('test cancel')
            must_raise(lambda:f.execute(after_freeze=abort))
            wait_until(lambda: identity(f.old['pid'])['state'] not in ('T','t'))
            assert alive(f.child()) and not f.state.get('irreversible_started')
            child=f.child(); descendant=f.descendant()
            send_exact(descendant,signal.SIGSTOP)
            wait_until(lambda: identity(descendant['pid'])['state']=='T')
            with FreezeGuard(f.ctx.plan) as guard:
                assert guard.freeze(child)
                assert not guard.freeze(descendant)
            wait_until(lambda: identity(child['pid'])['state'] not in ('T','t'))
            assert identity(descendant['pid'])['state']=='T'
            send_exact(descendant,signal.SIGCONT)
    checked('cancel_restores_our_stops_preserves_user_stops',cancel)

    def unexpected():
        with CpuFixture(unexpected=True) as f:
            child=f.child(); must_raise(f.execute)
            assert alive(child) and alive(f.old)
            assert identity(f.old['pid'])['state'] not in ('T','t')
    checked('unexpected_direct_child_is_untouched',unexpected)

    def successful(empty=False, child=True):
        with CpuFixture(empty_child_state=empty,child=child) as f:
            original=f.child(); descendant=f.descendant()
            foreign=subprocess.Popen([sys.executable,'-c','import time; time.sleep(300)'],
                                     cwd=f.ctx.root,start_new_session=True)
            wait_until(lambda: identity(foreign.pid) and '-c' in identity(foreign.pid)['argv'])
            outsider=identity(foreign.pid); f.extra.append(outsider)
            f.execute()
            assert f.state['stage']=='complete' and f.state['launch_requested']
            assert not alive(f.old) and alive(f.state['new_driver'])
            assert not lock_free(f.ctx.driver_lock)
            if child:
                assert not alive(original) and not alive(descendant)
                assert len(f.state['direct_children'])==1
                assert (f.state['state_child_pid'] is None)==empty
            else: assert not f.state['direct_children']
            assert alive(outsider)
            must_raise(lambda:launch(f.ctx)); must_raise(lambda:prepare(f.ctx,f.state['new_driver']['pid'],f.state['new_driver']['start_ticks']))
            send_exact(outsider,signal.SIGTERM); foreign.wait(timeout=3)
    checked('normal_handoff_tree_cleanup_foreign_untouched_once_only',lambda:successful())
    checked('popen_before_state_child_pid_is_found_and_cleaned',lambda:successful(empty=True))
    checked('pre_popen_empty_child_boundary_is_safe',lambda:successful(child=False))

    def killed_watcher():
        with CpuFixture() as f:
            ready=f.here/'guard_ready.json'; child=f.child(); descendant=f.descendant()
            code='''import importlib.util,json,sys,time
from pathlib import Path
spec=importlib.util.spec_from_file_location('handoff',sys.argv[1]); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
plan=Path(sys.argv[2]); ready=Path(sys.argv[3]); child=json.loads(sys.argv[4]); descendant=json.loads(sys.argv[5])
with m.FreezeGuard(plan) as guard:
    guard.freeze(child); guard.freeze(descendant)
    ready.write_text(json.dumps({'guardian_pid':guard.child.pid}))
    time.sleep(300)
'''
            watcher=subprocess.Popen([sys.executable,'-c',code,str(SELF),str(f.ctx.plan),str(ready),json.dumps(child),json.dumps(descendant)],
                                     cwd=f.ctx.root,start_new_session=True)
            try:
                wait_until(lambda:ready.exists())
                assert identity(f.old['pid'])['state']=='T'
                watcher.kill(); watcher.wait(timeout=3)
                wait_until(lambda: all(identity(p['pid']) and identity(p['pid'])['state'] not in ('T','t') for p in (f.old,child,descendant)),seconds=5)
            finally:
                if watcher.poll() is None: watcher.kill(); watcher.wait(timeout=3)
    checked('watcher_sigkill_pipe_guardian_restores_driver_and_descendants',killed_watcher)

    def durable_wait_cancel():
        with CpuFixture() as f:
            latest=f.here/'runs/perception/checkpoints/latest_checkpointed_iteration.txt'
            latest.write_text('299')
            launched=launch(f.ctx)
            wait_until(lambda:status(f.ctx).get('stage')=='watching')
            watcher=identity(launched['pid']); f.extra.append(watcher)
            assert status(f.ctx)['watcher_running'] and not lock_free(f.ctx.lock)
            duplicate=launch(f.ctx)
            assert duplicate.get('already_running') and duplicate['pid']==launched['pid']
            output=subprocess.check_output([sys.executable,str(SELF),'status','--plan',str(f.ctx.plan)],text=True)
            assert json.loads(output)['watcher_running']
            subprocess.check_output([sys.executable,str(SELF),'cancel','--plan',str(f.ctx.plan)],text=True)
            wait_until(lambda:not alive(watcher))
            assert status(f.ctx)['stage']=='cancelled' and lock_free(f.ctx.lock)
            assert alive(f.old) and alive(f.child())
            events=[json.loads(line) for line in f.ctx.events.read_text().splitlines()]
            assert any(e.get('stage')=='cancelled' for e in events)
    checked('detached_watcher_cli_lock_duplicate_status_cancel',durable_wait_cancel)

    return {'passed':True,'cpu_only':True,'real_driver_signalled':False,'gpu_queried':False,
            'created_at':now(),'script_sha256':sha(SELF),'python':sys.executable,'checks':checks}


if __name__ == '__main__':
    main()
