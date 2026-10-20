#!/usr/bin/env python3
"""Durable, resumable sequence: three 300-step teachers, exports, then evaluation."""
from __future__ import annotations
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import tempfile
from zoneinfo import ZoneInfo

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
RUNTIME=HERE/'runtime'
STATE=RUNTIME/'experiment_state.json'
TRAIN_PY=HERE/'.venv-train/bin/python'
EVAL_PY=ROOT/'pre-lab/venv/bin/python'
ROLES=('perception','reasoning','generalist')
GPUS='3,4,5,6'
MONITOR_PY=Path('/opt/venv/main/bin/python3')
MONITOR_STATE=HERE/'monitoring/state.json'
_monitor_error_last=None
_monitor_error_time=0.0


def monitor_health():
    """Inspect only our monitor's saved process identity, without importing it."""
    try:
        state=json.loads(MONITOR_STATE.read_text())
        proc=Path('/proc')/str(int(state['pid']))
        args=proc.joinpath('cmdline').read_bytes().split(b'\0')
        ticks=int(proc.joinpath('stat').read_text().rsplit(')',1)[1].split()[19])
        alive=(str(HERE/'monitor_training.py').encode() in args and b'run' in args
               and ticks==state.get('start_ticks'))
        age=max(0,time.time()-state.get('heartbeat_epoch',0))
        return {'running':alive,'pid':state['pid'],'heartbeat_age_seconds':age,
                'healthy':alive and age<120,'render_status':state.get('render_status'),
                'latest_metric_steps':state.get('latest_metric_steps',{}),
                'alert_count':state.get('alert_count',0)}
    except (OSError,ValueError,KeyError,IndexError,TypeError):
        return {'running':False,'healthy':False}


def ensure_monitor(required=False):
    """Restart a dead monitor; monitoring errors never terminate a live trainer."""
    global _monitor_error_last,_monitor_error_time
    try:
        health=monitor_health()
        if health['running'] and not health['healthy']:
            raise RuntimeError(f'Monitor heartbeat is stale: {health}')
        if not health['running']:
            result=subprocess.run([str(MONITOR_PY),str(HERE/'monitor_training.py'),'launch'],
                cwd=ROOT,capture_output=True,text=True,check=True,timeout=15)
            print('Monitor launch: '+result.stdout.strip(),flush=True)
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                health=monitor_health()
                if health['healthy']: break
                time.sleep(.25)
            if not health['healthy']:
                raise RuntimeError('Monitor did not produce a live heartbeat after launch')
        _monitor_error_last=None
        return health
    except Exception as exc:
        message=f'{type(exc).__name__}: {exc}'
        if message!=_monitor_error_last or time.time()-_monitor_error_time>=120:
            record={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'error':message}
            with (RUNTIME/'monitor_supervision_errors.jsonl').open('a') as log:
                log.write(json.dumps(record)+'\n')
            print('Monitor supervision: '+message,flush=True)
            _monitor_error_last,_monitor_error_time=message,time.time()
        if required: raise
        return {'running':False,'healthy':False,'error':message}


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''): digest.update(block)
    return digest.hexdigest()


def execution_budget():
    """An explicit endpoint overlay preserves the frozen legacy validation chain."""
    common=json.loads((HERE/'training/config/common.json').read_text())
    path=HERE/'training/config/execution_budget.json'
    if not path.exists():
        return common
    budget=json.loads(path.read_text())
    if (budget.get('steps_per_teacher')!=100 or budget.get('evaluation_step')!=100
            or budget.get('base_config_sha256')!=file_sha(HERE/'training/config/common.json')
            or budget.get('legacy_launch_sha256')!=file_sha(HERE/'training/launch.py')
            or budget.get('launch_100_sha256')!=file_sha(HERE/'training/launch_100.py')
            or common['steps']!=300 or common['batch_size']!=30 or common['group_size']!=8):
        raise RuntimeError('100-step execution overlay does not match the frozen base')
    return dict(common,steps=100)


def verify_execution_budget():
    if execution_budget()['steps']==300:
        return
    receipt=json.loads((HERE/'logs/budget100_validation.json').read_text())
    allowed={'trainer.total_training_steps':{'before':300,'after':100}}
    if (receipt.get('passed') is not True
            or set(receipt.get('effective_config_diffs',{}))!=set(ROLES)
            or any(value!=allowed for value in receipt['effective_config_diffs'].values())
            or receipt.get('constant_lr_with_unchanged_warmup') is not True):
        raise RuntimeError('100-step budget has no narrowly scoped CPU validation')
    required={'training/launch.py','training/launch_100.py',
              'training/config/common.json','training/config/execution_budget.json'}
    if not required.issubset(receipt.get('validated_artifacts',{})):
        raise RuntimeError('100-step validation artifact manifest is incomplete')
    for name,expected in receipt['validated_artifacts'].items():
        path=(HERE/name).resolve()
        if not path.is_relative_to(HERE) or file_sha(path)!=expected:
            raise RuntimeError(f'100-step budget artifact changed: {name}')


def checkpoint_is_complete(role,step):
    run=HERE/'runs'/role
    source=run/f'checkpoints/global_step_{step}/actor'
    latest=run/'checkpoints/latest_checkpointed_iteration.txt'
    if not latest.exists() or latest.read_text().strip()!=str(step):
        return False
    required=[source/'fsdp_config.json',source.parent/'data.pt']
    required += [source/f'{kind}_world_size_4_rank_{rank}.pt'
                 for kind in ('model','optim','extra_state') for rank in range(4)]
    return all(path.is_file() and path.stat().st_size>0 for path in required)


def verify_unused_topk_transition(batch_transition):
    """Accept only the separately validated removal of unused RL distillation work."""
    receipt=json.loads((HERE/'logs/unused_topk_transition_validation.json').read_text())
    allowed={
        'actor_rollout_ref.actor.self_distillation.full_logit_distillation': {'before':True,'after':False},
        'actor_rollout_ref.actor.self_distillation.distillation_topk': {'before':100,'after':None},
    }
    required_checks=(
        'vanilla_loss_and_rollout_correction_unchanged',
        'real_log_probs_and_all_parameter_gradients_validated',
        'repeat_noise_baseline_measured',
        'warm_adam_second_update_completed',
        'second_update_second_micro_has_adam_and_accumulated_gradients',
        'real_response_8192_tokens',
        'no_probe_cuda_or_host_oom',
    )
    if (receipt.get('passed') is not True
            or receipt.get('old_launch_sha256')!=batch_transition['new_launch_sha256']
            or receipt.get('new_launch_sha256')!=file_sha(HERE/'training/launch.py')
            or receipt.get('common_sha256')!=file_sha(HERE/'training/config/common.json')
            or any(receipt.get('checks',{}).get(key) is not True for key in required_checks)
            or set(receipt.get('effective_config_diffs',{}))!=set(ROLES)
            or any(changes!=allowed for changes in receipt.get('effective_config_diffs',{}).values())):
        raise RuntimeError('Unused top-k memory fix lacks the narrowly scoped validated transition')
    artifacts=receipt.get('validated_artifacts',{})
    required_artifacts={
        'training/launch.py', 'training/unused_topk_probe.py',
        'logs/unused_topk_probe/report.json', 'logs/memory_fix_cpu_validation.log',
        'logs/unused_topk_probe/supervisor.json',
    }
    if not required_artifacts.issubset(artifacts):
        raise RuntimeError('Unused top-k validation artifacts are incomplete')
    for relative,sha in artifacts.items():
        path=(HERE/relative).resolve()
        if not path.is_relative_to(HERE) or file_sha(path)!=sha:
            raise RuntimeError(f'Unused top-k validation artifact changed: {relative}')
    gpu_report=json.loads((HERE/'logs/unused_topk_probe/report.json').read_text())
    if gpu_report.get('passed') is not True:
        raise RuntimeError('Unused top-k actual GPU validation did not pass')
    supervisor=json.loads((HERE/'logs/unused_topk_probe/supervisor.json').read_text())
    if (supervisor.get('status')!='complete' or supervisor.get('returncode')!=0
            or supervisor.get('guard_triggered',False)
            or any(supervisor.get('memory_events_delta',{}).get(key)!=0 for key in ('oom','oom_kill'))):
        raise RuntimeError('Unused top-k four-rank probe did not exit cleanly without memory faults')
    return receipt


def verify_preparation():
    prepared=HERE/'data_pipeline/prepared'
    audit=json.loads((prepared/'validation.json').read_text())
    expected={
        'perception':{'tallyqa':4800,'clevr':4800,'refcoco':2400},
        'reasoning':{'geometry3k':4800,'mmk12':2400,'virl39k':4800},
        'generalist':{'vqav2':4800,'docvqa':600,'chartqa':600,'geometry3k':960,
                      'mmk12':480,'virl39k':960,'tallyqa':960,'clevr':960,'refcoco':480,'if_format':1200},
    }
    for role,counts in expected.items():
        actual=audit[role]
        if actual['rows']!=12000 or actual['source_counts']!=counts:
            raise RuntimeError(f'{role}: final data budget/mixture differs from agreed protocol')
        if actual['sha256']!=file_sha(prepared/(role+'.parquet')):
            raise RuntimeError(f'{role}: data changed after final audit')
        if actual['eval_image_overlap'] or actual['known_virl_original_test_leakage']:
            raise RuntimeError(f'{role}: data contamination check failed')
    images=audit['images']
    if not images['all_exist'] or images['hashes_recomputed']!=images['distinct_paths']:
        raise RuntimeError('Full training-image integrity validation is incomplete')
    if audit['if_image_overlap_with_other_generalist_sources']:
        raise RuntimeError('IF image partition validation failed')
    suite=HERE/'evaluation/suite.jsonl'
    if audit['eval_manifests'].get(str(suite.resolve()))!=file_sha(suite):
        raise RuntimeError('Evaluation suite changed after training-data decontamination')
    gate=json.loads((RUNTIME/'smoke_gate.json').read_text())
    if gate.get('status')!='complete':
        raise RuntimeError('Actual GPU RL smoke + checkpoint + export have not completed')
    smoke=json.loads((HERE/'logs/smoke_validation.json').read_text())
    if smoke.get('passed') is not True:
        raise RuntimeError('Actual smoke gradients/rewards/rollout budget have not passed validation')
    protocol=json.loads((HERE/'runs/perception_smoke/protocol.json').read_text())
    if protocol['reward_sha256']!=file_sha(HERE/'rewards.py'):
        raise RuntimeError('Reward changed after GPU smoke')
    current_protocol=json.loads((HERE/'training/config/common.json').read_text())
    batch_transition=None
    memory_transition=None
    if current_protocol!=protocol['protocol']:
        # The user changed only the formal prompt batch from 40 to 30.
        # Retain the original four-prompt GPU smoke as historical evidence;
        # a separate CPU receipt proves its effective execution is unchanged.
        # The complete formal batch-30 GPU step is validated independently.
        expected_protocol=dict(protocol['protocol'],batch_size=30)
        if protocol['protocol'].get('batch_size')!=40 or current_protocol!=expected_protocol:
            raise RuntimeError('Training hyperparameters changed outside the authorized batch30 transition')
        batch_transition=json.loads((HERE/'logs/batch30_transition_validation.json').read_text())
        if (batch_transition.get('passed') is not True
                or batch_transition.get('effective_smoke_overrides_identical') is not True
                or batch_transition.get('effective_smoke_resolved_config_identical') is not True
                or batch_transition.get('old_smoke_protocol_sha256')!=file_sha(HERE/'runs/perception_smoke/protocol.json')
                or batch_transition.get('common_sha256')!=file_sha(HERE/'training/config/common.json')
                or batch_transition.get('old_launch_sha256')!=protocol['training_code_sha256']['launch.py']
                or batch_transition.get('cpu_validation_log_sha256')!=file_sha(HERE/'logs/batch30_cpu_validation.log')):
            raise RuntimeError('Batch30 configuration transition has no matching validated receipt')
        if batch_transition.get('new_launch_sha256')!=file_sha(HERE/'training/launch.py'):
            memory_transition=verify_unused_topk_transition(batch_transition)
    for name,sha in protocol['training_code_sha256'].items():
        if file_sha(HERE/'training'/name)!=sha:
            if not (name=='launch.py' and batch_transition
                    and file_sha(HERE/'training'/name)==(
                        memory_transition['new_launch_sha256'] if memory_transition
                        else batch_transition['new_launch_sha256'])):
                raise RuntimeError(f'Training code changed after GPU smoke: {name}')
    diff=subprocess.check_output(['git','-C',str(ROOT/'pre-lab/repo'),'diff','HEAD','--'],text=True)
    if hashlib.sha256(diff.encode()).hexdigest()!=protocol['repository_diff_sha256']:
        raise RuntimeError('Training repository patch changed after GPU smoke')
    for name,sha in protocol.get('training_environment_sha256',{}).items():
        if file_sha(HERE/'logs'/name)!=sha:
            raise RuntimeError(f'Training environment changed after GPU smoke: {name}')


def memory_snapshot():
    base=Path('/sys/fs/cgroup')
    try:
        limit=(base/'memory.max').read_text().strip()
        stats=dict(line.split() for line in (base/'memory.stat').read_text().splitlines())
        current=int((base/'memory.current').read_text())
        return {'time':time.time(),'current_bytes':current,
                'working_set_bytes':max(0,current-int(stats.get('inactive_file',0))),
                'limit_bytes':int(limit) if limit!='max' else None}
    except (OSError,ValueError):
        return None


def write_state(stage, **fields):
    RUNTIME.mkdir(parents=True,exist_ok=True)
    old=json.loads(STATE.read_text()) if STATE.exists() else {}
    if stage != 'failed':
        old.pop('error', None)
    if stage != 'teachers_evaluated_baseline_pending':
        old.pop('incomplete', None)
    if stage != 'waiting_for_packing_test':
        old.pop('reason', None)
    old.update(stage=stage,updated_at=datetime.datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
               driver_pid=os.getpid(),**fields)
    tmp=STATE.with_suffix('.tmp'); tmp.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n'); tmp.replace(STATE)


def command(args, stage, role=None):
    write_state(stage,teacher=role,child_pid=None,command=[str(x) for x in args])
    child=subprocess.Popen([str(x) for x in args],cwd=ROOT,start_new_session=True)
    write_state(stage,child_pid=child.pid)
    def stop(signum, frame):
        try: os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError: pass
        write_state('interrupted',child_pid=child.pid)
        raise SystemExit(128+signum)
    previous={s:signal.signal(s,stop) for s in (signal.SIGTERM,signal.SIGINT)}
    try:
        while True:
            try:
                code=child.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                ensure_monitor()
                snapshot=memory_snapshot()
                if snapshot:
                    with (RUNTIME/'memory_usage.jsonl').open('a') as log:
                        log.write(json.dumps(dict(snapshot,stage=stage,teacher=role))+'\n')
                    limit=snapshot['limit_bytes']
                    if limit and snapshot['working_set_bytes']>.94*limit:
                        os.killpg(child.pid,signal.SIGTERM)
                        try: child.wait(timeout=30)
                        except subprocess.TimeoutExpired: pass
                        raise MemoryError('Cgroup working set exceeded 94% of its actual memory limit; '
                                          'sent SIGTERM only to the current experiment child group')
        if code: raise subprocess.CalledProcessError(code,args)
    finally:
        for s,h in previous.items(): signal.signal(s,h)


def run(wait_for_smoke=False):
    RUNTIME.mkdir(parents=True,exist_ok=True)
    # Shared EPC filesystem locks can remain stuck after a killed process. This
    # experiment owns local GPUs, so use a stable per-workspace local lock.
    lock_name='sharedexpert-driver-'+hashlib.sha256(str(RUNTIME.resolve()).encode()).hexdigest()+'.lock'
    with (Path(tempfile.gettempdir())/lock_name).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        (RUNTIME/'experiment.pid').write_text(str(os.getpid())+'\n')
        try:
            write_state('initializing_monitor',teacher=None,child_pid=None)
            ensure_monitor(required=True)
            if wait_for_smoke:
                write_state('waiting_for_smoke',teacher=None,child_pid=None)
                while True:
                    gate=json.loads((RUNTIME/'smoke_gate.json').read_text())
                    if gate.get('status')=='complete': break
                    if gate.get('status')=='failed':
                        raise RuntimeError(f'GPU smoke failed: {gate.get("error")}')
                    pid=gate.get('pid')
                    proc=Path('/proc')/str(pid)/'cmdline'
                    if not proc.exists() or b'smoke_gate.py' not in proc.read_bytes():
                        raise RuntimeError('Preparation process stopped before completing GPU smoke')
                    time.sleep(5)
            if not (HERE/'logs/smoke_validation.json').exists():
                command([TRAIN_PY,HERE/'training/validate_smoke.py',HERE/'runs/perception_smoke'],
                        'validating_smoke')
            write_state('validating_preparation',teacher=None,child_pid=None)
            verify_preparation()
            verify_execution_budget()
            steps=execution_budget()['steps']
            launcher=HERE/'training'/('launch_100.py' if steps==100 else 'launch.py')
            for role in ROLES:
                command([TRAIN_PY,launcher,role,'--gpus',GPUS,'--check'],'preflight',role)
            suite=HERE/'evaluation/suite.jsonl'
            if not suite.exists(): raise RuntimeError('Final frozen evaluation suite is not ready')
            for role in ROLES:
                if role=='perception' and steps==100:
                    if not checkpoint_is_complete(role,steps):
                        raise RuntimeError('Wait for the authorized perception checkpoint-100 handoff; do not reset its progress')
                    # Its live legacy controller was deliberately stopped only
                    # after checkpoint 100. Export directly, preserving the
                    # original 300-step-config run metadata as historical fact.
                    command([TRAIN_PY,launcher,role,'--gpus',GPUS,'--merge-only'],'training',role)
                    continue
                if role=='reasoning':
                    # Perception has finished and released its GPUs. Check the
                    # longer, multi-image R/G inputs once before either teacher.
                    # A validated existing receipt makes supervise CPU-only.
                    memory_check=HERE/'training/multimodal_memory_check.py'
                    command([TRAIN_PY,memory_check,'--supervise'],'preflight',role)
                    command([sys.executable,memory_check,'--verify'],'preflight',role)
                command([TRAIN_PY,launcher,role,'--gpus',GPUS,'--execute'],'training',role)
            for role in ROLES:
                export=HERE/f'runs/{role}/hf/global_step_{steps}'
                command([EVAL_PY,HERE/'evaluation/run_checkpoint.py','--model',export,
                         '--name',role,'--gpu','3','--suite',suite,'--concurrency','4'],'evaluation',role)
            # The completed baseline service can be temporarily suspended to
            # leave CPU memory for training. Its finalizer checks the endpoint
            # even when all predictions are cached, so restore it first.
            if (RUNTIME/'baseline_service_gpu0_suspension.json').exists():
                command([sys.executable,HERE/'manage_baseline_service.py','restore','--gpu','0'],
                        'restoring_baseline_service')
            if (RUNTIME/'baseline_service_suspension.json').exists():
                command([sys.executable,HERE/'manage_baseline_service.py','restore'],
                        'restoring_baseline_service')
            command([EVAL_PY,HERE/'evaluation/baseline_final.py'],'finishing_baseline')
            command([EVAL_PY,HERE/'evaluation/summarize.py','--suite',suite],'summarizing')
            summary=json.loads((HERE/'evaluation/results/summary.json').read_text())
            incomplete={k:v for k,v in summary['status'].items() if v['status']!='complete'}
            if (RUNTIME/'released_9b_servers.json').exists() and not (RUNTIME/'restored_9b_servers.json').exists():
                command([sys.executable,HERE/'manage_training_gpus.py','restore'],'restoring_previous_services')
            if incomplete:
                write_state('teachers_evaluated_baseline_pending',incomplete=incomplete)
            else:
                write_state('complete',summary=str(HERE/'evaluation/results/summary.md'),child_pid=None)
        except Exception as exc:
            write_state('failed',error=f'{type(exc).__name__}: {exc}')
            raise


def status():
    result=json.loads(STATE.read_text()) if STATE.exists() else {'stage':'not_started'}
    common=execution_budget()
    result['training_budget']={'prompt_batch_size':common['batch_size'],'group_size':common['group_size'],
                              'steps_per_teacher':common['steps'],
                              'prompts_per_teacher':common['batch_size']*common['steps'],
                              'responses_per_teacher':common['batch_size']*common['steps']*common['group_size']}
    gate=RUNTIME/'smoke_gate.json'
    if gate.exists(): result['preparation']=json.loads(gate.read_text())
    transition=HERE/'logs/batch30_transition_validation.json'
    if transition.exists() and common['batch_size']==30:
        proof=json.loads(transition.read_text())
        first_step=RUNTIME/'batch30_first_step_validation.json'
        result['batch30_validation']={'configuration_transition_passed':proof.get('passed') is True,
                                     'historical_gpu_smoke_prompt_batch':4,
                                     'formal_gpu_step':json.loads(first_step.read_text()) if first_step.exists() else 'pending'}
    memory_fix=HERE/'logs/unused_topk_transition_validation.json'
    if memory_fix.exists():
        proof=json.loads(memory_fix.read_text())
        result['unused_topk_memory_fix']={
            'validated':proof.get('passed') is True,
            'current_launch_matches':proof.get('new_launch_sha256')==file_sha(HERE/'training/launch.py'),
            'receipt':str(memory_fix),
            'scope':'Independent numerical and warm-Adam memory validation; not a formal training step',
        }
    multimodal_check=HERE/'logs/multimodal_memory_check'
    if (multimodal_check/'cpu_report.json').exists():
        result['multimodal_memory_check']={
            'cpu_prepared':True,
            'gpu_receipt_present':(multimodal_check/'receipt.json').exists(),
            'receipt':str(multimodal_check/'receipt.json'),
            'scope':'Additional R/G memory check at the completed perception boundary; not formal training',
        }
    handoff=RUNTIME/'controller_handoff/status.json'
    if handoff.exists():
        result['controller_handoff']=json.loads(handoff.read_text())
    budget_handoff=RUNTIME/'budget100_handoff/status.json'
    if budget_handoff.exists():
        result['budget100_handoff']=json.loads(budget_handoff.read_text())
    result['data_validation_complete']=(HERE/'data_pipeline/prepared/validation.json').exists()
    result['monitor']=monitor_health()
    driver=result.get('driver_pid')
    if driver:
        try:
            result['driver_running']=b'run_experiment.py' in (Path('/proc')/str(driver)/'cmdline').read_bytes()
        except FileNotFoundError:
            result['driver_running']=False
    result['teachers']={}
    for role in ROLES:
        p=HERE/f'runs/{role}/checkpoints/latest_checkpointed_iteration.txt'
        logged=[int(path.stem) for path in (HERE/f'runs/{role}/rollouts').glob('*.jsonl') if path.stem.isdigit()]
        result['teachers'][role]={'checkpoint_step':int(p.read_text()) if p.exists() else 0,
                                  'last_logged_step':max(logged,default=0),
                                  'target_step':common['steps'],
                                  'exported':(HERE/f'runs/{role}/hf/global_step_{common["steps"]}/sharedexpert_export_complete.json').exists()}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['run','launch','status'])
    parser.add_argument('--wait-for-smoke',action='store_true',help='Wait for the running real GPU preparation gate')
    args=parser.parse_args()
    if args.action=='status': status()
    elif args.action=='run': run(args.wait_for_smoke)
    else:
        RUNTIME.mkdir(parents=True,exist_ok=True)
        logpath=RUNTIME/'experiment.log'
        with logpath.open('ab') as log:
            command_line=[sys.executable,str(Path(__file__).resolve()),'run']
            if args.wait_for_smoke: command_line.append('--wait-for-smoke')
            child=subprocess.Popen(command_line,cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,
                env=dict(os.environ,PYTHONUNBUFFERED='1'))
        print(json.dumps({'pid':child.pid,'log':str(logpath)}))
