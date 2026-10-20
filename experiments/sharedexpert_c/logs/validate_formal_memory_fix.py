#!/usr/bin/env python3
"""Read only current formal artifacts and publish independently scoped evidence."""
import collections, datetime, hashlib, json, math, re, sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from omegaconf import OmegaConf
B=Path('/home/ml-user/workdir/CVPR2027/experiments/sharedexpert_c')
R=B/'runs/perception'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
 return h.hexdigest()
def write(p,v):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def main(target):
 protocol=json.loads((R/'protocol.json').read_text());cfg=OmegaConf.load(R/'resolved_config.yaml')
 launch=json.loads((B/'runtime/unused_topk_fix_launch.json').read_text())
 assert protocol['training_code_sha256']['launch.py']==sha(B/'training/launch.py')==launch['launch_sha256']
 assert protocol['protocol']['batch_size']==30 and protocol['protocol']['group_size']==8
 assert cfg.data.train_batch_size==30 and cfg.actor_rollout_ref.actor.ppo_mini_batch_size==30
 assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode=='vanilla'
 assert not cfg.actor_rollout_ref.actor.self_distillation.full_logit_distillation
 assert cfg.actor_rollout_ref.actor.self_distillation.distillation_topk is None
 assert not cfg.actor_rollout_ref.actor.use_kl_loss and not cfg.algorithm.use_kl_in_reward
 assert cfg.actor_rollout_ref.actor.entropy_coeff==0 and not cfg.actor_rollout_ref.actor.calculate_entropy
 events={};event_paths=[]
 for p in sorted((R/'tensorboard').rglob('events.out.tfevents.*')):
  ea=EventAccumulator(str(p),size_guidance={'scalars':0});ea.Reload();event_paths.append(str(p))
  for tag in ea.Tags()['scalars']:
   for e in ea.Scalars(tag):
    if e.step in range(1,target+1):events.setdefault(e.step,{})[tag]={'value':e.value,'wall_time':e.wall_time}
 steps=[]
 for step in range(1,target+1):
  roll=R/f'rollouts/{step}.jsonl';rows=[json.loads(x) for x in roll.read_text().splitlines()]
  assert len(rows)==240
  groups=collections.defaultdict(list)
  for row in rows:groups[row['group_uid']].append(row)
  assert len(groups)==30 and all(len(g)==8 for g in groups.values())
  assert max(x['prompt_tokens'] for x in rows)<=2048 and max(x['response_tokens'] for x in rows)<=8192
  scalars=events[step];assert scalars and all(math.isfinite(x['value']) for x in scalars.values())
  required=['actor/pg_loss','actor/grad_norm','actor/lr','critic/score/mean','rollout_corr/rollout_is_mean','rollout_corr/rollout_is_min','rollout_corr/rollout_is_max','training/global_step']
  assert all(k in scalars for k in required)
  assert scalars['actor/grad_norm']['value']>0 and scalars['training/global_step']['value']==step
  if step==1:assert scalars['actor/lr']['value']==0
  if step==2:assert scalars['actor/lr']['value']>0
  steps.append({'step':step,'rollout_sha256':sha(roll),'responses':len(rows),'prompt_groups':len(groups),'responses_per_group':8,
   'source_counts':dict(collections.Counter(x['audit_source'] for x in rows)),
   'max_prompt_tokens':max(x['prompt_tokens'] for x in rows),'max_response_tokens':max(x['response_tokens'] for x in rows),
   'truncated_responses':sum(bool(x['truncated']) for x in rows),'truncated_scores_zero':all(x['score']==0 for x in rows if x['truncated']),
   'nonuniform_reward_groups':sum(len({x['score'] for x in g})>1 for g in groups.values()),
   'unique_outputs_per_group':dict(collections.Counter(len({x['output'] for x in g}) for g in groups.values())),
   'scalar_count':len(scalars),'all_scalars_finite':True,'selected_metrics':{k:scalars[k]['value'] for k in required},
   'all_scalars':scalars,'completed_wall_time':max(x['wall_time'] for x in scalars.values())})
 start=datetime.datetime.fromisoformat(launch['launch_window_start']).timestamp();end=steps[-1]['completed_wall_time']
 resource={'sampling_interval_seconds':5,'scope':'This driver attempt, initialization and completed formal steps only',
  'window_start_epoch':start,'window_end_epoch':end,'driver_pid':launch['driver_pid'],'driver_start_ticks':launch['driver_start_ticks'],
  'sample_count':0,'peak_working_set_bytes':0,'peak_current_bytes':0,'peak_anon_bytes':0,'gpu_peak_used_mib':{},'oom_baseline':{'oom':19,'oom_kill':12}}
 latest=None
 with (B/'monitoring/samples.jsonl').open() as f:
  for line in f:
   m=re.search(r'"time":\s*"([^"]+)"',line[:100])
   if not m:continue
   stamp=datetime.datetime.fromisoformat(m[1]).timestamp()
   if stamp<start or stamp>end:continue
   sample=json.loads(line);driver=sample.get('driver') or {}
   if driver.get('pid')!=launch['driver_pid'] or driver.get('start_ticks')!=launch['driver_start_ticks'] or sample.get('test_mode'):continue
   resource['sample_count']+=1;c=sample['cgroup'];latest=c['events']
   for key in ('working_set_bytes','current_bytes','anon_bytes'):resource['peak_'+key]=max(resource['peak_'+key],c.get(key,0))
   for g in sample['gpus']:
    if g['index'] in (3,4,5,6):resource['gpu_peak_used_mib'][str(g['index'])]=max(resource['gpu_peak_used_mib'].get(str(g['index']),0),g['memory_used_mib'])
 assert resource['sample_count']>10 and latest is not None
 resource['events_at_end']=latest;resource['oom_delta']={k:latest[k]-v for k,v in resource['oom_baseline'].items()}
 assert all(x==0 for x in resource['oom_delta'].values())
 # CUDA OOM does not increment host cgroup counters. Check the current run's
 # own log snapshot independently, and preserve all historical monitor alerts.
 train_log=R/'training.log';train_bytes=train_log.read_bytes()
 cuda_pattern=r'CUDA out of memory|torch\.cuda\.OutOfMemoryError|CUDA error:\s*out of memory|cudaErrorMemoryAllocation'
 cuda_matches=[{'line_number':i,'text':line} for i,line in enumerate(train_bytes.decode(errors='replace').splitlines(),1)
  if re.search(cuda_pattern,line,re.I)]
 assert not cuda_matches, cuda_matches
 alert_path=B/'monitoring/alerts.jsonl';alert_bytes=alert_path.read_bytes()
 all_alerts=[json.loads(line) for line in alert_bytes.decode().splitlines() if line.strip()]
 alerts_in_window=[]
 for alert in all_alerts:
  if alert.get('test_mode'):continue
  stamp=datetime.datetime.fromisoformat(alert['observed_at']).timestamp()
  if start<=stamp<=end:alerts_in_window.append(alert)
 oom_kinds={'cuda_oom_log','host_or_ray_oom_log','host_memory_error_log','cgroup_oom_counter_increment','host_oom','cuda_oom'}
 oom_alerts=[a for a in alerts_in_window if a['kind'] in oom_kinds]
 assert not oom_alerts, oom_alerts
 oom_evidence={'host_cgroup_oom_delta':resource['oom_delta'],
  'current_run_training_log':{'path':str(train_log),'snapshot_size_bytes':len(train_bytes),
   'snapshot_sha256':hashlib.sha256(train_bytes).hexdigest(),'cuda_oom_regex':cuda_pattern,
   'cuda_oom_matches':cuda_matches,'scope':'Full current-run log snapshot through validation time; may extend beyond completed-step window.'},
  'monitor_alerts':{'path':str(alert_path),'snapshot_sha256':hashlib.sha256(alert_bytes).hexdigest(),
   'snapshot_size_bytes':len(alert_bytes),'historical_alerts_preserved':len(all_alerts),
   'observed_window_start_epoch':start,'observed_window_end_epoch':end,
   'alert_kinds_checked':sorted(oom_kinds),'all_non_test_alerts_in_window':alerts_in_window,
   'oom_alerts_in_window':oom_alerts,'timestamp_semantics':'Monitor observed_at, not reconstructed fault time.'},
  'no_observed_host_or_cuda_oom':True}
 result={'passed':True,'validated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'teacher':'perception','completed_steps':target,
  'current_run_directory':str(R),'protocol_sha256':sha(R/'protocol.json'),'launch_sha256':sha(B/'training/launch.py'),
  'resolved_config_sha256':sha(R/'resolved_config.yaml'),'reward_sha256':protocol['reward_sha256'],
  'repository_diff_sha256':protocol['repository_diff_sha256'],'training_code_sha256':protocol['training_code_sha256'],
  'training_environment_sha256':protocol['training_environment_sha256'],
  'prompt_batch_size':30,'group_size':8,'target_steps':300,'unused_topk_disabled':True,'kl_and_entropy_disabled':True,
  'steps':steps,'resources':resource,'oom_evidence':oom_evidence,'tensorboard_sources':event_paths,'audit_script_sha256':sha(__file__),
  'limitations':['Finite scalar, completed update and rollout evidence are scoped to the listed steps, not all 300.',
   'Step 1 has warmup learning rate zero; it establishes computation and Adam initialization, not a nonzero parameter change.',
   'Physical resource peaks use independent 5-second NVML/cgroup samples and may miss shorter peaks.',
   'Framework perf memory scalars are not used as physical GPU/container peaks.']}
 if target==1:
  result['numerical_parameter_update_claim']=False;path=B/'runtime/batch30_first_step_validation.json'
 else:
  result['second_step_positive_learning_rate_optimizer_update']=True;path=B/'runtime/unused_topk_formal_validation.json'
 write(path,result)
 print(json.dumps({'passed':True,'path':str(path),'steps':target,'resource':resource,'selected_metrics':[x['selected_metrics'] for x in steps]}))
if __name__=='__main__':main(int(sys.argv[1]))
