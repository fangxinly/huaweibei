"""Bounded three-process official-seed queue. Preserve originals, never retry failures."""
import argparse,datetime as dt,json,os,pathlib,shutil,subprocess,sys,time
P=pathlib.Path
from group5_release_transport_v1 import digest,write,seal,plan as range_plan
def now():return dt.datetime.now(dt.timezone.utc)
def physical(root):
 maximum=int(P('/sys/fs/cgroup/memory.max').read_text());current=int(P('/sys/fs/cgroup/memory.current').read_text())
 gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
 return dict(actual_UTC=now().isoformat(),GPU=gpu,cgroup_maximum=maximum,cgroup_current=current,available_container_RAM=maximum-current,remote_free=shutil.disk_usage(root).free,compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip())
def main():
 a=argparse.ArgumentParser();a.add_argument('--root',type=P,required=True);a.add_argument('--plan-sha',required=True);a=a.parse_args()
 root=a.root;p=json.loads((root/'plan.json').read_bytes());assert digest(root/'plan.json')==a.plan_sha and p['execution_enabled']
 assert sys.executable==p['python'] and not (root/'supervisor.once').exists()
 for n,h in p['source_SHA'].items():assert digest(root/'source'/n)==h
 runtime=P(p['runtime_receipt']);assert runtime.exists() and json.loads(runtime.read_bytes())['CUDA_available']
 first=physical(root);assert first['GPU'].split(',')[0]==p['GPU_UUID'] and not first['compute']
 assert first['available_container_RAM']>24*1024**3 and first['remote_free']>20*1024**3
 with (root/'supervisor.once').open('x') as f:f.write(now().isoformat())
 (root/'jobs').mkdir();write(root/'fresh_dispatch_physical.json',first)
 queue=[(s,m) for s in p['seeds'] for m in p['methods']];active={};finished=[];launches=[];pending_sealed=[];failure_hold=False
 def report():write(root/'queue_status.json',dict(actual_UTC=now().isoformat(),protocol=p['protocol'],planned_jobs=len(queue)+len(active)+len(finished),pending=[dict(seed=s,method=m) for s,m in queue],active=[dict(key=k,pid=v['child'].pid,seed=v['seed'],method=v['method']) for k,v in active.items()],finished=finished,pending_sealed=pending_sealed,physical=physical(root),failure_hold=failure_hold,TEST_cannot_change_queue=True))
 while queue or active:
  for key,v in list(active.items()):
   code=v['child'].poll()
   if code is None:continue
   v['stdout'].close();v['stderr'].close();job=v['root'];write(job/'train_natural_exit.json',dict(actual_UTC=now().isoformat(),pid=v['child'].pid,natural_exit=code,argv=v['argv']))
   done=dict(key=key,seed=v['seed'],method=v['method'],train_exit=code)
   if code==0:
    cmd=[sys.executable,'-B',str(root/'source/node20631_CPU_audit_v1.py'),'--job',str(job)]
    with (job/'CPU_stdout.log').open('wb') as o,(job/'CPU_stderr.log').open('wb') as e:
     audit=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=o,stderr=e,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''));rc=audit.wait()
    write(job/'CPU_natural_exit.json',dict(actual_UTC=now().isoformat(),pid=audit.pid,natural_exit=rc,argv=cmd));done['CPU_exit']=rc
   # The original closed logs and full states are sealed even on failure.
   original_source=job/'original_source';shutil.copytree(root/'source',original_source);shutil.copyfile(root/'plan.json',job/'frozen_plan.json')
   proof=seal(job,job/'complete_original.zip');write(root/'jobs'/(key+'_capture_receipt.json'),proof)
   manifest=range_plan(P(proof['archive']),'official5seed20631-'+key+'-'+proof['archive_SHA'][:12]);manifest.update(repository='fangxinly/huaweibei',tag='autonomous-flow-health-20261008',publication_authorized=True,method=v['method'],seed=v['seed'],plan_SHA=a.plan_sha)
   write(root/'jobs'/(key+'_publication_manifest.json'),manifest);pending_sealed.append(key)
   done['archive_SHA']=proof['archive_SHA'];done['archive_bytes']=proof['bytes'];finished.append(done);del active[key]
   if code!=0 or done.get('CPU_exit',0)!=0:failure_hold=True
   report()
  # Admission is based only on resource/timing status, never VAL/TEST scores.
  cap=p['maximum_concurrent_jobs'];fresh=physical(root)
  if active:
   for v in active.values():
    health=v['root']/'out/health.json'
    progress=v['root']/'out/progress.json'
    if not health.exists() or json.loads(health.read_bytes())['step']<40 or not progress.exists():cap=min(cap,len(active))
  left=(dt.datetime.fromisoformat(p['lease_end_UTC'])-now()).total_seconds()
  prior_peak=max([json.loads((v['root']/'out/health.json').read_bytes())['peak_reserved']/1024**2 for v in active.values() if (v['root']/'out/health.json').exists()]+[0])
  needed_GPU=max(p['GPU_free_for_new_job_MiB'],int(prior_peak*1.20))
  if queue and not failure_hold and len(active)<cap and left>p['per_job_budget_seconds']+7200 and fresh['available_container_RAM']>12*1024**3 and fresh['remote_free']>p['new_job_minimum_free_bytes'] and int(fresh['GPU'].split(',')[1])>needed_GPU:
   seed,method=queue.pop(0);key=method+'_seed'+str(seed);job=root/'jobs'/key;job.mkdir()
   cmd=[sys.executable,'-B',str(root/'source/node20631_train_iteration_v1.py'),'--plan',str(root/'plan.json'),'--plan-sha',a.plan_sha,'--bundle',str(root/'source'),'--assets',p['assets_root'],'--out',str(job/'out'),'--method',method,'--seed',str(seed)]
   o=(job/'train_stdout.log').open('wb');e=(job/'train_stderr.log').open('wb');child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=o,stderr=e,env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1'))
   write(job/'actual_dispatch.json',dict(actual_UTC=now().isoformat(),pid=child.pid,argv=cmd,source_SHA=p['source_SHA'],plan_SHA=a.plan_sha,physical=fresh))
   active[key]=dict(child=child,stdout=o,stderr=e,root=job,seed=seed,method=method,argv=cmd);launches.append(key);report()
  if not active and queue and (failure_hold or left<=p['per_job_budget_seconds']+7200):
   report();write(root/'queue_stopped_for_saving.json',dict(actual_UTC=now().isoformat(),pending_not_dispatched=len(queue),saving_reserve_seconds=7200));break
  report();time.sleep(20)
 write(root/'supervisor_finished.json',dict(actual_UTC=now().isoformat(),finished=finished,pending=[dict(seed=s,method=m) for s,m in queue],pending_sealed=pending_sealed,all_results_preserved_remotely=True,GitHub_full_publication_pending=bool(pending_sealed)))
if __name__=='__main__':main()

