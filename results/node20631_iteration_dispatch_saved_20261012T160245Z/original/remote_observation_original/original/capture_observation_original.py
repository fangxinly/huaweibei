import pathlib,datetime,json,sys,subprocess,shutil,hashlib
P=pathlib.Path
root=P('/root/autodl-tmp/ni_node20631_20261012T152116Z/iteration1')
sys.path.insert(0,str(root/'source'))
from group5_release_transport_v1 import write,seal,digest
now=datetime.datetime.now(datetime.timezone.utc)
cap=root.parent/('iteration_observation_'+now.strftime('%Y%m%dT%H%M%SZ'))
cap.mkdir();original=cap/'original';original.mkdir()
rows=[]
def copy(q,rel):
 if not q.exists():return
 assert q.is_file() and q.stat().st_size<2*1024**2
 b=q.read_bytes();target=original/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
 rows.append(dict(name=rel,source=str(q),bytes=len(b),SHA=hashlib.sha256(b).hexdigest(),read_completed_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat()))
for n in ['plan.json','actual_pre_dispatch_original.json','supervisor_wrapper_original.py','supervisor_actual_dispatch.json','supervisor.once','fresh_dispatch_physical.json','queue_status.json','supervisor_stdout.log','supervisor_stderr.log','supervisor_natural_exit.json','supervisor_finished.json']:
 copy(root/n,'root/'+n)
queue=json.loads((original/'root/queue_status.json').read_bytes())
for job in sorted((root/'jobs').iterdir()):
 if not job.is_dir():continue
 for n in ['actual_dispatch.json','out/train.once','out/construction.json','out/health.json','out/progress.json','out/history.json','out/checkpoint_receipt.json','out/result.json','out/failed_original.json','out/guard_journal.json','train_natural_exit.json','CPU_natural_exit.json','train_stdout.log','train_stderr.log','CPU_stdout.log','CPU_stderr.log']:
  copy(job/n,'jobs/'+job.name+'/'+n)
maximum=int(P('/sys/fs/cgroup/memory.max').read_text());current=int(P('/sys/fs/cgroup/memory.current').read_text())
pids=[str(x['pid']) for x in queue['active']]+['2731','2733']
physical=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),GPU=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip(),compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip(),owned_process_fullargv=subprocess.check_output(['ps','-ww','-p',','.join(pids),'-o','pid,ppid,lstart,stat,args'],text=True),cgroup_maximum=maximum,cgroup_current=current,cgroup_available=maximum-current,remote_free=shutil.disk_usage(root).free)
write(original/'actual_physical_observation.json',physical)
write(original/'observation_inventory.json',dict(actual_UTC=now.isoformat(),scope='LIVE_TRAINING_SMALL_METADATA_OBSERVATION_NOT_COMPLETE_RUN',snapshots_are_individually_observed_not_atomic_group=True,source_files=rows,model_arrays_decoded=False,training_restarted=False,new_scientific_once_consumed_by_capture=False))
shutil.copyfile(__file__,original/'capture_observation_original.py')
proof=seal(original,cap/'complete_observation_original.zip');write(cap/'capture_receipt.json',proof)
print(json.dumps(dict(root=str(cap),proof=proof,active=[x['key'] for x in queue['active']],failure_hold=queue['failure_hold'],physical=physical)))
