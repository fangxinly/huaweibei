"""Read-only active training observation and immutable small prefix preservation."""
from pathlib import Path
import sys,json,hashlib,subprocess,zipfile,datetime
root=Path(sys.argv[1]);dest=Path(sys.argv[2]);bundle=Path(sys.argv[3]);expected=sys.argv[4]
def sha(b):return hashlib.sha256(b).hexdigest()
plan_bytes=(bundle/'paired_fulltrain_execution_plan.json').read_bytes()
assert sha(plan_bytes)==expected
plan=json.loads(plan_bytes)
assert root.parent==dest.parent==Path('/data/coding') and root.name.startswith('paired_fulltrain_train100_') and dest.name.startswith('paired_fulltrain_active_observation_') and not dest.exists()
dest.mkdir()
sources={n:sha((bundle/n).read_bytes()) for n in plan['source_sha256']}
assert sources==plan['source_sha256']
launch=json.loads((root/'actual_child_launch.json').read_text());pid=launch['pid']
argv=Path('/proc')/str(pid)/'cmdline'
observed_argv=argv.read_bytes().decode().strip('\0').split('\0') if argv.exists() else None
obs={'status':'ACTUAL_ACTIVE_PREFIX_OBSERVATION_NOT_COMPLETE_CAPTURE','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'plan_sha256':expected,'original_child_launch':launch,'actual_child_proc_fullargv':observed_argv,'original_all_source_SHA':sources,'natural_exit_present_at_observation':(root/'natural_exit.json').exists(),'not_complete_training_or_final_weights':True,'no_model_labels_predictions_decoded':True,'NVML_compute_PID_not_assumed_container_PID':True}
members=[]
with zipfile.ZipFile(dest/'active_prefix_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for name in ['actual_child_launch.json','wrapper_actual_start.json','out/actual_training_start.json','out/actual_direct_preflight.json','out/progress.json','out/history.json','out/actual_TRAIN_steps.jsonl','out/cumulative_budget.jsonl','child.stdout.log','child.stderr.log','natural_exit.json','out/actual_stage_receipt.json']:
  p=root/name
  if not p.is_file():continue
  b=p.read_bytes()
  if name.endswith('.jsonl'):
   # Preserve only complete append records, never a torn final write.
   b=b[:b.rfind(b'\n')+1]
  members.append({'member':name,'bytes':len(b),'sha256':sha(b),'original_path':str(p)})
  z.writestr(name,b)
  if name=='out/actual_TRAIN_steps.jsonl':obs['completed_updates']=len(b.splitlines());obs['last_complete_update']=json.loads(b.splitlines()[-1])
  if name=='out/cumulative_budget.jsonl':obs['last_original_cumulative_budget']=json.loads(b.splitlines()[-1])
  if name=='out/progress.json':obs['original_selection_progress_only']=json.loads(b)
  if name=='child.stderr.log':obs['original_stderr_bytes']=len(b);obs['Traceback_in_original_stderr']=b'Traceback' in b
 for cmd in [['nvidia-smi','--query-gpu=uuid,memory.used,memory.total','--format=csv,noheader'],['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],['ps','-eo','pid,ppid,args'],['df','-B1','/data','/']]:
  r=subprocess.run(cmd,capture_output=True,text=True);obs[' '.join(cmd)]={'exit_code':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
 obs['members']=members
 z.writestr('actual_active_observation.json',json.dumps(obs,ensure_ascii=False,indent=2)+'\n')
with zipfile.ZipFile(dest/'active_prefix_snapshot.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
receipt={'status':'ACTUAL_ACTIVE_PREFIX_BYTES_FROZEN_NOT_COMPLETE_CAPTURE','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(root),'dest':str(dest),'snapshot_sha256':sha((dest/'active_prefix_snapshot.zip').read_bytes()),'snapshot_bytes':(dest/'active_prefix_snapshot.zip').stat().st_size,'members':len(members)+1,'completed_updates':obs.get('completed_updates'),'epoch_completed':obs.get('original_selection_progress_only',{}).get('epoch'),'original_child_proc_argv_matches_launch':observed_argv==launch['full_argv'],'Traceback_in_original_stderr':obs.get('Traceback_in_original_stderr'),'not_final_weights_or_completed_capture':True,'no_stop_restart_or_label_metric_read':True}
(dest/'actual_active_prefix_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
