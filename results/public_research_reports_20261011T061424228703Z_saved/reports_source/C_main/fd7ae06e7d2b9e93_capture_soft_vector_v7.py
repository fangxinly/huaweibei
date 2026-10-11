"""Current Jacobian run plus earlier new research and rotated preservation metadata."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,zipfile,shutil
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--deployment',type=Path,required=True);p.add_argument('--stamp',required=True);a=p.parse_args()
out=a.deployment/('capture_'+a.stamp);out.mkdir(exist_ok=False)
q=lambda args:subprocess.check_output(args,text=True).strip();sha=lambda b:hashlib.sha256(b).hexdigest()
launch=json.loads((a.deployment/'launch.json').read_text());proc=Path('/proc')/str(launch['pid']);argv=None;status=None
if (proc/'cmdline').exists():argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1];status=(proc/'status').read_text()
inv={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'capture_source_sha256':sha(Path(__file__).read_bytes()),'gpu':q(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader,nounits']),'compute':q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),'launch':launch,'actual_process_argv':argv,'actual_process_status':status,'data_disk_free':shutil.disk_usage('/data').free,'root':str(a.root),'deployment':str(a.deployment),'selection_present':(a.deployment/'run/selection.json').exists()}
members={};large={};final_path=out/'snapshot.zip';zpath=out/'snapshot.pending'
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
 def add(path,name):
  b=path.read_bytes();z.writestr(name,b);members[name]={'bytes':len(b),'sha256':sha(b)}
 z.writestr('inventory.json',json.dumps(inv,indent=2));members['inventory.json']={'bytes':len(json.dumps(inv,indent=2).encode()),'sha256':sha(json.dumps(inv,indent=2).encode())}
 for n in ['formal_plan_v1.json','formal_plan_v2.json','train_soft_vector_v1.py','train_soft_vector_v2.py','soft_vector_runtime_v1.py','soft_vector_runtime_v2.py','task_gradient_vector_candidate_v3.py','soft_vector_jacobian_candidate_v1.py','capture_soft_vector_v7.py','verify_soft_preservation_cpu_v1.py','assets_verified.json','teacher_cache_v1/collection.json','teacher_cache_v1/train_gradient_rms.npy','label_free_jacobian_v1/jacobian_receipt.json','label_free_jacobian_v1/train_jacobian.npz','label_free_jacobian_v1/dev_jacobian.npz']:
  add(a.root/n,'source/'+n)
 for path in sorted(a.deployment.glob('*.json')):add(path,path.name)
 def tree(folder,prefix):
  if not folder.exists():return
  for path in sorted(folder.rglob('*')):
   if not path.is_file() or path.suffix not in ['.json','.npz','.npy','.pt']:continue
   name=prefix+'/'+path.relative_to(folder).as_posix()
   if path.stat().st_size>8*1024**2:large[name]={'path':str(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'sha256':sha(path.read_bytes())}
   else:add(path,name)
 tree(a.deployment/'run','run');tree(a.deployment/'diagnostics_v2','diagnostics');tree(a.deployment/'full_checkpoints','full_checkpoints')
 previous=Path('/data/coding/soft_vector_v1_deployment_20261005T1255Z');tree(previous/'run','previous_run');tree(previous/'diagnostics_v1','previous_diagnostics');tree(previous/'full_checkpoints','previous_full_checkpoints')
 tree(Path('/data/coding/soft_vector_preservation_20261005T1310Z'),'previous_preservation')
 tree(Path('/data/coding/jacobian_preservation_20261005T1406Z'),'current_preservation')
 finite_root=Path('/data/coding/finite_task_risk_preflight_20261005T1428Z')
 for item in sorted(finite_root.glob('*.py')):add(item,'finite_source/'+item.name)
 for item in sorted(finite_root.glob('*.json')):add(item,'finite_source/'+item.name)
 for filename in ['preflight.log','head_holdout.log','head_holdout_v2.log','full_inference.log']:
  if (finite_root/filename).exists():add(finite_root/filename,'finite_logs/'+filename)
 tree(finite_root/'checks','finite_preflight')
 formal=Path('/data/coding/finite_task_risk_v1_deployment_20261005T1450Z')
 for path in sorted(formal.glob('*')):
  if path.is_file() and path.suffix in ['.py','.json','.npy']:add(path,'finite_formal_source/'+path.name)
 tree(formal/'run','finite_run')
 tree(formal/'full_checkpoints','finite_full_checkpoints')
 tree(Path('/data/coding/finite_preservation_20261005T1515Z'),'finite_preservation')
 failed=Path('/data/coding/failed_finite_vector_diagnostics_20261005T1503Z')
 tree(failed,'finite_failed_diagnostics')
 if (failed/'anomaly_trace.txt').exists():add(failed/'anomaly_trace.txt','finite_failed_diagnostics/anomaly_trace.txt')
 for name in ['training.log','wrapper.log','assembly_a.log','assembly_b.log']:
  if (formal/name).exists():add(formal/name,'finite_formal_logs/'+name)
 if (finite_root/'failed_vector_diagnostic.log').exists():add(finite_root/'failed_vector_diagnostic.log','finite_failed_diagnostics/diagnostic.log')

 fin_launch=json.loads((formal/'launch.json').read_text())
 fin_proc=Path('/proc')/str(fin_launch['pid'])
 fin_inv={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu':inv['gpu'],'compute':inv['compute'],'launch':fin_launch,'actual_process_argv':None,'actual_process_status':None,'data_disk_free':shutil.disk_usage('/data').free,'root':str(formal)}
 if (fin_proc/'cmdline').exists():
  fin_inv['actual_process_argv']=(fin_proc/'cmdline').read_bytes().decode().split('\0')[:-1]
  fin_inv['actual_process_status']=(fin_proc/'status').read_text()
 if (formal/'exit.json').exists():fin_inv['exit']=json.loads((formal/'exit.json').read_text())
 encoded=json.dumps(fin_inv,indent=2).encode();z.writestr('finite_inventory.json',encoded);members['finite_inventory.json']={'bytes':len(encoded),'sha256':sha(encoded)}


 z.writestr('large_file_manifest.json',json.dumps(large,indent=2));encoded=json.dumps(large,indent=2).encode();members['large_file_manifest.json']={'bytes':len(encoded),'sha256':sha(encoded)}
 z.writestr('member_manifest.json',json.dumps(members,indent=2))
receipt={'utc':inv['utc'],'sha256':sha(zpath.read_bytes()),'bytes':zpath.stat().st_size,'members':len(members),'large_file_count':len(large),'scope':'Large fullweights fresh SHA referenced, not included in ZIP. Already permanent copies verified separately; snapshot is not itself a new fullweight download.'}
zpath.replace(final_path)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('NEW_RESEARCH_CAPTURE_COMPLETE',a.stamp,len(members),len(large),flush=True)
