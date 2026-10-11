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
  b=path.read_bytes();meta={'bytes':len(b),'sha256':sha(b)}
  if name in members:
   assert members[name]==meta;return
  z.writestr(name,b);members[name]=meta
 z.writestr('inventory.json',json.dumps(inv,indent=2));members['inventory.json']={'bytes':len(json.dumps(inv,indent=2).encode()),'sha256':sha(json.dumps(inv,indent=2).encode())}
 for n in ['formal_plan_v1.json','formal_plan_v2.json','train_soft_vector_v1.py','train_soft_vector_v2.py','soft_vector_runtime_v1.py','soft_vector_runtime_v2.py','task_gradient_vector_candidate_v3.py','soft_vector_jacobian_candidate_v1.py','capture_soft_vector_v16.py','verify_soft_preservation_cpu_v1.py','assets_verified.json','teacher_cache_v1/collection.json','teacher_cache_v1/train_gradient_rms.npy','label_free_jacobian_v1/jacobian_receipt.json','label_free_jacobian_v1/train_jacobian.npz','label_free_jacobian_v1/dev_jacobian.npz']:
  add(a.root/n,'source/'+n)
 for path in sorted(a.deployment.glob('*.json')):add(path,path.name)
 def tree(folder,prefix):
  if not folder.exists():return
  for path in sorted(folder.rglob('*')):
   if not path.is_file() or path.suffix not in ['.json','.npz','.npy','.pt','.py']:continue
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
 singleton=Path('/data/coding/finite_single_token_precheck_20261005T1525Z')
 tree(singleton,'finite_single_token_precheck')
 for f in sorted(singleton.glob('*.py')):add(f,'finite_single_token_precheck/'+f.name)
 if (singleton/'precheck.log').exists():add(singleton/'precheck.log','finite_single_token_precheck/precheck.log')
 for n in ['warmup.log','full_raw.log']:
  if (singleton/n).exists():add(singleton/n,'finite_single_token_precheck/'+n)

 failed=Path('/data/coding/failed_finite_vector_diagnostics_20261005T1503Z')
 tree(failed,'finite_failed_diagnostics')
 if (failed/'diagnose_finite_zero_norm_v2.py').exists():add(failed/'diagnose_finite_zero_norm_v2.py','finite_failed_diagnostics/diagnose_finite_zero_norm_v2.py')
 if (failed/'zero_norm_probe_v2.log').exists():add(failed/'zero_norm_probe_v2.log','finite_failed_diagnostics/zero_norm_probe_v2.log')
 if (failed/'zero_norm_probe.log').exists():add(failed/'zero_norm_probe.log','finite_failed_diagnostics/zero_norm_probe.log')
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


 for version in ['v1','v2','v3']:
  tree(formal/('diagnostics_'+version),'finite_diagnostics_'+version)
  log=formal/('diagnostics_'+version+'.log')
  if log.exists():add(log,'finite_diagnostics_'+version+'/execution.log')
 secondorder=Path('/data/coding/finite_single_token_secondorder_20261005T1552Z')
 tree(secondorder,'finite_single_token_secondorder')
 if (singleton/'secondorder.log').exists():add(singleton/'secondorder.log','finite_single_token_precheck/secondorder.log')
 c2=Path('/data/coding/finite_task_risk_c2_deployment_20261005T1600Z')
 tree(c2,'finite_c2')
 if c2.exists():
  for n in ['training.log','wrapper.log']:
   if (c2/n).exists():add(c2/n,'finite_c2/'+n)
  cl=json.loads((c2/'launch.json').read_text());cp=Path('/proc')/str(cl['pid'])
  ci=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=inv['gpu'],compute=q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),launch=cl,actual_process_argv=None,actual_process_status=None,data_disk_free=shutil.disk_usage('/data').free,root=str(c2))
  if (cp/'cmdline').exists():ci['actual_process_argv']=(cp/'cmdline').read_bytes().decode().split('\0')[:-1];ci['actual_process_status']=(cp/'status').read_text()
  if (c2/'exit.json').exists():ci['exit']=json.loads((c2/'exit.json').read_text())
  encoded=json.dumps(ci,indent=2).encode();z.writestr('finite_c2_inventory.json',encoded);members['finite_c2_inventory.json']={'bytes':len(encoded),'sha256':sha(encoded)}

 assembly=Path('/data/coding/finite_c2_assembly_20261005T1615Z')
 tree(assembly,'finite_c2_assembly')
 for folder,prefix in [(assembly,'finite_c2_assembly'),(c2,'finite_c2')]:
  for n in ['assembly.log','diagnostics_v4.log']:
   if (folder/n).exists():add(folder/n,prefix+'/'+n)
 tree(Path('/data/coding/finite_c2_preservation_20261005T1620Z'),'finite_c2_preservation')

 tree(Path('/data/coding/train_group_mapping_20261005T1628Z'),'train_group_mapping')
 tree(Path('/data/coding/train_group_mapping_source_20261005T1628Z'),'train_group_mapping_source')

 oracle=Path('/data/coding/train_oracle_diagnostic_20261006T014605Z')
 tree(oracle,'train_oracle')
 if oracle.exists():
  for path in sorted(oracle.rglob('*')):
   if path.is_file() and path.suffix in ['.log','.txt']:
    add(path,'train_oracle/'+path.relative_to(oracle).as_posix())
 if oracle.exists():
  for filename in ['execution.log','wrapper.log']:
   if (oracle/filename).is_file():add(oracle/filename,'train_oracle/'+filename)
  matches=[]
  for candidate in sorted(Path('/proc').iterdir()):
   if not candidate.name.isdigit():continue
   try:
    raw=(candidate/'cmdline').read_bytes().decode().split('\0')[:-1]
   except (OSError,UnicodeError):continue
   if any(Path(arg).name=='diagnose_train_oracle_v2.py' for arg in raw) and str(oracle/'train_oracle_plan_v2.json') in raw:
    matches.append({'pid':int(candidate.name),'argv':raw})
  oi=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(oracle),gpu=inv['gpu'],
          compute=q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),
          matching_processes=matches,data_disk_free=shutil.disk_usage('/data').free,
          receipt_present=(oracle/'out/receipt.json').is_file(),progress_present=(oracle/'out/progress.json').is_file(),
          scope='Capture evidence, not proof of diagnostic completion; original receipt and arrays need independent audit.')
  encoded=json.dumps(oi,indent=2).encode();z.writestr('train_oracle_inventory.json',encoded)
  members['train_oracle_inventory.json']={'bytes':len(encoded),'sha256':sha(encoded)}
 z.writestr('large_file_manifest.json',json.dumps(large,indent=2));encoded=json.dumps(large,indent=2).encode();members['large_file_manifest.json']={'bytes':len(encoded),'sha256':sha(encoded)}
 z.writestr('member_manifest.json',json.dumps(members,indent=2))
receipt={'utc':inv['utc'],'sha256':sha(zpath.read_bytes()),'bytes':zpath.stat().st_size,'members':len(members),'large_file_count':len(large),'scope':'Large fullweights fresh SHA referenced, not included in ZIP. Already permanent copies verified separately; snapshot is not itself a new fullweight download.'}
zpath.replace(final_path)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print('NEW_RESEARCH_CAPTURE_COMPLETE',a.stamp,len(members),len(large),flush=True)
