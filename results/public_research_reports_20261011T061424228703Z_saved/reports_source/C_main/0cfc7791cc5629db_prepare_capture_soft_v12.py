from pathlib import Path
import hashlib,json
w=Path(__file__).parent;s=(w/'capture_soft_vector_v11.py').read_text(encoding='utf-8').replace('capture_soft_vector_v11.py','capture_soft_vector_v12.py')
needle=" z.writestr('large_file_manifest.json',json.dumps(large,indent=2));"
assert needle in s
extra=""" for version in ['v1','v2','v3']:
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
  if (cp/'cmdline').exists():ci['actual_process_argv']=(cp/'cmdline').read_bytes().decode().split('\\0')[:-1];ci['actual_process_status']=(cp/'status').read_text()
  if (c2/'exit.json').exists():ci['exit']=json.loads((c2/'exit.json').read_text())
  encoded=json.dumps(ci,indent=2).encode();z.writestr('finite_c2_inventory.json',encoded);members['finite_c2_inventory.json']={'bytes':len(encoded),'sha256':sha(encoded)}

"""
s=s.replace(needle,extra+needle)
p=w/'capture_soft_vector_v12.py';assert not p.exists();p.write_text(s,encoding='utf-8')
print(json.dumps({'capture12_sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
