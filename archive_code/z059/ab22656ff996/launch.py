import argparse,datetime,hashlib,json,subprocess,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--node',required=True);a=p.parse_args()
B=Path('/data/coding/selective_flow/strong_baselines');P=B.parent/'provision_20261004T0647Z';V=B/'matched_repeat_deployment_20261004T0650Z'
assert json.loads((P/'destination_verification.json').read_text())['status']=='PORTABLE_INPUTS_ALL_FILES_SHA_AND_RUNTIME_VERIFIED'
assert hashlib.sha256((P/'deployment.zip').read_bytes()).hexdigest()=='8937302b4bcd01bfdec805e1028bb9c718068720068a453c563805da8489645f'
assert not V.exists()
with zipfile.ZipFile(P/'deployment.zip') as z:
 assert all('..' not in Path(n).parts and not n.startswith('/') for n in z.namelist())
 z.extractall(V)
manifest=json.loads((V/'deployment_manifest.json').read_text())
assert all(hashlib.sha256((V/n).read_bytes()).hexdigest()==h for n,h in manifest.items())
plan=json.loads((V/'plan.json').read_text());node=plan['nodes'][a.node]
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==node['gpu_uuid']
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not (B/'own_flow_v8_identifiability/matched_repeat_20261004T0650Z'/a.node).exists()
cmd=[str(B/'.venv/bin/python'),'-u',str(V/'repeat_matched_worker.py'),'--node',a.node]
with (V/(a.node+'_worker.log')).open('x') as log:
 child=subprocess.Popen(cmd,cwd=B,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
receipt={'status':'PROCESS_STARTED_NOT_YET_GPU_TRAINING_VERIFIED','pid':child.pid,'node':a.node,'command':cmd,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deployment_manifest':manifest}
with (V/(a.node+'_launch.json')).open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt),flush=True)
