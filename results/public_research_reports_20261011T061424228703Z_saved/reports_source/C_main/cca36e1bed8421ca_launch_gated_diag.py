from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['none','state','task'],required=True);a.add_argument('--uuid',required=True);c=a.parse_args()
root=Path('/data/coding/selective_flow/gated_diagnostics_20261005T0330Z');root.mkdir(exist_ok=False)
script=Path('/data/coding/selective_flow/diagnose_gated_conditions_v1_20261005T0330Z.py')
assert hashlib.sha256(script.read_bytes()).hexdigest()=='f1067166d7da9297351f047d9921f82b9fbf0c09dae97e3771863704151b46fa'
args=['/data/coding/selective_flow/strong_baselines/.venv/bin/python',str(script),'--mode',c.mode,'--expected-gpu-uuid',c.uuid,'--source-root','/data/coding/selective_flow/inflow_gated_v3_deployment_20261005T0241Z','--out',str(root/('run_'+c.mode))]
env=dict(os.environ);env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
with (root/'diagnostic.log').open('xb') as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=root,env=env,start_new_session=True)
r={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'args':args,'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'mode':c.mode,'gpu_uuid':c.uuid,'type':'new gated frozen DEV diagnosis; no optimizer or TEST'}
with (root/'launch.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
