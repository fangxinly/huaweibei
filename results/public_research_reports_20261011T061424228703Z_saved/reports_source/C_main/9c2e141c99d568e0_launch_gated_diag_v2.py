from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['none','state','task'],required=True);a.add_argument('--uuid',required=True);c=a.parse_args()
root=Path('/data/coding/selective_flow/gated_diagnostics_20261005T0332Z');root.mkdir(exist_ok=False)
script=Path('/data/coding/selective_flow/diagnose_gated_conditions_v2_20261005T0332Z.py')
assert hashlib.sha256(script.read_bytes()).hexdigest()=='1e7ade4228c4757a9fe9ea2d514e080a6b48e222850e44a48adc071a0cfa20a0'
args=['/data/coding/selective_flow/strong_baselines/.venv/bin/python',str(script),'--mode',c.mode,'--expected-gpu-uuid',c.uuid,'--source-root','/data/coding/selective_flow/inflow_gated_v3_deployment_20261005T0241Z','--out',str(root/('run_'+c.mode))]
env=dict(os.environ);env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
with (root/'diagnostic.log').open('xb') as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=root,env=env,start_new_session=True)
r={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'args':args,'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'mode':c.mode,'gpu_uuid':c.uuid,'type':'new gated frozen DEV diagnosis; no optimizer or TEST'}
with (root/'launch.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
