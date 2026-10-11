from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['none','fixed','predicted'],required=True);a.add_argument('--uuid',required=True);c=a.parse_args()
root=Path('/data/coding/selective_flow/utility_diagnostics_20261005T0439Z');root.mkdir(exist_ok=False)
script=Path('/data/coding/selective_flow/diagnose_utility_v4_v1_20261005T0439Z.py')
assert hashlib.sha256(script.read_bytes()).hexdigest()=='37b0df32b26c2e413543eb7551a15d0dbf55303c822a7397d646e1dfb29c8511'
args=['/data/coding/selective_flow/strong_baselines/.venv/bin/python',str(script),'--mode',c.mode,'--expected-gpu-uuid',c.uuid,'--source-root','/data/coding/selective_flow/inflow_utility_v4_deployment_20261005T0346Z','--out',str(root/('run_'+c.mode))]
env=dict(os.environ);env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
with (root/'diagnostic.log').open('xb') as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=root,env=env,start_new_session=True)
r={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':p.pid,'args':args,'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest(),'mode':c.mode,'gpu_uuid':c.uuid,'type':'utility frozen DEV diagnosis; zero optimizer updates and no TEST'}
with (root/'launch.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
