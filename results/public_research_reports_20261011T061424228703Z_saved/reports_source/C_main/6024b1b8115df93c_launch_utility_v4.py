from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess
a=argparse.ArgumentParser();a.add_argument('--stage',choices=['check','run'],required=True);a.add_argument('--mode',choices=['none','fixed','predicted'],required=True);a.add_argument('--uuid',required=True);a.add_argument('--initial-sha');c=a.parse_args()
root=Path('/data/coding/selective_flow/inflow_utility_v4_deployment_20261005T0346Z')
source=json.loads((root/'source_manifest.json').read_text())['source_sha256']
for name,digest in source.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
prefix='check' if c.stage=='check' else 'training'
assert not (root/(prefix+'_launch.json')).exists() and not (root/(prefix+'.log')).exists()
base=Path('/data/coding/selective_flow/strong_baselines')
args=[str(base/'.venv/bin/python'),str(root/'run_inflow_utility_v4.py'),'--stage',c.stage,'--mode',c.mode,'--expected-gpu-uuid',c.uuid,'--baseline-root',str(base),'--repo',str(base/'CaReFlow'),'--backbone',str(base/'deberta-v3-base'),'--data',str(base/'mosi.pkl'),'--out',str(root/(('check_' if c.stage=='check' else 'run_')+c.mode)),'--epochs','100','--seed','91813']
if c.stage=='run':
 assert c.initial_sha and len(c.initial_sha)==64
 args+=['--expected-initial-sha',c.initial_sha]
env=dict(os.environ);env.update(PYTHONHASHSEED='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
with (root/(prefix+'.log')).open('xb') as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=root,env=env,start_new_session=True)
r={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':c.stage,'mode':c.mode,'gpu_uuid':c.uuid,'pid':p.pid,'args':args,'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_manifest_sha256':hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest()}
with (root/(prefix+'_launch.json')).open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
