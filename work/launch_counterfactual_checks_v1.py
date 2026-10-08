from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['none','fixed','predicted'],required=True);a.add_argument('--uuid',required=True);c=a.parse_args()
root=Path('/data/coding/selective_flow/inflow_counterfactual_v5_checks_20261005T0502Z')
source=json.loads((root/'source_manifest.json').read_text())['source_sha256']
for name,digest in source.items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
assert json.loads((root/'check_plan.json').read_text())['budget_check_updates']==20
actual=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().strip();assert actual==c.uuid
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader']).decode().strip()
assert not (root/'check_launch.json').exists() and not (root/'check.log').exists()
base=Path('/data/coding/selective_flow/strong_baselines')
args=[str(base/'.venv/bin/python'),str(root/'check_counterfactual_v5.py'),'--stage','check','--mode',c.mode,'--expected-gpu-uuid',c.uuid,'--baseline-root',str(base),'--repo',str(base/'CaReFlow'),'--backbone',str(base/'deberta-v3-base'),'--data',str(base/'mosi.pkl'),'--out',str(root/('check_'+c.mode)),'--epochs','100','--seed','91814']
env=dict(os.environ);env.update(PYTHONHASHSEED='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONUNBUFFERED='1')
with (root/'check.log').open('xb') as f:p=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,cwd=root,env=env,start_new_session=True)
r={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stage':'mechanism_check_only','mode':c.mode,'gpu_uuid':actual,'pid':p.pid,'args':args,'launcher_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'source_manifest_sha256':hashlib.sha256((root/'source_manifest.json').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((root/'check_plan.json').read_bytes()).hexdigest(),'performance_training_started':False}
with (root/'check_launch.json').open('x',encoding='utf-8') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
