from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,os
a=argparse.ArgumentParser();a.add_argument('--mode',choices=['none','fixed','predicted'],required=True);a.add_argument('--uuid',required=True);c=a.parse_args()
b=Path('/data/coding/selective_flow');src=b/'inflow_counterfactual_v5_deployment_20261005T0520Z';root=b/'counterfactual_diagnostics_20261005T0604Z';script=b/'diagnose_counterfactual_v5_v1_20261005T0604Z.py'
assert hashlib.sha256(script.read_bytes()).hexdigest()=='4094387b5cde52f750bf21d540087d58bb84e83b23ad367accffef02cd356c2e'
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().strip()==c.uuid
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).decode().strip()
assert json.loads((src/('run_'+c.mode)/'selection.json').read_text())['epochs']==100
root.mkdir(exist_ok=False)
args=[str(b/'strong_baselines/.venv/bin/python'),str(script),'--mode',c.mode,'--expected-gpu-uuid',c.uuid,'--source-root',str(src),'--out',str(root/('run_'+c.mode))]
with (root/'diagnostic.log').open('xb') as f:r=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
report={'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':r.pid,'argv':args,'mode':c.mode,'gpu_uuid':c.uuid,'source_sha256':hashlib.sha256(script.read_bytes()).hexdigest()}
(root/'launch.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
