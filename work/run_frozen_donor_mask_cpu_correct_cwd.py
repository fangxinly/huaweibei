"""Supplemental wrapper only: correct cwd for unchanged, not-yet-started frozen CPU worker."""
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
def write(p,x):p.write_text(json.dumps(x,indent=2),encoding='utf8')
planpath=Path('/data/coding/fixed_weak_donor_mask_plan_20261007T161226Z.json');psha='499de527c37719e3f2129668c9dc4b730f9c21e903b1736a8494050bfb7e6d05'
assert hashlib.sha256(planpath.read_bytes()).hexdigest()==psha;plan=json.loads(planpath.read_text())
source=Path('/data/coding/fixed_weak_pilot_donor_mask_diagnostic_20261007T161226Z.py');assert hashlib.sha256(source.read_bytes()).hexdigest()==plan['diagnostic_source_sha256']
base=Path('/data/coding/fixed_weak_donor_mask_20261007T161226Z');execution=base/'B_execution_correct_cwd';assert not execution.exists();execution.mkdir()
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert uuid=='GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f'
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);assert not compute.strip()
write(execution/'actual_physical_preflight.json',dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),uuid=uuid,compute=compute,fullargv=[sys.executable]+sys.argv,process_table=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True),frozen_worker_SHA=plan['diagnostic_source_sha256'],plan_SHA=psha,CPU_model_forward=False))
argv=[sys.executable,str(source),'cpu','--plan',str(planpath),'--plan-sha',psha,'--root',str(base/'B_CPU_original'),'--input',str(base/'A_original'),'--node','B']
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf8')
with (execution/'stdout.log').open('wb') as out,(execution/'stderr.log').open('wb') as err:
 child=subprocess.Popen(argv,cwd=plan['B_bundle'],stdout=out,stderr=err,env=env)
 write(execution/'actual_child.json',dict(pid=child.pid,fullargv=argv,actual_start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=psha))
 code=child.wait()
write(execution/'natural_exit.json',dict(pid=child.pid,fullargv=argv,natural_exit=code,actual_exit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=psha))
print(json.dumps(dict(status='SUPPLEMENTAL_CWD_WRAPPER_FINISHED',natural_exit=code,pid=child.pid,fullargv=argv)),flush=True);sys.exit(code)
