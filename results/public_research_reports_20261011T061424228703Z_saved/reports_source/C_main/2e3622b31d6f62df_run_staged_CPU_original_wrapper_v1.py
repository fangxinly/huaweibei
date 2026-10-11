"""Record natural child exit and original PID/argv/source for full CPU audits."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--auditor',required=True);p.add_argument('--run',required=True);p.add_argument('--out',required=True);p.add_argument('--prior-run');a=p.parse_args()
out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
assert not out.exists() and not out.with_suffix('.actual_exit.json').exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
argv=[sys.executable,a.auditor,'--run',a.run,'--out',a.out]
if a.prior_run:argv+=['--prior-run',a.prior_run]
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
with out.with_suffix('.stdout.log').open('x') as log,out.with_suffix('.stderr.log').open('x') as err:
 child=subprocess.Popen(argv,stdout=log,stderr=err,env=env)
 start={'actual_start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wrapper_pid':os.getpid(),'child_pid':child.pid,'child_full_argv':argv,'auditor_source_sha256':sha(a.auditor),'wrapper_source_sha256':sha(__file__),'GPU_visibility_disabled':True}
 out.with_suffix('.actual_start.json').write_text(json.dumps(start,indent=2)+'\n')
 print('ACTUAL_CPU_AUDIT_STARTED '+json.dumps(start),flush=True)
 code=child.wait()
record=dict(start,actual_exit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,natural_wait_verified=True,original_receipt_available=out.is_file())
if out.is_file():record['original_receipt_sha256']=sha(out)
out.with_suffix('.actual_exit.json').write_text(json.dumps(record,indent=2)+'\n');print('ACTUAL_CPU_AUDIT_EXIT '+json.dumps(record),flush=True)
raise SystemExit(code)
