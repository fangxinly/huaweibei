"""Future capture completion/natural-exit proof; original stage stays unchanged."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
for n in ['root','dest','bundle','kind']:p.add_argument('--'+n,required=True)
p.add_argument('--checkpoint');a=p.parse_args();bundle=Path(a.bundle);dest=Path(a.dest)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();plan=json.loads((bundle/'matched_head_execution_plan.json').read_text())
if plan['status']!='COMPLETE_MATCHED_HEAD_EXECUTION_PROTOCOL_FROZEN':raise PermissionError('FULL_PROTOCOL_NOT_READY_NO_CAPTURE_CHILD')
source=bundle/'capture_fixed_matched_heads_v1.py';assert sha(source)==plan['source_and_role_sha256'][source.name] and not dest.exists()
argv=[sys.executable,str(source),'--root',a.root,'--dest',a.dest,'--kind',a.kind]
if a.checkpoint:argv+=['--checkpoint',a.checkpoint]
result=subprocess.Popen(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True);stdout,stderr=result.communicate()
record={'actual_exit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wrapper_pid':os.getpid(),'child_pid':result.pid,'child_full_argv':argv,'source_sha256':sha(source),'wrapper_source_sha256':sha(__file__),'exit_code':result.returncode,'natural_wait_verified':True,'stdout_sha256':hashlib.sha256(stdout.encode()).hexdigest(),'stderr_sha256':hashlib.sha256(stderr.encode()).hexdigest()}
if result.returncode==0:
 receipt=dest/'capture_receipt.json';assert receipt.is_file() and 'CAPTURE_COMPLETE' in stdout;record['original_receipt_sha256']=sha(receipt)
 (dest/'capture_actual_exit.json').write_text(json.dumps(record,indent=2)+'\n')
else:
 (Path(a.root)/('failed_capture_'+dest.name+'.json')).write_text(json.dumps(dict(record,stdout=stdout,stderr=stderr),indent=2)+'\n')
print(stdout,end='');print(stderr,file=sys.stderr,end='');print('ACTUAL_CAPTURE_NATURAL_EXIT '+json.dumps(record),flush=True);raise SystemExit(result.returncode)
