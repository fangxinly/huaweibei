"""Future allowlisted natural-exit stage wrapper. No credentials or daemon shell."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser()
for n in ['root','bundle','stage','child-arguments']:p.add_argument('--'+n,required=True)
a=p.parse_args();bundle=Path(a.bundle);root=Path(a.root)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
plan=read(bundle/'matched_head_execution_plan.json')
if plan['status']!='COMPLETE_MATCHED_HEAD_EXECUTION_PROTOCOL_FROZEN':raise PermissionError('FULL_PROTOCOL_NOT_READY_NO_CHILD')
allowed={'predict':('fixed_head_development_collector_v1.py','out/actual_candidate_collection_receipt.json'),'score':('score_frozen_matched_heads201_v1.py','out/actual_matched_head_evaluation_receipt.json'),'audit_fit':('audit_matched_head_fit_original_CPU_v1.py','cpu_original_receipt.json'),'audit_predict':('audit_matched_development_arrays_CPU_v1.py','cpu_original_receipt.json'),'audit_score':('audit_matched_development_scores_CPU_v1.py','cpu_original_receipt.json')}
if a.stage not in allowed:raise PermissionError('PREDECLARED_STAGE_ONLY')
assert root.parent==Path('/data/coding') and root.name.startswith('fixed_head_matched_') and root.is_dir() and not (root/'natural_exit.json').exists() and not (root/'out').exists()
name,receiptname=allowed[a.stage];source=bundle/name
assert sha(source)==plan['source_and_role_sha256'][name]
arguments=read(a.child_arguments)
assert isinstance(arguments,list) and all(isinstance(x,str) for x in arguments)
assert arguments.count('--out')==1 and arguments[arguments.index('--out')+1]==str(root/('out' if a.stage in ('predict','score') else 'cpu_original_receipt.json'))
argv=[sys.executable,str(source),*arguments];env=os.environ.copy()
if a.stage!='predict':env['CUDA_VISIBLE_DEVICES']=''
with (root/'child_stdout.log').open('x') as log,(root/'child_stderr.log').open('x') as err:
 child=subprocess.Popen(argv,stdout=log,stderr=err,env=env)
 start={'actual_start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wrapper_pid':os.getpid(),'child_pid':child.pid,'child_full_argv':argv,'source_sha256':sha(source),'wrapper_source_sha256':sha(__file__),'stage':a.stage,'plan_sha256':sha(bundle/'matched_head_execution_plan.json')}
 (root/'actual_child_launch.json').write_text(json.dumps(start,indent=2)+'\n');print('ACTUAL_MATCHED_STAGE_STARTED '+json.dumps(start),flush=True);code=child.wait()
receipt=root/receiptname
record=dict(start,actual_exit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,natural_wait_verified=True,stdout_sha256=sha(root/'child_stdout.log'),stderr_sha256=sha(root/'child_stderr.log'),original_receipt_available=receipt.is_file())
if receipt.is_file():record['original_receipt_sha256']=sha(receipt)
(root/'natural_exit.json').write_text(json.dumps(record,indent=2)+'\n');print('ACTUAL_MATCHED_STAGE_EXIT '+json.dumps(record),flush=True);raise SystemExit(code)
