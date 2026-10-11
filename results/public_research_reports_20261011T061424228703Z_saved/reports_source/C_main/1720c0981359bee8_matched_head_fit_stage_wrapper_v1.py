"""Future natural-exit wrapper; no start from incomplete protocol."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bundle',required=True);p.add_argument('--asset-base',required=True);p.add_argument('--candidate-arrays',required=True);p.add_argument('--parent-joint',required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
root=Path(a.root);bundle=Path(a.bundle);plan=json.loads((bundle/'matched_head_execution_plan.json').read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if plan['status']!='COMPLETE_MATCHED_HEAD_EXECUTION_PROTOCOL_FROZEN':raise PermissionError('FULL_PROTOCOL_NOT_READY_NO_CHILD')
assert root.parent==Path('/data/coding') and root.name.startswith('fixed_head_matched_fit_') and root.is_dir()
assert not (root/'out').exists() and not (root/'natural_exit.json').exists()
source=bundle/'fit_matched_fixed_heads_original_v1.py';assert sha(source)==plan['source_and_role_sha256'][source.name]
argv=[sys.executable,str(source),'--bundle',str(bundle),'--asset-base',a.asset_base,'--candidate-arrays',a.candidate_arrays,'--parent-joint',a.parent_joint,'--evidence',a.evidence,'--out',str(root/'out')]
env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
with (root/'child_stdout.log').open('x') as log,(root/'child_stderr.log').open('x') as err:
 child=subprocess.Popen(argv,stdout=log,stderr=err,env=env);start={'actual_start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wrapper_pid':os.getpid(),'child_pid':child.pid,'child_full_argv':argv,'source_sha256':sha(source),'wrapper_source_sha256':sha(__file__),'GPU_visibility_disabled':True}
 (root/'actual_child_launch.json').write_text(json.dumps(start,indent=2)+'\n');print('ACTUAL_MATCHED_FIT_STARTED '+json.dumps(start),flush=True);code=child.wait()
receipt=root/'out/actual_matched_head_fit_receipt.json';r=dict(start,actual_exit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,natural_wait_verified=True,stdout_sha256=sha(root/'child_stdout.log'),stderr_sha256=sha(root/'child_stderr.log'),original_receipt_available=receipt.is_file())
if receipt.is_file():r['original_receipt_sha256']=sha(receipt)
(root/'natural_exit.json').write_text(json.dumps(r,indent=2)+'\n');print('ACTUAL_MATCHED_FIT_EXIT '+json.dumps(r),flush=True);raise SystemExit(code)
