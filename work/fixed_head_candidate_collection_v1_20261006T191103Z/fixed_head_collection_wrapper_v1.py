import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--bundle',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--joint',required=True);p.add_argument('--evidence',required=True);a=p.parse_args()
root=Path(a.root);bundle=Path(a.bundle);plan=json.loads((bundle/'candidate_collection_plan.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source=bundle/'fixed_head_input_collector_v1.py';assert sha(source)==plan['source_and_role_sha256'][source.name]
assert not (root/'natural_exit.json').exists() and not (root/'out').exists()
argv=[sys.executable,str(source),'--asset-base','/data/coding/multimodal_flow_public_20261006T1341Z','--bundle',str(bundle),'--checkpoint',a.checkpoint,'--completion-joint',a.joint,'--evidence',a.evidence,'--out',str(root/'out')]
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (root/'child_stdout.log').open('x') as log,(root/'child_stderr.log').open('x') as err:
 child=subprocess.Popen(argv,stdout=log,stderr=err)
 actual={'actual_start_utc':start,'wrapper_pid':os.getpid(),'child_pid':child.pid,'child_full_argv':argv,'source_sha256':sha(source)}
 (root/'actual_child_launch.json').write_text(json.dumps(actual,indent=2)+'\n');print('ACTUAL_COLLECTION_STARTED '+json.dumps(actual),flush=True)
 code=child.wait()
r=dict(actual,actual_exit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,natural_wait_verified=True,stdout_sha256=sha(root/'child_stdout.log'),stderr_sha256=sha(root/'child_stderr.log'),actual_candidate_receipt_available=(root/'out/actual_candidate_collection_receipt.json').is_file())
if r['actual_candidate_receipt_available']:r['original_receipt_sha256']=sha(root/'out/actual_candidate_collection_receipt.json')
(root/'natural_exit.json').write_text(json.dumps(r,indent=2)+'\n');print('ACTUAL_COLLECTION_EXIT '+json.dumps(r),flush=True);raise SystemExit(code)
