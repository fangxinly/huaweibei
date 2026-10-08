"""Natural process receipt; launcher success is not training completion."""
import argparse,datetime,hashlib,json,os,pathlib,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--run',required=True);p.add_argument('--evidence',required=True);p.add_argument('--phase',choices=['stage10','continue100'],required=True);p.add_argument('--resume');a=p.parse_args()
root=pathlib.Path(a.run);bundle=pathlib.Path(a.bundle)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
plan=json.loads((bundle/'staged_reference_plan.json').read_text());source=bundle/'minimal_fixed_staged_training_v3.py'
assert sha(source)==plan['source_sha256'][source.name]
args=['/data/coding/multimodal_flow_public_20261006T1341Z/.venv/bin/python',str(source),'--asset-base','/data/coding/multimodal_flow_public_20261006T1341Z','--bundle',str(bundle),'--evidence',a.evidence,'--out',str(root/'out'),'--phase',a.phase]
if a.resume:args+=['--resume',a.resume]
start=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (root/'child_stdout.log').open('x') as out,(root/'child_stderr.log').open('x') as err:
 child=subprocess.Popen(args,stdout=out,stderr=err)
 (root/'actual_child_launch.json').write_text(json.dumps({'actual_utc':start,'wrapper_pid':os.getpid(),'child_pid':child.pid,'child_full_argv':args,'source_sha256':sha(source)},indent=2)+'\n')
 code=child.wait()
record={'actual_start_utc':start,'actual_exit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wrapper_pid':os.getpid(),'child_pid':child.pid,'child_full_argv':args,'exit_code':code,'natural_wait_verified':True,'source_sha256':sha(source),'stdout_sha256':sha(root/'child_stdout.log'),'stderr_sha256':sha(root/'child_stderr.log'),'actual_training_receipt_available':(root/'out/actual_training_receipt.json').is_file(),'formal100_complete':False,'D_B_CPU_preservation_complete':False}
if record['actual_training_receipt_available']:
 record['original_training_receipt_sha256']=sha(root/'out/actual_training_receipt.json')
 record['formal100_complete']=json.loads((root/'out/actual_training_receipt.json').read_text())['formal100_complete']
(root/'natural_exit.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record),flush=True);raise SystemExit(code)
