"""Natural child exit and raw argv receipt, no task/checkpoint reuse."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
p=argparse.ArgumentParser();p.add_argument('--asset-base',required=True);p.add_argument('--bundle',required=True);p.add_argument('--evidence',required=True);p.add_argument('--out',required=True);p.add_argument('--child',action='store_true');a=p.parse_args()
bundle=Path(a.bundle);plan=json.loads((bundle/'second_lease_precheck_plan_v5.json').read_text(encoding='utf-8'))
for name,digest in plan['source_sha256'].items():
 if sha(bundle/name)!=digest:raise ValueError('FROZEN_NEW_DEPLOYMENT_SOURCE_SHA: '+name)
if a.child:
    sys.path.insert(0,str(bundle))
    evidence=json.loads(Path(a.evidence).read_text(encoding='utf-8'))
    # Fresh driver identity and absence of other compute are checked again.
    uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if uuids!=evidence['gpu_uuid'] or compute:raise PermissionError('FRESH_CHILD_UUID_OR_COMPUTE_NOT_CLEAR')
    from precheck_minimal_fixed_v5 import run_precheck
    r=run_precheck(a.asset_base,bundle,a.out,evidence)
    print('ACTUAL_PRECHECK_COMPLETE '+json.dumps({'receipt':str(Path(a.out)/'actual_precheck_receipt.json'),'status':r['status'],'pid':os.getpid()}),flush=True)
    raise SystemExit(0)
root=Path(a.out).parent;root.mkdir(parents=True,exist_ok=True)
if (root/'natural_exit.json').exists():raise FileExistsError('UNIQUE_WRAPPER_ROOT_REQUIRED')
args=[sys.executable,str(Path(__file__).resolve()),'--child','--asset-base',a.asset_base,'--bundle',a.bundle,'--evidence',a.evidence,'--out',a.out]
with (root/'child_stdout.log').open('x') as stdout,(root/'child_stderr.log').open('x') as stderr:
    child=subprocess.Popen(args,stdout=stdout,stderr=stderr)
    launch={'actual_utc':utc(),'wrapper_pid':os.getpid(),'wrapper_argv':sys.argv,'child_pid':child.pid,'child_argv':args,'new_plan_sha256':sha(bundle/'second_lease_precheck_plan_v5.json'),'trusted_evidence_sha256':sha(a.evidence),'scope':'TWO_STEP_PRECHECK_ONLY_NOT_FORMAL100'}
    (root/'launch.json').write_text(json.dumps(launch,indent=2)+'\n',encoding='utf-8')
    # No forced termination of a healthy run. Candidate enforces 20min/6GiB
    # and complete save gates. Failure remains a natural nonzero child exit.
    code=child.wait()
exit={'actual_utc':utc(),'wrapper_pid':os.getpid(),'child_pid':child.pid,'child_argv':args,'exit_code':code,'natural_exit':True,'raw_stdout_sha256':sha(root/'child_stdout.log'),'raw_stderr_sha256':sha(root/'child_stderr.log'),'actual_scientific_precheck_complete':code==0 and (Path(a.out)/'actual_precheck_receipt.json').is_file(),'formal100_complete':False,'D_or_other_node_CPU_saved':False}
(root/'natural_exit.json').write_text(json.dumps(exit,indent=2)+'\n',encoding='utf-8')
print('NATURAL_PRECHECK_CHILD_EXIT '+json.dumps(exit),flush=True)
raise SystemExit(code)
