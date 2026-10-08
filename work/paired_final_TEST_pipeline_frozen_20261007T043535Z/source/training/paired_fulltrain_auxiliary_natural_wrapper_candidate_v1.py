"""Natural wait for an explicit input-only identity or fresh replay child; no duplicate run."""
import argparse
from pathlib import Path
import subprocess
import sys
from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write, plan_gate

def main():
    p=argparse.ArgumentParser()
    for n in ('root','bundle','python','child-arguments'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--tool',choices=('identity','fresh'),required=True)
    a=p.parse_args()
    if a.tool=='identity':
        f=a.bundle/'paired_TRAIN_DEV_input_identity_plan.json'
        require(sha(f)==a.plan_sha,'Exact input-only plan SHA required')
        plan=read(f)
        require(plan.get('status')=='PAIRED_TRAIN_DEV_IDENTITY_INPUT_ONLY_PROTOCOL_FROZEN' and
                plan.get('labels_enabled') is False and plan.get('model_or_training_enabled') is False and
                plan.get('TEST_entry_enabled') is False,'Frozen input-only identity role protocol required')
        for n,h in plan['source_sha256'].items():require(sha(a.bundle/n)==h,'Identity frozen source mismatch')
        name='paired_fulltrain_identity_input_only_candidate_v1.py'
    else:
        plan=plan_gate(a.bundle,a.plan_sha);name='paired_fulltrain_fresh_selected_replay_candidate_v1.py'
    require(a.root.parent==Path('/data/coding') and a.root.name.startswith('paired_fulltrain_'+a.tool+'_') and
            not (a.root/'out').exists() and not (a.root/'natural_exit.json').exists(),'Fresh auxiliary root required')
    arguments=read(a.child_arguments)
    require(isinstance(arguments,list) and all(isinstance(x,str) for x in arguments),'Physical auxiliary argument list required')
    for flag,value in (('--root',str(a.root)),('--bundle',str(a.bundle)),('--plan-sha',a.plan_sha)):
        require(arguments.count(flag)==1 and arguments[arguments.index(flag)+1]==value,'Auxiliary child field mismatch')
    source=a.bundle/name;require(sha(source)==plan['source_sha256'][name],'Auxiliary source SHA mismatch')
    command=[str(a.python),str(source),*arguments]
    write(a.root/'wrapper_actual_start.json',{'actual_utc':now(),'wrapper_argv':sys.argv,'child_full_argv':command,
          'child_source_sha256':sha(source),'child_args_sha256':sha(a.child_arguments)})
    with (a.root/'child.stdout.log').open('w') as stdout,(a.root/'child.stderr.log').open('w') as stderr:
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr)
        write(a.root/'actual_child_launch.json',{'actual_utc':now(),'pid':child.pid,'full_argv':command})
        code=child.wait()
    receipt=a.root/'out/actual_stage_receipt.json'
    write(a.root/'natural_exit.json',{'actual_utc':now(),'child_pid':child.pid,'exit_code':code,'natural_exit':True,
          'receipt_sha256':sha(receipt) if receipt.is_file() else None,'full_argv':command})
    require(code!=0 or receipt.is_file(),'Auxiliary natural0 without actual receipt')
    raise SystemExit(code)

if __name__=='__main__':main()
