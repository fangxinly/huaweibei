"""Unique new stage root, complete source copies and original natural child wait."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from paired_final_TEST_pipeline_candidate_v1 import plan_gate, require, sha, utc, write

def main():
    p=argparse.ArgumentParser()
    for n in ('root','bundle','assets','python'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=('A','B','C'),required=True)
    p.add_argument('--stage',choices=('predict','cpu','score'),required=True)
    p.add_argument('--method');p.add_argument('--original',type=Path)
    p.add_argument('--score-protocol',type=Path);p.add_argument('--score-protocol-sha')
    a=p.parse_args();plan=plan_gate(a.bundle,a.plan_sha)
    require(a.root.parent==Path('/data/coding') and a.root.name.startswith('paired_final_TEST_') and
            not a.root.exists(),'Unique immutable original stage root required')
    a.root.mkdir();source=a.root/'source';source.mkdir()
    for name in list(plan['source_sha256'])+['final_pair_execution_plan.json']:
        target=source/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(a.bundle/name,target)
    plan_gate(source,a.plan_sha)
    command=[str(a.python),str(source/'paired_final_TEST_pipeline_candidate_v1.py'),
             '--root',str(a.root),'--bundle',str(source),'--assets',str(a.assets),
             '--plan-sha',a.plan_sha,'--node',a.node,'--stage',a.stage]
    if a.method:command+=['--method',a.method]
    if a.original:
        require(a.stage=='cpu' and a.original.is_dir(),'Only original CPU evidence copied')
        shutil.copytree(a.original,a.root/'original')
        command+=['--original',str(a.root/'original')]
    if a.score_protocol:
        require(a.stage=='score' and sha(a.score_protocol)==a.score_protocol_sha,'Exact once-score binding')
        shutil.copyfile(a.score_protocol,a.root/'score_protocol.json')
        command+=['--score-protocol',str(a.root/'score_protocol.json'),'--score-protocol-sha',a.score_protocol_sha]
    # Reserve before child launch. No equivalent command, root or nonce retry.
    token=Path(plan['fixed_stage_once_tokens'][a.stage][a.method if a.stage!='score' else 'pair'])
    write(token,{'actual_utc':utc(),'wrapper_pid':os.getpid(),'root':str(a.root),
                 'plan_sha256':a.plan_sha,'fullargv':command,'intent_before_child':True})
    shutil.copyfile(token,a.root/'actual_stage_attempt_intent.json')
    stdout_path=a.root/'child.stdout.log';stderr_path=a.root/'child.stderr.log'
    with stdout_path.open('xb') as stdout,stderr_path.open('xb') as stderr:
        child=subprocess.Popen(command,stdout=stdout,stderr=stderr)
        launch={'actual_utc':utc(),'wrapper_pid':os.getpid(),'wrapper_fullargv':sys.argv,
                'child_pid':child.pid,'fullargv':command,'root':str(a.root),
                'stage':a.stage,'method':a.method,'plan_sha256':a.plan_sha}
        write(a.root/'actual_child_launch.json',launch)
        code=child.wait()
    exitdata={'actual_utc':utc(),'natural_exit':True,'exit_code':code,
              'child_pid':child.pid,'fullargv':command,'stdout_sha256':sha(stdout_path),
              'stderr_sha256':sha(stderr_path),'original_launch_sha256':sha(a.root/'actual_child_launch.json')}
    receipt=a.root/'out/actual_stage_receipt.json'
    if code==0:
        require(receipt.is_file(),'Natural zero without stage receipt is incomplete')
        exitdata['stage_receipt_sha256']=sha(receipt)
    write(a.root/'natural_exit.json',exitdata)
    print('ACTUAL_ORIGINAL_CHILD_NATURAL_EXIT '+json.dumps(exitdata),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
