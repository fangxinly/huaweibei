"""One natural input-only child, with original fullargv/source/pid/exit capture."""
import argparse,json,os,shutil,subprocess,sys
from pathlib import Path
from paired_final_TEST_identity_candidate_v1 import frozen_gate,require,sha,utc,write

def main():
    p=argparse.ArgumentParser()
    for n in ('root','bundle','assets','python'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=('A','B'),required=True)
    p.add_argument('--parent-identity',type=Path);p.add_argument('--parent-identity-sha')
    a=p.parse_args();plan=frozen_gate(a.bundle,a.plan_sha)
    require(a.root.parent==Path('/data/coding') and a.root.name.startswith('paired_final_TEST_identity_') and
            not a.root.exists(),'Fresh root, no duplicate input-only execution')
    require((a.parent_identity is None)==(a.node=='A'),'B original A identity required')
    a.root.mkdir();source=a.root/'source';source.mkdir()
    for name in [*plan['source_sha256'],'input_identity_plan.json']:
        shutil.copyfile(a.bundle/name,source/name);require(sha(source/name)==sha(a.bundle/name),'Copied original source mismatch')
    command=[str(a.python),str(a.bundle/'paired_final_TEST_identity_candidate_v1.py'),
             '--root',str(a.root),'--bundle',str(a.bundle),'--assets',str(a.assets),
             '--plan-sha',a.plan_sha,'--node',a.node]
    if a.parent_identity is not None:
        require(sha(a.parent_identity)==a.parent_identity_sha,'Original A physical receipt SHA')
        original=a.root/'original_A_input_identity.json';shutil.copyfile(a.parent_identity,original)
        command+=['--parent-identity',str(original),'--parent-identity-sha',a.parent_identity_sha]
    write(a.root/'actual_wrapper_start.json',{'actual_utc':utc(),'wrapper_pid':os.getpid(),
          'wrapper_fullargv':sys.argv,'child_fullargv':command,'plan_sha256':a.plan_sha,
          'child_source_sha256':sha(a.bundle/'paired_final_TEST_identity_candidate_v1.py')})
    with (a.root/'child.stdout.log').open('wb') as out,(a.root/'child.stderr.log').open('wb') as err:
        child=subprocess.Popen(command,stdout=out,stderr=err)
        write(a.root/'actual_child_launch.json',{'actual_utc':utc(),'child_pid':child.pid,'fullargv':command})
        code=child.wait()
    receipt=a.root/'out/actual_stage_receipt.json'
    write(a.root/'natural_exit.json',{'actual_utc':utc(),'child_pid':child.pid,'fullargv':command,
          'natural_exit':True,'exit_code':code,'original_stage_receipt_sha256':sha(receipt) if receipt.exists() else None})
    print(json.dumps({'actual_utc':utc(),'root':str(a.root),'child_pid':child.pid,'natural_exit_code':code}),flush=True)
    raise SystemExit(code)

if __name__=='__main__':main()
