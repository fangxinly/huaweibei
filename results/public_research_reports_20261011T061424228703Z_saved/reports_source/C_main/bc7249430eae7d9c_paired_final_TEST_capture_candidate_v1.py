"""Complete original root capture after natural exit; no model/label access."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from paired_final_TEST_pipeline_candidate_v1 import plan_gate,native_gate,require,sha,load,utc,write

def capture(a):
    plan=plan_gate(a.bundle,a.plan_sha)
    require(a.root.parent==Path('/data/coding') and a.root.name.startswith('paired_final_TEST_'), 'Original root scope')
    launch,exitdata=load(a.root/'actual_child_launch.json'),load(a.root/'natural_exit.json')
    require(exitdata['natural_exit'] is True and exitdata['child_pid']==launch['child_pid'] and
            exitdata['fullargv']==launch['fullargv'] and
            exitdata['original_launch_sha256']==sha(a.root/'actual_child_launch.json') and
            sha(a.root/'child.stdout.log')==exitdata['stdout_sha256'] and
            sha(a.root/'child.stderr.log')==exitdata['stderr_sha256'],'Original child/argv/full logs/natural exit')
    require(a.dest.parent==Path('/data/coding') and not a.dest.exists(),'New external capsule required')
    a.dest.mkdir();native=native_gate(plan,a.assets,a.node,a.root)
    write(a.dest/'actual_capture_native_preflight.json',native)
    plan_gate(a.root/'source',a.plan_sha)
    if exitdata['exit_code']==0:
        require(sha(a.root/'out/actual_stage_receipt.json')==exitdata['stage_receipt_sha256'], 'Original receipt SHA')
    files=sorted(p for p in a.root.rglob('*') if p.is_file())
    require(all(not p.is_symlink() for p in files),'Original regular files only')
    manifest={str(p.relative_to(a.root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    write(a.dest/'original_member_manifest.json',{'original_root':str(a.root),'members':manifest})
    with zipfile.ZipFile(a.dest/'snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in files:z.write(f,str(f.relative_to(a.root)))
    with zipfile.ZipFile(a.dest/'snapshot.zip') as z:
        require(z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest), 'Complete CRC/unique')
        for name,item in manifest.items():
            require(hashlib.sha256(z.read(name)).hexdigest()==item['sha256'] and
                    sha(a.root/name)==item['sha256'],'All original SHA and unchanged during capture')
    receipt={'status':'CAPTURE_COMPLETE_FINAL_TEST_ORIGINAL_SUCCESS' if exitdata['exit_code']==0 else
                      'CAPTURE_COMPLETE_FINAL_TEST_ORIGINAL_FAILURE',
             'actual_utc':utc(),'capture_child_pid':os.getpid(),'capture_fullargv':sys.argv,
             'plan_sha256':a.plan_sha,'node':a.node,'original_root':str(a.root),
             'original_child_pid':exitdata['child_pid'],'original_child_fullargv':exitdata['fullargv'],
             'original_child_exit_code':exitdata['exit_code'],'natural_exit_sha256':sha(a.root/'natural_exit.json'),
             'snapshot_sha256':sha(a.dest/'snapshot.zip'),'snapshot_bytes':(a.dest/'snapshot.zip').stat().st_size,
             'manifest_sha256':sha(a.dest/'original_member_manifest.json'),'members':len(manifest),
             'native_sha256':sha(a.dest/'actual_capture_native_preflight.json'),'full_CRC_unique_member_SHA':True}
    write(a.dest/'capture_receipt.json',receipt)
    print('CAPTURE_COMPLETE '+json.dumps(receipt),flush=True)

def main():
    p=argparse.ArgumentParser()
    for n in ('root','bundle','assets','dest','python'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=('A','B','C'),required=True)
    p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:return capture(a)
    command=[str(a.python),str(Path(__file__).resolve()),*sys.argv[1:],'--child']
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=child.communicate()
    record={'actual_utc':utc(),'wrapper_pid':os.getpid(),'wrapper_fullargv':sys.argv,
            'capture_child_pid':child.pid,'capture_child_fullargv':command,'natural_exit_code':child.returncode,
            'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'stderr_sha256':hashlib.sha256(stderr).hexdigest()}
    if child.returncode==0:
        require(b'CAPTURE_COMPLETE ' in stdout,'Capture COMPLETE marker absent')
        record['capture_receipt_sha256']=sha(a.dest/'capture_receipt.json')
        write(a.dest/'capture_actual_natural_exit.json',record)
        (a.dest/'capture.stdout.log').write_bytes(stdout);(a.dest/'capture.stderr.log').write_bytes(stderr)
    else:
        write(a.root/('capture_failure_'+a.dest.name+'.json'),dict(record,stdout=stdout.decode(errors='replace'),stderr=stderr.decode(errors='replace')))
    sys.stdout.buffer.write(stdout);sys.stderr.buffer.write(stderr)
    print('ACTUAL_CAPTURE_NATURAL_EXIT '+json.dumps(record),flush=True)
    raise SystemExit(child.returncode)

if __name__=='__main__':main()
