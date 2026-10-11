"""Natural complete/failure originals snapshot; all original root/source bytes."""
import argparse,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from paired_final_TEST_identity_candidate_v1 import frozen_gate,require,sha,utc,write,native_gate

def capture(a):
    plan=frozen_gate(a.bundle,a.plan_sha)
    require(a.root.parent==Path('/data/coding') and a.root.name.startswith('paired_final_TEST_identity_'),
            'Original input-only root')
    exitfile=a.root/'natural_exit.json';exitdata=json.loads(exitfile.read_text(encoding='utf-8'))
    require(exitdata['natural_exit'] is True,'Original natural wait absent')
    launch=json.loads((a.root/'actual_child_launch.json').read_text(encoding='utf-8'))
    require(launch['child_pid']==exitdata['child_pid'] and launch['fullargv']==exitdata['fullargv'],
            'Original child fullargv/exit mismatch')
    require(not a.dest.exists() and a.dest.parent==Path('/data/coding'),'New external snapshot root required')
    a.dest.mkdir()
    direct=native_gate(plan,a.assets,a.node,a.root)
    write(a.dest/'actual_capture_native_preflight.json',direct)
    if exitdata['exit_code']==0:
        receipt=a.root/'out/actual_stage_receipt.json'
        require(sha(receipt)==exitdata['original_stage_receipt_sha256'],'Original natural0 receipt SHA')
        r=json.loads(receipt.read_text(encoding='utf-8'))
        require(r['status']=='ACTUAL_OFFICIAL_TEST_INPUT_ONLY_IDENTITY_COMPLETE' and
                r['node']==a.node and r['plan_sha256']==a.plan_sha and r['all_role_labels_read'] is False,
                'Original successful input-only task identity')
    files=sorted(p for p in a.root.rglob('*') if p.is_file())
    require(all(not p.is_symlink() for p in files),'Only original regular files')
    manifest={str(p.relative_to(a.root)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files}
    for n,h in plan['source_sha256'].items():require(sha(a.root/'source'/n)==h,'Original root source changed')
    require(sha(a.root/'source/input_identity_plan.json')==a.plan_sha,'Original root plan mismatch')
    write(a.dest/'original_member_manifest.json',{'root':str(a.root),'members':manifest})
    with zipfile.ZipFile(a.dest/'snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,str(p.relative_to(a.root)))
    with zipfile.ZipFile(a.dest/'snapshot.zip') as z:
        require(z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest),
                'Original full ZIP CRC/unique')
        for name,spec in manifest.items():require(hashlib.sha256(z.read(name)).hexdigest()==spec['sha256'],'Full original member SHA')
    # Detect original file changes during snapshot, before COMPLETE.
    for p in files:require(sha(p)==manifest[str(p.relative_to(a.root))]['sha256'],'Original changed during snapshot')
    result={'status':'CAPTURE_COMPLETE_ORIGINAL_INPUT_ONLY_SUCCESS' if exitdata['exit_code']==0 else
                    'CAPTURE_COMPLETE_ORIGINAL_INPUT_ONLY_FAILURE',
            'actual_utc':utc(),'capture_child_pid':os.getpid(),'capture_fullargv':sys.argv,
            'node':a.node,'original_root':str(a.root),'plan_sha256':a.plan_sha,
            'original_natural_exit_sha256':sha(exitfile),'original_child_exit_code':exitdata['exit_code'],
            'original_child_pid':exitdata['child_pid'],'original_child_fullargv':exitdata['fullargv'],
            'snapshot_sha256':sha(a.dest/'snapshot.zip'),'snapshot_bytes':(a.dest/'snapshot.zip').stat().st_size,
            'member_manifest_sha256':sha(a.dest/'original_member_manifest.json'),
            'full_CRC_unique_all_member_SHA':True,'members':len(manifest),
            'direct_native_capture_sha256':sha(a.dest/'actual_capture_native_preflight.json')}
    write(a.dest/'capture_receipt.json',result)
    print('CAPTURE_COMPLETE '+json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser()
    for n in ('root','bundle','assets','dest','python'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=('A','B'),required=True)
    p.add_argument('--child',action='store_true');a=p.parse_args()
    if a.child:return capture(a)
    command=[str(a.python),str(Path(__file__).resolve()),*sys.argv[1:],'--child']
    child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    stdout,stderr=child.communicate()
    record={'actual_utc':utc(),'wrapper_pid':os.getpid(),'wrapper_fullargv':sys.argv,
            'capture_child_pid':child.pid,'capture_child_fullargv':command,'natural_exit_code':child.returncode,
            'stdout_sha256':hashlib.sha256(stdout).hexdigest(),'stderr_sha256':hashlib.sha256(stderr).hexdigest()}
    if child.returncode==0:
        require(b'CAPTURE_COMPLETE ' in stdout and (a.dest/'capture_receipt.json').exists(),'Original capture COMPLETE required')
        record['original_capture_receipt_sha256']=sha(a.dest/'capture_receipt.json')
        write(a.dest/'capture_actual_natural_exit.json',record)
        (a.dest/'capture.stdout.log').write_bytes(stdout);(a.dest/'capture.stderr.log').write_bytes(stderr)
    else:
        # Preserve actual failed capture, no overwrite or equivalent-tool retry.
        write(a.root/('capture_failure_'+a.dest.name+'.json'),dict(record,stdout=stdout.decode(errors='replace'),stderr=stderr.decode(errors='replace')))
    sys.stdout.buffer.write(stdout);sys.stderr.buffer.write(stderr)
    print('ACTUAL_CAPTURE_NATURAL_EXIT '+json.dumps(record),flush=True)
    raise SystemExit(child.returncode)

if __name__=='__main__':main()
