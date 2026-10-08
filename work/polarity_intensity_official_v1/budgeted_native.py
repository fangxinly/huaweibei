"""Budgeted CPU synthetic qualification; records original process evidence."""
import datetime, hashlib, importlib.metadata, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def command(args):
    return subprocess.check_output(args, text=True).strip()

def capture(root):
    return dict(actual_UTC=now(), pid=os.getpid(), fullargv=[sys.executable]+sys.argv,
                python=sys.executable, torch=importlib.metadata.version('torch'),
                numpy=importlib.metadata.version('numpy'),
                GPU=command(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader']),
                compute=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']),
                processes=command(['ps','-eo','pid,ppid,args','--width','10000']),
                disk=shutil.disk_usage(root)._asdict())

def main():
    root=Path(__file__).resolve().parent
    out=root/'out'
    out.mkdir(exist_ok=False)
    plan=json.loads((root/'plan.json').read_text())
    pre=capture(root)
    (out/'pre_capture.json').write_text(json.dumps(pre,indent=2))
    observed={name:sha(root/name) for name in plan['sources']}
    (out/'source_verification.json').write_text(json.dumps(observed,indent=2))
    record=dict(start_UTC=now(), parent_pid=os.getpid(), fullargv=[sys.executable]+sys.argv,
                synthetic_only=True, real_data_or_weights=False, new_VAL_TEST_scores=False)
    try:
        assert observed==plan['sources'], 'Frozen source mismatch'
        assert pre['GPU'].split(',')[0] in plan['allowed_UUIDs'], 'Unexpected GPU UUID'
        assert pre['compute']=='', 'Preflight found active GPU compute; no intervention'
        assert pre['python']==plan['python'] and pre['torch']==plan['torch'] and pre['numpy']==plan['numpy'], 'Runtime mismatch'
        remaining=(datetime.datetime.fromisoformat(plan['conservative_cutoff_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
        assert remaining>=plan['timeout_seconds']+7200, 'Execution plus two-hour save reserve required'
        assert pre['disk']['free']>=32*1024*1024, 'Remote save space insufficient'
        argv=[sys.executable,str(root/'integration_contract.py'),str(out/'native_result.json')]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
        with (out/'stdout.txt').open('w') as stdout, (out/'stderr.txt').open('w') as stderr:
            child=subprocess.Popen(argv,cwd=root,stdout=stdout,stderr=stderr,env=env)
            record.update(child_pid=child.pid,child_fullargv=argv,child_start_UTC=now())
            try:
                code=child.wait(timeout=plan['timeout_seconds'])
                record.update(natural_exit_code=code,child_finish_UTC=now(),timeout=False)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
                record.update(natural_exit_code=None,child_finish_UTC=now(),timeout=True,terminated_own_synthetic_child_only=True)
        record['status']='NATIVE_COMPLETE' if record.get('natural_exit_code')==0 else 'NATIVE_FAILED'
    except Exception as exc:
        record.update(status='PREFLIGHT_FAILED',error=repr(exc),natural_exit_code=None)
    (out/'execution.json').write_text(json.dumps(record,indent=2))
    (out/'post_capture.json').write_text(json.dumps(capture(root),indent=2))
    files=sorted(p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='complete.zip')
    manifest={str(p.relative_to(root)):dict(bytes=p.stat().st_size,SHA256=sha(p)) for p in files}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    with zipfile.ZipFile(root/'complete.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in files+[out/'manifest.json']: z.write(path,str(path.relative_to(root)))
    with zipfile.ZipFile(root/'complete.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    print(json.dumps(dict(status=record['status'],complete_UTC=now(),ZIP_SHA256=sha(root/'complete.zip'),ZIP_bytes=(root/'complete.zip').stat().st_size,execution=record)),flush=True)

if __name__=='__main__': main()
