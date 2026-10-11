"""Dispatch one native audit of authenticated original bytes, then capture natural exit."""
import argparse, hashlib, json, os, shutil, subprocess, sys, zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);args=p.parse_args()
plan=json.loads(args.plan.read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
assert sha(__file__)==plan['operator_SHA']
bundle=Path(plan['bundle']);sys.path.insert(0,str(bundle));os.chdir(bundle)
from fold_evidence import verify_capsule
from fold_runtime import write,utc
assert sys.executable==plan['python']
assert sha(bundle/plan['protocol_name'])==plan['protocol_SHA']
manifest,original_receipt=verify_capsule(Path(plan['original_capsule']),plan['original_capsule_SHA'])
original=Path(plan['original']);assert original.is_dir()
checkpoint=original/'out'/Path(original_receipt.get('complete_checkpoint',original_receipt.get('complete_resume_and_selected'))['path']).name
assert sha(checkpoint)==plan['checkpoint_SHA']
with zipfile.ZipFile(plan['original_capsule']) as z:
    for name in z.namelist():
        if not name.startswith('run/'):continue
        dest=(original/name[4:]).resolve();assert dest.is_relative_to(original.resolve())
        data=z.read(name)
        if dest.exists():assert dest.read_bytes()==data
        else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
root=Path(plan['CPU_root']);assert not root.exists();root.mkdir();(root/'out').mkdir()
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert uuid==plan['B_UUID']
physical=dict(actual_utc=utc().isoformat(),uuid=uuid,
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),
    full_process_table=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True),
    remote_free_bytes=shutil.disk_usage(original).free,operator_pid=os.getpid(),operator_fullargv=[sys.executable]+sys.argv)
assert not physical['compute'].strip()
write(root/'out/actual_physical_CPU_audit_preflight.json',physical)
shutil.copy2(__file__,root/Path(__file__).name);shutil.copy2(args.plan,root/'operator_plan.json')
argv=[sys.executable,str(bundle/'fold_cpu_audit.py'),'--protocol',str(bundle/plan['protocol_name']),
    '--protocol-sha',plan['protocol_SHA'],'--bundle',str(bundle),'--original',str(original),
    '--output',str(root/'out/actual_stage_receipt.json')]
write(root/'CPU_worker_argv.json',argv)
wrapper=[sys.executable,str(bundle/'fold_evidence.py'),'run','--bundle',str(bundle),'--protocol',str(bundle/plan['protocol_name']),
    '--protocol-sha',plan['protocol_SHA'],'--execution',plan['CPU_execution'],'--argv-json',str(root/'CPU_worker_argv.json')]
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8')
with (root/'wrapper_stdout.log').open('wb') as out,(root/'wrapper_stderr.log').open('wb') as err:
    result=subprocess.run(wrapper,cwd=bundle,env=env,stdout=out,stderr=err)
write(root/'actual_operator_natural_wrapper_exit.json',dict(actual_utc=utc().isoformat(),natural_exit=result.returncode,wrapper_fullargv=wrapper))
assert result.returncode==0
capture=[sys.executable,str(bundle/'fold_evidence.py'),'capture','--original',str(root),
    '--execution',plan['CPU_execution'],'--bundle',str(bundle),'--protocol',str(bundle/plan['protocol_name']),
    '--protocol-sha',plan['protocol_SHA'],'--output',plan['CPU_capture']]
result=subprocess.run(capture,cwd=bundle,env=env)
print(json.dumps(dict(actual_utc=utc().isoformat(),status='CPU_ORIGINAL_AUDIT_AND_CAPTURE_OPERATOR_FINISHED',natural_exit=result.returncode,capture_fullargv=capture)),flush=True)
sys.exit(result.returncode)
