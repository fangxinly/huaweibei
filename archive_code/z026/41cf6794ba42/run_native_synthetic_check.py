import subprocess,json,sys,hashlib,datetime,zipfile,shutil,os
from pathlib import Path
bundle=Path(__file__).resolve().parent;plan=json.loads((bundle/'candidate_protocol.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert plan['execution_enabled'] is False
for n,h in plan['source_sha256'].items():assert sha(bundle/n)==h
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']).decode().strip();assert uuid=='GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f'
root=bundle.parent/'native_check';root.mkdir();argv=[sys.executable,str(bundle/'check_weak_retention_native.py')];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['PYTHONDONTWRITEBYTECODE']='1'
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
(root/'preflight.json').write_text(json.dumps(dict(actual_utc=utc(),uuid=uuid,processes=subprocess.check_output(['ps','-eo','pid,args','--width','4000']).decode(),compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']).decode(),space=shutil.disk_usage('/data')._asdict())))
with (root/'stdout.log').open('wb') as out,(root/'stderr.log').open('wb') as err:
 child=subprocess.Popen(argv,stdout=out,stderr=err,env=env,cwd=bundle);(root/'actual_child.json').write_text(json.dumps(dict(pid=child.pid,fullargv=argv,actual_start_utc=utc())));code=child.wait(timeout=60)
(root/'natural_exit.json').write_text(json.dumps(dict(pid=child.pid,fullargv=argv,natural_exit=code,actual_exit_utc=utc())))
if code==0:assert json.loads((root/'stdout.log').read_text())['weak_harm_gradient_direction_passed']
capture=bundle.parent/'complete_native_capture.zip'
files=[p for p in bundle.parent.rglob('*') if p.is_file() and p!=capture];manifest={p.relative_to(bundle.parent).as_posix():sha(p) for p in files};(bundle.parent/'capture_manifest.json').write_text(json.dumps(dict(actual_utc=utc(),member_sha256=manifest)))
with zipfile.ZipFile(capture,'x',zipfile.ZIP_DEFLATED) as z:
 for p in files+[bundle.parent/'capture_manifest.json']:z.write(p,p.relative_to(bundle.parent).as_posix())
with zipfile.ZipFile(capture) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
(bundle.parent/'capture_receipt.json').write_text(json.dumps(dict(actual_utc=utc(),sha256=sha(capture),natural_exit=code)))
print('NATIVE_CHECK_NATURAL_EXIT_'+str(code),flush=True);raise SystemExit(code)
