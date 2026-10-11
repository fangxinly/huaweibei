"""Verified same-version wheel restore; never starts a GPU experiment."""
import argparse,datetime,hashlib,importlib.metadata,json,os,pathlib,signal,subprocess,sys,time,zipfile
p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--sha256',required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
r=pathlib.Path('/data/coding/multimodal_flow_public_20261006T1341Z');archive=pathlib.Path(a.archive)
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
assert sha(archive)==a.sha256
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()==a.expected_uuid
wheelroot=r/'offline_linux_wheels_v1';wheelroot.mkdir(exist_ok=False)
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n in z.namelist():
  dest=(wheelroot/n).resolve();assert dest.is_relative_to(wheelroot.resolve());dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(n))
manifest=json.loads((wheelroot/'wheel_manifest.json').read_text())
assert json.loads((wheelroot/'linux_dependency_closure.json').read_text())['complete']
for n,m in manifest['wheel_files'].items():assert sha(wheelroot/n)==m['sha256']
old=json.loads((r/'dependency_install_launch.json').read_text());proc=pathlib.Path('/proc')/str(old['pid']);interrupted=False;old_actual_argv=None
if proc.exists():
 state=next(x for x in (proc/'status').read_text().splitlines() if x.startswith('State:'))
 if 'Z (' not in state:
  old_actual_argv=(proc/'cmdline').read_bytes().decode().split('\0')[:-1]
  assert old_actual_argv==old['argv'],'REFUSE_TO_INTERRUPT_MISMATCHED_PROCESS'
  # Only this owned dependency download is interrupted after full verified
  # offline files are present. No GPU process or healthy training is stopped.
  os.kill(old['pid'],signal.SIGINT);interrupted=True;begin=time.monotonic()
  while proc.exists():
   state=next(x for x in (proc/'status').read_text().splitlines() if x.startswith('State:'))
   if 'Z (' in state:break
   if time.monotonic()-begin>120:raise RuntimeError('OWN_DOWNLOAD_DID_NOT_EXIT_NO_SECOND_WRITER_STARTED')
   time.sleep(1)
args=[sys.executable,'-m','pip','install','--no-index','--find-links',str(wheelroot),'--no-deps','-r',str(wheelroot/'offline_requirements.txt')]
with (r/'offline_dependency_v2_stdout.log').open('x') as out,(r/'offline_dependency_v2_stderr.log').open('x') as err:
 child=subprocess.Popen(args,stdout=out,stderr=err)
 (r/'offline_dependency_v2_launch.json').write_text(json.dumps({'actual_utc':utc(),'parent_pid':os.getpid(),'child_pid':child.pid,'child_argv':args,'archive_sha256':a.sha256,'owned_old_download_interrupted':interrupted,'old_pid':old['pid'],'old_actual_argv':old_actual_argv,'old_download_exit_code_unknown':True,'healthy_GPU_training_stopped':False},indent=2)+'\n')
 code=child.wait()
versions={name:importlib.metadata.version(name) for name in manifest['exact_versions']} if code==0 else {}
assert not versions or versions==manifest['exact_versions']
imports=subprocess.run([sys.executable,'-c','import torch,numpy,transformers,sentencepiece,sklearn,scipy,tqdm; print(torch.__version__,numpy.__version__)'],capture_output=True,text=True) if code==0 else None
freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True) if code==0 else None
if freeze:(r/'offline_dependency_v2_freeze.txt').write_text(freeze.stdout)
res={'actual_utc':utc(),'status':'VERIFIED_OFFLINE_LINUX_DEPS_COMPLETE' if code==0 and imports.returncode==0 and freeze.returncode==0 else 'OFFLINE_DEPENDENCY_FAILED','parent_pid':os.getpid(),'pip_child_pid':child.pid,'pip_child_argv':args,'pip_natural_exit_code':code,'original_download_exit_code_unknown':True,'owned_original_download_interrupted':interrupted,'exact_versions':versions,'imports_exit':None if imports is None else imports.returncode,'imports_stdout':None if imports is None else imports.stdout,'imports_stderr':None if imports is None else imports.stderr,'pip_freeze_exit':None if freeze is None else freeze.returncode,'archive_sha256':a.sha256,'source_sha256':sha(__file__),'actual_GPU_precheck_started':False,'healthy_training_stopped':False}
(r/'offline_dependency_v2_result.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res),flush=True)
raise SystemExit(0 if res['status']=='VERIFIED_OFFLINE_LINUX_DEPS_COMPLETE' else 1)
