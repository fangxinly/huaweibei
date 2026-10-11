import datetime,hashlib,importlib.metadata as m,json,pathlib,subprocess,time,sys,os
r=pathlib.Path('/data/coding/multimodal_flow_public_20261006T1341Z')
launch=json.loads((r/'dependency_install_launch.json').read_text())
start=time.monotonic()
expected={'transformers':'4.37.2','sentencepiece':'0.1.99','scikit-learn':'1.5.2','scipy':'1.13.1','tqdm':'4.66.5'}
while True:
 p=pathlib.Path('/proc')/str(launch['pid'])
 state=next((x for x in (p/'status').read_text().splitlines() if x.startswith('State:')),'absent') if p.exists() else 'absent'
 if state=='absent' or 'Z (' in state:break
 if time.monotonic()-start>3600:raise RuntimeError('DEPENDENCY_WAIT_TIMEOUT_NO_GPU_STARTED')
 time.sleep(30)
versions={}
for k in expected:
 try:versions[k]=m.version(k)
 except m.PackageNotFoundError:versions[k]=None
if versions!=expected:
 result={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'ORIGINAL_INSTALL_NOT_VERIFIED_COMPLETE','versions':versions,'old_exit_unknown':True,'GPU_started':False}
 (r/'dependency_verification_result.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result),flush=True);raise SystemExit(1)
args=[sys.executable,'-m','pip','install']+[k+'=='+v for k,v in expected.items()]
t=datetime.datetime.now(datetime.timezone.utc).isoformat()
with (r/'idempotent_dependency_verify_stdout.log').open('x') as out,(r/'idempotent_dependency_verify_stderr.log').open('x') as err:
 child=subprocess.Popen(args,stdout=out,stderr=err)
 (r/'idempotent_dependency_verify_launch.json').write_text(json.dumps({'actual_utc':t,'pid':child.pid,'argv':args,'scope':'NEW_IDEMPOTENT_ENV_VERIFICATION_NOT_OLD_EXIT_REFILL'})+'\n')
 code=child.wait()
if code:raise RuntimeError('IDEMPOTENT_DEPENDENCY_VERIFICATION_FAILED_'+str(code))
imports=subprocess.run([sys.executable,'-c','import torch,transformers,sentencepiece,sklearn,scipy,tqdm,numpy;print(torch.__version__)'],capture_output=True,text=True)
freeze=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
(r/'verified_dependency_freeze.txt').write_text(freeze.stdout)
res={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'NEW_IDEMPOTENT_DEPENDENCY_NATURAL_EXIT_AND_IMPORTS_COMPLETE' if imports.returncode==0 and freeze.returncode==0 else 'IMPORT_OR_FREEZE_FAILED','old_install_exit_code_unknown':True,'new_pip_pid':child.pid,'new_pip_argv':args,'new_pip_natural_exit_code':code,'imports_exit_code':imports.returncode,'imports_stdout':imports.stdout,'imports_stderr':imports.stderr,'freeze_exit_code':freeze.returncode,'expected_versions':expected,'version_query':versions,'GPU_precheck_started':False,'own_pid':os.getpid()}
(r/'dependency_verification_result.json').write_text(json.dumps(res,indent=2)+'\n')
print('DEPENDENCY_VERIFICATION '+json.dumps(res),flush=True)
raise SystemExit(0 if imports.returncode==0 and freeze.returncode==0 else 1)

