import ast,base64,datetime as dt,hashlib,json,pathlib,shutil
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent
ptrpath=BASE/'work/new_node_23668_runtime_pointer.json';ptr=json.loads(ptrpath.read_bytes());root=P(ptr['root']);remote=ptr['remote_root']
failure=BASE/'work/new_node_23668_preflight/runtime_failure_original_capture.zip'
assert hashlib.sha256(failure.read_bytes()).hexdigest()=='00d6e97beb50a2a5e800aa0adc20df3c09c8b4fdc7326e2f755db8e792faf8d2'
plan=json.loads((root/'dispatch_plan.json').read_bytes())
plan.update(local_space_capture_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),local_C_free_bytes=shutil.disk_usage('C:/').free,local_D_free_bytes=shutil.disk_usage('D:/').free,new_runtime_root=remote+'/runtime_canonical')
plan['torch_wheel']['first_official_mirror_HTTP403_original_capture_SHA']=hashlib.sha256(failure.read_bytes()).hexdigest()
plan['torch_wheel']['wheel_link_with_original_SHA']='https://download.pytorch.org/whl/cu121/torch-2.1.0%2Bcu121-cp310-cp310-linux_x86_64.whl#sha256=0d4e8c52a1fcf5ed6cfc256d9a370fcf4360958fc79d0b08a51d55e70914df46'
plan['torch_wheel']['canonical_HEAD_actual_UTC']='2026-10-11T11:51:03.806460+00:00'
plan['torch_wheel']['canonical_HEAD_Content_Length']=2200630334
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();h=hashlib.sha256(raw).hexdigest()
with (root/'canonical_dispatch_plan.json').open('xb') as f:f.write(raw)
wrapper='''import datetime as dt,json,pathlib,subprocess,sys
P=pathlib.Path
root=P(REMOTE)
argv=[sys.executable,'-I','-S','-B',str(root/'new_node_23668_public_runtime.py'),'--plan',str(root/'canonical_dispatch_plan.json'),'--plan-sha',PLAN_SHA,'--root',str(root/'runtime_canonical')]
with (root/'canonical_runtime_stdout.log').open('xb') as log:
    child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (root/'canonical_runtime_dispatch.json').write_text(json.dumps({'actual_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':child.pid,'argv':argv,'plan_SHA':PLAN_SHA}))
    code=child.wait()
(root/'canonical_runtime_wrapper_exit.json').write_text(json.dumps({'actual_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':child.pid,'natural_exit':code}))
raise SystemExit(code)
'''
wrapper='REMOTE='+repr(remote)+'\nPLAN_SHA='+repr(h)+'\n'+wrapper
ast.parse(wrapper)
with (root/'canonical_runtime_wrapper.py').open('x',encoding='utf8') as f:f.write(wrapper)
ptr.update(canonical_dispatch_plan_SHA=h,canonical_runtime_root=plan['new_runtime_root'],first_runtime_attempt_natural_exit=1,first_runtime_failure_capture_SHA=hashlib.sha256(failure.read_bytes()).hexdigest())
ptrpath.write_text(json.dumps(ptr,indent=2)+'\n')
print(json.dumps({'remote_root':remote,'files':{'canonical_dispatch_plan.json':base64.b64encode(raw).decode(),'canonical_runtime_wrapper.py':base64.b64encode(wrapper.encode()).decode()},'plan_SHA':h}))
