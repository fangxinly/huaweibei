import ast,base64,datetime as dt,hashlib,json,pathlib,shutil
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent
ptrpath=BASE/'work/new_node_23668_runtime_pointer.json';ptr=json.loads(ptrpath.read_bytes());root=P(ptr['root']);remote=ptr['remote_root']
plan=json.loads((root/'plan.json').read_bytes())
plan.update(local_space_capture_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),local_C_free_bytes=shutil.disk_usage('C:/').free,local_D_free_bytes=shutil.disk_usage('D:/').free)
raw=(json.dumps(plan,ensure_ascii=False,indent=2)+'\n').encode();(root/'dispatch_plan.json').write_bytes(raw);h=hashlib.sha256(raw).hexdigest()
wrapper='''import datetime as dt,json,pathlib,subprocess,sys
P=pathlib.Path
root=P(REMOTE)
argv=[sys.executable,'-I','-S','-B',str(root/'new_node_23668_public_runtime.py'),'--plan',str(root/'dispatch_plan.json'),'--plan-sha',PLAN_SHA,'--root',str(root/'runtime')]
with (root/'runtime_stdout.log').open('xb') as log:
    child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (root/'runtime_dispatch.json').write_text(json.dumps({'actual_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':child.pid,'argv':argv,'plan_SHA':PLAN_SHA}))
    code=child.wait()
(root/'runtime_wrapper_exit.json').write_text(json.dumps({'actual_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':child.pid,'natural_exit':code}))
raise SystemExit(code)
'''
wrapper='REMOTE='+repr(remote)+'\nPLAN_SHA='+repr(h)+'\n'+wrapper
ast.parse(wrapper);(root/'runtime_wrapper.py').write_text(wrapper,encoding='utf8')
ptr.update(dispatch_plan_SHA=h,dispatch_plan_local=str(root/'dispatch_plan.json'),wrapper_SHA=hashlib.sha256(wrapper.encode()).hexdigest())
ptrpath.write_text(json.dumps(ptr,indent=2)+'\n')
payload={'remote_root':remote,'files':{'dispatch_plan.json':base64.b64encode(raw).decode(),'runtime_wrapper.py':base64.b64encode(wrapper.encode()).decode()},'plan_SHA':h}
print(json.dumps(payload))
