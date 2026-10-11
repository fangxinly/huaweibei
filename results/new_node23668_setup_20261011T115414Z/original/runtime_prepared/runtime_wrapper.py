REMOTE='/root/autodl-tmp/ni_node23668_20261011T114428Z'
PLAN_SHA='b1190fe97e3249c5e29fe0a64244c968d2ef70037bc72c4ab540a3f9512d2b0b'
import datetime as dt,json,pathlib,subprocess,sys
P=pathlib.Path
root=P(REMOTE)
argv=[sys.executable,'-I','-S','-B',str(root/'new_node_23668_public_runtime.py'),'--plan',str(root/'dispatch_plan.json'),'--plan-sha',PLAN_SHA,'--root',str(root/'runtime')]
with (root/'runtime_stdout.log').open('xb') as log:
    child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (root/'runtime_dispatch.json').write_text(json.dumps({'actual_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':child.pid,'argv':argv,'plan_SHA':PLAN_SHA}))
    code=child.wait()
(root/'runtime_wrapper_exit.json').write_text(json.dumps({'actual_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':child.pid,'natural_exit':code}))
raise SystemExit(code)
