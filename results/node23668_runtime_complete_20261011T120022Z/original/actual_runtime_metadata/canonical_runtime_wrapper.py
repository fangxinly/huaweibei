REMOTE='/root/autodl-tmp/ni_node23668_20261011T114428Z'
PLAN_SHA='87c210ffaaade533931673038d47a4b0337034dddf8b6be161ba656aa84b81c7'
import datetime as dt,json,pathlib,subprocess,sys
P=pathlib.Path
root=P(REMOTE)
argv=[sys.executable,'-I','-S','-B',str(root/'new_node_23668_public_runtime.py'),'--plan',str(root/'canonical_dispatch_plan.json'),'--plan-sha',PLAN_SHA,'--root',str(root/'runtime_canonical')]
with (root/'canonical_runtime_stdout.log').open('xb') as log:
    child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
    (root/'canonical_runtime_dispatch.json').write_text(json.dumps({'actual_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':child.pid,'argv':argv,'plan_SHA':PLAN_SHA}))
    code=child.wait()
(root/'canonical_runtime_wrapper_exit.json').write_text(json.dumps({'actual_UTC':dt.datetime.now(dt.timezone.utc).isoformat(),'pid':child.pid,'natural_exit':code}))
raise SystemExit(code)
