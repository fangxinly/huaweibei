REMOTE='/data/coding/g5careflow100_20261010T043810Z'
PYTHON='/data/coding/batch5_continue_actual_20261009T163550Z/N2_public_runtime_actual_20261009T174820Z/.venv/bin/python'
ASSETS='/data/coding/batch5_continue_actual_20261009T163550Z/N2_public_runtime_actual_20261009T174820Z'
PLAN_SHA='443b6eb01ba6f4f3f2e14119d93715abe8a8a6a2590019a42dc05a33f91e6845'
import datetime as dt,json,os,pathlib,subprocess
P=pathlib.Path
root=P(REMOTE)
env=os.environ.copy();env['PYTHONPATH']=str(root/'source')
argv=[PYTHON,'-B',str(root/'source/group5_direct_runtime_v2.py'),'--plan',str(root/'plan.json'),'--plan-sha',PLAN_SHA,'--bundle',str(root/'source'),'--assets',ASSETS,'--out',str(root/'out'),'--method','careflow','--fold','0','--stage','train']
with (root/'stdout.log').open('xb') as log:
    child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env)
    (root/'dispatch.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),child=child.pid,argv=argv,plan_SHA=PLAN_SHA)),encoding='utf8')
    code=child.wait()
(root/'wrapper_exit.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),child=child.pid,natural_exit=code)),encoding='utf8')
raise SystemExit(code)
