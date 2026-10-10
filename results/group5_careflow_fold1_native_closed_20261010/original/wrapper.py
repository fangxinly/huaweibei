REMOTE='/data/coding/g5careflow1_native_20261010T112228Z'
PYTHON='/data/coding/batch5_continue_actual_20261009T163550Z/N2_public_runtime_actual_20261009T174820Z/.venv/bin/python'
ASSETS='/data/coding/batch5_continue_actual_20261009T163550Z/N2_public_runtime_actual_20261009T174820Z'
PLAN_SHA='02d0e24c1f9bc385160468bd21f448926b6435e5949ed466d9fd67d1f6a41da3'
import datetime as dt,json,os,pathlib,subprocess
root=pathlib.Path(REMOTE)
env=dict(os.environ,PYTHONPATH=str(root/'source'),PYTHONDONTWRITEBYTECODE='1')
argv=[PYTHON,'-B',str(root/'source/group5_direct_runtime_v2.py'),'--plan',str(root/'plan.json'),'--plan-sha',PLAN_SHA,'--bundle',str(root/'source'),'--assets',ASSETS,'--out',str(root/'out'),'--method','careflow','--fold','1','--stage','precheck']
with (root/'stdout.log').open('xb') as log:
 child=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,env=env)
 (root/'dispatch.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),child=child.pid,argv=argv,plan_SHA=PLAN_SHA)))
 code=child.wait()
(root/'wrapper_exit.json').write_text(json.dumps(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),child=child.pid,natural_exit=code)))
raise SystemExit(code)
