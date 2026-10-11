"""Capture actual separate support diagnostic exit; no fit or scoring entry."""
import datetime,json,pathlib,subprocess,sys,os
P=pathlib.Path;sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import seal,sha,raw
planpath=P(sys.argv[1]);expected=sys.argv[2];assert sha(planpath)==expected
p=json.loads(planpath.read_bytes());base=P(p['root']).parent;assert not (base/'natural_exit.json').exists()
env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (base/'diagnostic.out').open('xb') as out:
 child=subprocess.Popen([p['interpreter'],'-B',str(planpath.parent/'TRAIN_saved_fit_support_diagnostic_v1.py'),'--plan',str(planpath),'--sha',expected],env=env,stdout=out,stderr=subprocess.STDOUT)
 (base/'dispatch.json').write_bytes(raw(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=child.pid,plan_SHA=expected,role='SEPARATE_SAVED_FIT_SUPPORT_NO_LABELS_NO_FITTING_NO_SCORING')))
 code=child.wait()
exitrec=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),child=child.pid,natural_exit=code,plan_SHA=expected)
(base/'natural_exit.json').write_bytes(raw(exitrec))
if code==0:
 result=json.loads((P(p['root'])/'support_result.json').read_bytes());assert result['status']=='TRAIN_SAVED_FIT_SUPPORT_DESCRIPTION_COMPLETE' and result['TRAIN_labels_decoded']==0 and not result['VAL_TEST_numeric_decode'] and not result['new_solve_fit_score']
receipt=seal(base,'complete_actual_TRAIN_support_original.zip');receipt.update(exitrec)
(base/'capture_receipt.json').write_bytes(raw(receipt));print(json.dumps(receipt),flush=True);sys.exit(code)
