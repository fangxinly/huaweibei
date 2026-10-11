import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--clock',required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
base=Path('D:/CodexBackups/selective_flow_20261003_1105')
stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
root=base/('inner_lovo_output_gate_'+stamp);assert not root.exists();root.mkdir();bundle=root/'source';bundle.mkdir()
old=base/'weak_prediction_gate_control_20261007T124056Z/source';oldplan=json.loads((old/'protocol.json').read_text(encoding='utf8'))
src=Path('work/inner_lovo_output_gate_diagnostic.py');metric=Path('work/sentiment_metrics_careflow_v1.py')
assert sha(metric)=='a91e46693d8a690e20240a39232c9b63287d7c7fc6148711bb474e6b6e9b99a6'
for item in [src,metric]:shutil.copy2(item,bundle/item.name)
inputs={}
for name,h in oldplan['input_sha256'].items():
 assert sha(old/name)==h;shutil.copy2(old/name,bundle/name);inputs[name]=h
plan=dict(status='SAVED_PREDICTION_INNER_LOVO_OUTPUT_GATE_DIAGNOSTIC_FROZEN',actualclock_freeze_UTC=a.clock,
 scope='Existing selected20 b/p output interpolation on already explored INNER. b is raw readout, not same-flow donor-message-off p0. Checkpoint selected on all INNER, therefore LOVO gate predictions are exploratory, not fully nested independent OOF. No TEST or OUTER evaluation, no candidate selection or deployment.',
 ridge=.01,features=['abs(b)','abs(p-b)','sign(b)*sign(p-b)'],intercept_penalty=0.,training_weight='equal videos, equal rows within each training video',
 gate='sigmoid(intercept + standardized fixed three features)',optimizer='BFGS zero initialization; max600, gradient_inf<1e-7; Armijo1e-4',
 constant_gate='clip(sum a*delta*(y-b)/sum a*delta^2,0,1); zero if denominator zero',
 bootstrap_draws=10000,bootstrap_seed=20261007,bootstrap_unit='video paired; preserve all rows; same draws for both gates and regions',
 bootstrap_scope='Fixed LOVO predictions conditional on selected checkpoint and fitted gates; no bootstrap refitting or checkpoint reselection. Fraction below zero is descriptive, not probability of population improvement.',
 weak_definition='abs(y)<=1 for reporting only; no true-label gate input',
 source_sha256={q.name:sha(q) for q in bundle.iterdir() if q.suffix=='.py'},input_sha256=inputs,
 parent_checkpoint_SHA=oldplan['parent_checkpoint_SHA'],parent_selected_state_SHA=oldplan['selected_model_SHA'],parent_original_receipt_SHA=oldplan['parent_receipt_SHA'],
 no_new_GPU_forward=True,no_model_training=True,no_TEST_read=True)
write(bundle/'protocol.json',plan);psha=sha(bundle/'protocol.json')
cmd=[sys.executable,'-X','utf8',str((bundle/src.name).resolve()),'--plan',str((bundle/'protocol.json').resolve()),'--plan-sha',psha,'--output',str((root/'actual').resolve())]
write(root/'actual_child_dispatch.json',dict(actual_start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),fullargv=cmd))
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf8')
with (root/'stdout.log').open('wb') as out,(root/'stderr.log').open('wb') as err:
 child=subprocess.Popen(cmd,stdout=out,stderr=err,env=env);code=child.wait()
write(root/'natural_exit.json',dict(pid=child.pid,fullargv=cmd,natural_exit=code,actual_exit_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),protocol_SHA=psha))
shutil.copy2(__file__,root/Path(__file__).name)
if code:print((root/'stderr.log').read_text(encoding='utf8'));sys.exit(code)
receipt=json.loads((root/'actual/actual_stage_receipt.json').read_text(encoding='utf8'));assert receipt['pid']==child.pid and receipt['fullargv']==cmd
members={str(q.relative_to(root)).replace('\\','/'):sha(q) for q in root.rglob('*') if q.is_file()}
write(root/'local_original_manifest.json',dict(actual_capture_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),member_sha256=members,source='Actual local CPU run, not remote GPU capture'))
with zipfile.ZipFile(root/'complete_local_CPU_capture.zip','x',zipfile.ZIP_DEFLATED) as z:
 for q in root.rglob('*'):
  if q.is_file() and q.suffix!='.zip':z.write(q,str(q.relative_to(root)).replace('\\','/'))
with zipfile.ZipFile(root/'complete_local_CPU_capture.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
write(root/'capture_receipt.json',dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=sha(root/'complete_local_CPU_capture.zip'),bytes=(root/'complete_local_CPU_capture.zip').stat().st_size))
pointer=dict(D=str(root),protocol_SHA=psha,capture_SHA=sha(root/'complete_local_CPU_capture.zip'),status=receipt['status'])
write(Path('work/inner_lovo_output_gate_current.json'),pointer);print(json.dumps(pointer));print(json.dumps(receipt['inner'],ensure_ascii=False))
