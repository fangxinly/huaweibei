import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--clock',required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
D=Path('D:/CodexBackups/selective_flow_20261003_1105');stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z';root=D/('formal_video_paired_bootstrap_'+stamp);assert not root.exists();root.mkdir();bundle=root/'source';bundle.mkdir()
def check_snapshot(parent):
 r=json.loads((parent/'capture_receipt.json').read_text(encoding='utf8'));assert sha(parent/'snapshot.zip')==r['snapshot_sha256']
 with zipfile.ZipFile(parent/'snapshot.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 return r
testparent=D/'paired_final_TEST_prediction_actual_20261007T043535Z/score_pair';tr=check_snapshot(testparent);tm=json.loads((testparent/'original_member_manifest.json').read_text(encoding='utf8'));assert sha(testparent/'original_member_manifest.json')==tr['manifest_sha256']
tp=testparent/'run/out/joint_targets_and_fixed_predictions.npz';assert sha(tp)==tm['members']['out/joint_targets_and_fixed_predictions.npz']['sha256'];shutil.copy2(tp,bundle/'test_fixed_pair.npz')
devparent=D/'paired_fulltrain100_complete_actual_20261007T030311Z';devrecord=json.loads(Path('outputs/正式双方fullTRAIN100固定best一次DEV五项实际结果.json').read_text(encoding='utf8'))
dev={};refs={}
for name,epoch in [('minimal_fixed_F',89),('careflow',93)]:
 parent=devparent/name/'a';cr=check_snapshot(parent);m=json.loads((parent/'member_manifest.json').read_text(encoding='utf8'));fn=f'DEV_epoch_{epoch:03d}_prediction_only.npz';pred=parent/'run/out'/fn
 member=next(v for k,v in m['small_members'].items() if k.endswith('/'+fn));assert sha(pred)==member['sha256']
 with np.load(pred,allow_pickle=False) as z:
  assert str(z['model_state_sha256'])==devrecord['methods'][name]['best_state_SHA'];dev[name]=z['prediction'];ids=z['row_ids']
  if name=='minimal_fixed_F':common_ids=ids.copy()
  else:assert np.array_equal(ids,common_ids)
 refs[name]=dict(original_prediction=str(pred),sha256=sha(pred),snapshot_SHA=cr['snapshot_sha256'])
lp=D/'selected20_requested_VAL_TEST_20261007T111903Z/A_original_capsule';lm=json.loads((lp/'manifest.json').read_text(encoding='utf8'))
labelpath=lp/'original/val_evaluation_targets.npy';assert sha(labelpath)==lm['member_sha256']['original/val_evaluation_targets.npy']
with np.load(lp/'original/val_bfp.npz',allow_pickle=False) as z:assert np.array_equal(z['row_ids'],common_ids)
np.savez(bundle/'val_fixed_pair.npz',row_ids=common_ids,labels=np.load(labelpath,allow_pickle=False).astype(np.float64),**dev)
for source in [Path('work/fixed_formal_video_paired_bootstrap.py'),Path('work/sentiment_metrics_careflow_v1.py')]:shutil.copy2(source,bundle/source.name)
testrecord=json.loads((testparent/'run/out/actual_stage_receipt.json').read_text(encoding='utf8'))
plan=dict(status='SAVED_FIXED_FORMAL_VAL_TEST_PAIRED_VIDEO_BOOTSTRAP_PROTOCOL_FROZEN',actualclock_freeze_UTC=a.clock,draws=10000,seed=20261008,
 scope='Fixed formal F epoch89 and CaReFlow epoch93 on original official VAL229/TEST685. Paired video clusters, all segment rows kept, official segment-weighted metrics. All five share identical bootstrap draws per role. Conditional on fixed trained/selected models; no refitting/epoch reselection/method selection; VAL winner curse and training-seed uncertainty not corrected. Joint win fraction descriptive, not posterior probability or simultaneous confidence guarantee.',
 source_sha256={q.name:sha(q) for q in bundle.iterdir() if q.suffix=='.py'},input_sha256={q.name:sha(q) for q in bundle.iterdir() if q.suffix=='.npz'},
 expected_fixed_points={'val':{k:devrecord['methods'][k]['fixed_DEV_five'] for k in dev},'test':testrecord['metrics']},
 original_predictions_provenance=refs,official_VAL_target_reference=dict(path=str(labelpath),sha256=sha(labelpath)),official_TEST_original_snapshot_SHA=tr['snapshot_sha256'])
write(bundle/'protocol.json',plan);psha=sha(bundle/'protocol.json');out=root/'actual';cmd=[sys.executable,str((bundle/'fixed_formal_video_paired_bootstrap.py').resolve()),'--plan',str((bundle/'protocol.json').resolve()),'--plan-sha',psha,'--output',str(out.resolve())]
with (root/'stdout.log').open('wb') as stdout,(root/'stderr.log').open('wb') as stderr:
 child=subprocess.Popen(cmd,stdout=stdout,stderr=stderr,env=dict(os.environ,PYTHONIOENCODING='utf8',PYTHONDONTWRITEBYTECODE='1'));code=child.wait()
write(root/'natural_exit.json',dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=child.pid,natural_exit=code,fullargv=cmd,protocol_SHA=psha))
shutil.copy2(__file__,root/Path(__file__).name)
if code:print((root/'stderr.log').read_text(encoding='utf8'));sys.exit(code)
r=json.loads((out/'actual_stage_receipt.json').read_text(encoding='utf8'));assert r['pid']==child.pid and r['fullargv']==cmd
members={str(q.relative_to(root)).replace('\\','/'):sha(q) for q in root.rglob('*') if q.is_file()};write(root/'local_original_manifest.json',dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),member_sha256=members))
with zipfile.ZipFile(root/'complete_local_CPU_capture.zip','x',zipfile.ZIP_DEFLATED) as z:
 for q in root.rglob('*'):
  if q.is_file() and q.suffix!='.zip':z.write(q,str(q.relative_to(root)).replace('\\','/'))
with zipfile.ZipFile(root/'complete_local_CPU_capture.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for k,h in members.items():assert hashlib.sha256(z.read(k)).hexdigest()==h
pointer=dict(D=str(root),status=r['status'],protocol_SHA=psha,capture_SHA=sha(root/'complete_local_CPU_capture.zip'))
write(root/'capture_receipt.json',pointer);write('work/formal_video_paired_bootstrap_current.json',pointer);print(json.dumps(pointer));print(json.dumps(r['roles']))
