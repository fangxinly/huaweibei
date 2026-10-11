"""Actual stage10 originals: NumPy/stdlib local audit; no Torch or model forward."""
import datetime, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np

BASE=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference_first10_actual_20261006T151954Z')
A=BASE/'a'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
assert shutil.disk_usage('D:/').free>12*1024**3
cap=read(A/'capture_receipt.json');ce=read(A/'actual_capture_exit.json')
assert ce['exit_code']==0 and ce['natural_wait_verified'] and ce['child_pid']==cap['pid']
assert ce['child_full_argv'][1:]==cap['argv']
assert ce['capture_source_sha256']==cap['capture_source_sha256']
assert sha(A/'snapshot.zip')==cap['snapshot_sha256']
raw=A/'original_small_files';raw.mkdir(exist_ok=True)
with zipfile.ZipFile(A/'snapshot.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==cap['members']
 mf=json.loads(z.read('member_manifest.json'))
 assert set(z.namelist())==set(mf['small_members'])|{'member_manifest.json'}
 for n,m in mf['small_members'].items():
  assert not n.startswith('/') and '..' not in Path(n).parts
  b=z.read(n);assert len(b)==m['bytes'] and hashlib.sha256(b).hexdigest()==m['sha256']
  p=raw/n;p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert sha(p)==m['sha256']
  else:p.write_bytes(b)
 (raw/'member_manifest.json').write_bytes(z.read('member_manifest.json'))
r=read(raw/'run/out/actual_training_receipt.json');ex=read(raw/'run/natural_exit.json')
assert ex['exit_code']==0 and ex['natural_wait_verified'] and ex['child_pid']==r['pid']
assert ex['child_full_argv'][1:]==r['argv']
assert r['epochs']==10 and r['optimizer_steps']==220 and not r['formal100_complete']
assert not r['OUTER_input_or_labels_used'] and not r['head_training_started']
assert not r['fresh_instance_replay_pending'] and r['fresh_instance_strict_full_disk_replay_error']<=1e-6
assert r['dummy_label_replacement_error']==0 and r['strict_same_instance_disk_replay_error']<=1e-6
source=raw/'source';plan=read(source/'staged_reference_plan.json')
assert sha(source/'staged_reference_plan.json')==r['plan_sha256']
for n,h in {**plan['source_sha256'],**plan['role_order_sha256']}.items():assert sha(source/n)==h,n
inner=np.load(source/'inner_0.npy',allow_pickle=False);fit=np.load(source/'fit_0.npy',allow_pickle=False)
orders=np.load(source/'fit_orders_seed91819_100.npy',allow_pickle=False)
assert inner.shape==(153,) and fit.shape==(695,) and orders.shape==(100,695)
for order in orders:assert np.array_equal(np.sort(order),np.sort(fit))
steps=[json.loads(s) for s in (raw/'run/out/actual_fit_steps.jsonl').read_text(encoding='utf-8').splitlines()]
assert len(steps)==220
for i,s in enumerate(steps):
 epoch=i//22+1;b=i%22
 assert s['epoch']==epoch and s['optimizer_steps']==i+1 and s['step_in_epoch']==b+1
 assert s['rows']==orders[epoch-1,b*32:min((b+1)*32,695)].tolist()
 assert s['batch_size']==(23 if b==21 else 32) and s['all_finite_nonNone_tensors']==364
best=float('inf');best_epoch=None
for h in r['history']:
 e=h['epoch'];p=raw/f'run/out/inner_epoch_{e:03d}.npz'
 assert sha(p)==h['inner_prediction_file_sha256'] and h['optimizer_steps']==e*22
 assert h['fit_rows']==695 and h['batch_count']==22 and h['tail_rows']==23
 assert not h['OUTER_or_head_labels_read']
 with np.load(p,allow_pickle=False) as z:
  assert np.array_equal(z['row_ids'],inner) and z['prediction'].shape==(153,) and np.isfinite(z['prediction']).all()
  assert str(z['model_state_sha256'])==h['state_sha256']
 if h['INNER_video_equal_MSE_selection_only']<best:best=h['INNER_video_equal_MSE_selection_only'];best_epoch=e
 assert h['best_epoch']==best_epoch and h['best_metric']==best
assert len(r['history'])==10 and r['best_epoch']==best_epoch and r['INNER_video_equal_MSE_selection_only']==best
proof=read(raw/'run/out/actual_next_update_continuation_proof.json')
assert proof==r['next_update_continuation_check'] and proof['continuous']==proof['restored']
assert proof['formal_steps_before_and_after']==220 and proof['isolated_diagnostic_steps']==2
assert proof['formal_step221_not_committed'] and proof['original_boundary_restored']
assert proof['exact_complete_model_Adam_scheduler_RNG_compared'] and not proof['new_selection_or_OUTER_labels_used']
assert proof['fit_rows']==orders[10,:32].tolist()
for n,key in [('complete_resume_full.pt','complete_resume_full'),('selected_best_full.pt','selected_best_full')]:
 pending=A/(n+'.pending');final=A/n;p=final if final.exists() else pending
 assert p.stat().st_size==r[key]['bytes'] and sha(p)==r[key]['sha256']
 ref=mf['large_references_not_downloads']['run/out/'+n]
 assert ref['sha256']==r[key]['sha256'] and ref['bytes']==r[key]['bytes'] and ref['stable_before_after']
 if p==pending:pending.rename(final)
budget=r['final_budget'];assert budget['peak_allocated_bytes']<=6*1024**3 and budget['peak_reserved_bytes']<=6*1024**3
record={'status':'ACTUAL_SHARED10_D_FULL_FILES_NUMPY_SOURCE_ARRAY_ORDER_EXIT_AUDIT_PASSED_NOT_TORCH',
 'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),
 'original_training_receipt_sha256':sha(raw/'run/out/actual_training_receipt.json'),
 'original_capture_receipt_sha256':sha(A/'capture_receipt.json'),'original_capture_exit_sha256':sha(A/'actual_capture_exit.json'),
 'snapshot_sha256':cap['snapshot_sha256'],'capture_actual_utc':cap['actual_utc'],
 'original_run':mf['run'],'original_source':mf['source'],'epochs':10,'formal_updates':220,
 'best_epoch':best_epoch,'INNER_video_MSE_selection_only':best,'final_budget':budget,
 'full_files':{n:{'sha256':r[k]['sha256'],'bytes':r[k]['bytes']} for n,k in [('complete_resume_full.pt','complete_resume_full'),('selected_best_full.pt','selected_best_full')]},
 'exact_next_update_proof':proof,'fresh_disk_replay_error':r['fresh_instance_strict_full_disk_replay_error'],
 'donor_mechanism':r['donor_mechanism'],'original_other_node_CPU_pending':True,
 'Torch_imported':False,'CPU_model_forward':False,'formal100_complete':False,'OUTER_used':False,
 'transfer_note':'A-to-B direct SFTP connection had no prompt and was cancelled before authentication or transfer. Original B snapshot hash manual transcription assertion failed before extraction; corrected by original capture receipt, not bypassed.'}
(BASE/'local_D_joint_audit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:record[k] for k in ('status','actual_utc','epochs','formal_updates','best_epoch','INNER_video_MSE_selection_only','formal100_complete')},ensure_ascii=False))
