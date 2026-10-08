import json,hashlib,zipfile,datetime,shutil
from pathlib import Path
import numpy as np
D=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference100_complete_actual_20261006T183917Z')
A=D/'a'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
read=lambda p:json.loads(Path(p).read_text())
c=read(A/'capture_receipt.json');ex=read(A/'capture_actual_exit.json')
assert ex['exit_code']==0 and ex['natural_wait_verified'] and c['formal100_complete'] and c['natural_training_exit_code']==0
assert sha(A/'snapshot.zip')==c['snapshot_sha256'] and (A/'snapshot.zip').stat().st_size==c['snapshot_bytes']
with zipfile.ZipFile(A/'snapshot.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==c['members']
 m=json.loads(z.read('member_manifest.json'))
 for n,r in m['small_members'].items():
  assert hashlib.sha256(z.read(n)).hexdigest()==r['sha256']
  p=A/n
  assert '..' not in Path(n).parts and not Path(n).is_absolute()
  p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():assert sha(p)==r['sha256']
  else:p.write_bytes(z.read(n))
r=read(A/'run/out/actual_training_receipt.json');e=read(A/'run/natural_exit.json')
assert r['epochs']==100 and r['optimizer_steps']==2200 and r['formal100_complete'] and not r['OUTER_input_or_labels_used']
assert e['exit_code']==0 and e['child_pid']==r['pid'] and e['child_full_argv'][1:]==r['argv']
assert e['original_training_receipt_sha256']==sha(A/'run/out/actual_training_receipt.json')
assert r['plan_sha256']=='67cc46eea7d2f7dbf0e1f681a28fbbdbb7281eb34c9b703bddca4bf0de7a3db4'
assert e['source_sha256']==sha(A/'source/minimal_fixed_staged_training_v3.py')=='f8295d4002a15c7ffdb6d8c1ce941fe3e8cc3f2862ee40437da3bc2af36741ad'
whole={}
for name,key in [('complete_resume_full.pt','complete_resume_full'),('selected_best_full.pt','selected_best_full')]:
 p=A/'out'/name;rr=r[key];assert sha(p)==rr['sha256'] and p.stat().st_size==rr['bytes']
 lr=m['large_references_not_downloads']['run/out/'+name];assert rr['sha256']==lr['sha256'] and rr['bytes']==lr['bytes'] and lr['stable_before_after']
 with zipfile.ZipFile(p) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 whole[name]={'sha256':sha(p),'bytes':p.stat().st_size,'whole_original_file_downloaded':True,'torch_archive_CRC_and_unique_passed':True}
 # Move this new download into its source-relative location; never old data.
 dest=A/'run/out'/name;assert not dest.exists();p.rename(dest)
orders=np.load(A/'source/fit_orders_seed91819_100.npy',allow_pickle=False)
steps=[json.loads(s) for s in (A/'run/out/actual_fit_steps.jsonl').read_text().splitlines()]
assert orders.shape==(100,695) and len(steps)==1980
for i,s in enumerate(steps):
 ep=10+i//22;b=i%22
 assert s['epoch']==ep+1 and s['optimizer_steps']==221+i and s['rows']==orders[ep,b*32:min(b*32+32,695)].tolist()
 assert s['batch_size']==(23 if b==21 else 32) and s['all_finite_nonNone_tensors']==364
assert len(r['history'])==100
best=min(r['history'],key=lambda h:h['INNER_video_equal_MSE_selection_only']);assert best['epoch']==r['best_epoch']
for h in r['history'][10:]:
 p=A/'run/out'/('inner_epoch_%03d.npz'%h['epoch']);assert sha(p)==h['inner_prediction_file_sha256']
proof={'status':'ACTUAL_COMPLETE100_D_WHOLE_FILES_SOURCE_ORDER_CRC_ARRAY_PASSED_CPU_PENDING','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'whole':whole,'A_capture_receipt_sha256':sha(A/'capture_receipt.json'),'A_capture_exit_sha256':sha(A/'capture_actual_exit.json'),'A_snapshot_sha256':sha(A/'snapshot.zip'),'training_receipt_sha256':sha(A/'run/out/actual_training_receipt.json'),'training_natural_exit_sha256':sha(A/'run/natural_exit.json'),'steps':2200,'epochs':100,'best_epoch':r['best_epoch'],'strict_replay_error':r['fresh_instance_strict_full_disk_replay_error'],'CPU_model_forward':False,'other_node_CPU_pending':True,'new_metrics_or_OUTER_labels':False,'fresh_space':{k:shutil.disk_usage(k+':/').free for k in ['C','D']}}
(D/'complete100_D_original_audit.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(proof))
