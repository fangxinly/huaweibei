from pathlib import Path
import argparse,datetime,hashlib,json,zipfile,io
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest();reports={};histories={};shared={};common={}
expected={'a':('fixed','GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7'),'b':('finite_scalar','GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'),'c':('finite_vector','GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa')}
for node,(mode,uuid) in expected.items():
 d=a.directory/node;r=json.loads((d/'receipt.json').read_text());b=(d/'snapshot.zip').read_bytes();assert len(b)==r['bytes'] and sha(b)==r['sha256']
 with zipfile.ZipFile(d/'snapshot.zip') as z:
  assert z.testzip() is None;m=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
  for n,x in m.items():assert len(z.read(n))==x['bytes'] and sha(z.read(n))==x['sha256']
  get=lambda n:json.loads(z.read(n));inv=get('finite_inventory.json');pr=get('finite_run/protocol.json');plan=get('finite_formal_source/finite_formal_plan_v1.json');h=get('finite_run/history.json');histories[node]=h
  assert inv['gpu'].split(',')[0]==pr['gpu_uuid']==uuid and pr['requested_mode']==mode
  assert pr['seed']==91817 and pr['epochs']==100 and pr['train_rows']==1281 and pr['dev_rows']==229 and not pr['test_requested']
  assert sha(z.read('finite_formal_source/finite_formal_plan_v1.json'))==pr['formal_plan_sha256']
  for n,v in plan['source_sha256'].items():assert sha(z.read('finite_formal_source/'+n))==pr['source_sha256'][n]==v
  assert sha(z.read('finite_run/orders.npy'))==pr['orders_sha256']==plan['orders_sha256']
  orders=np.load(io.BytesIO(z.read('finite_run/orders.npy')));assert orders.shape==(100,1281) and all(np.array_equal(np.sort(x),np.arange(1281)) for x in orders)
  for n,key in [('train_scales.json','scales_sha256'),('head_fit_rows.npy','head_fit_rows_sha256')]:assert sha(z.read('finite_formal_source/'+n))==pr[key]==plan[key]
  assert [x['epoch'] for x in h]==list(range(1,len(h)+1)) and all(x['effective_mode']==('fixed' if x['epoch']<=10 else mode) for x in h)
  for k in ['initial_tensor_sha256','orders_sha256','scales_sha256','head_fit_rows_sha256','formal_plan_sha256']:common.setdefault(k,[]).append(pr[k])
  shared[node]=sha(z.read('finite_run/shared_phase_addon.pt'))
  state='LIVE';best=None;metrics=None
  if 'finite_run/selection.json' in m:
   s=get('finite_run/selection.json');assert len(h)==s['epochs']==100;best=int(np.argmin([x['dev_author_batch_mean_mse'] for x in h]))+1;assert best==s['best_epoch']
   assert s['effective_mode']==('fixed' if best<=10 else mode);assert s['addon_sha256']==m['finite_run/best_addon.pt']['sha256'] and s['prediction_sha256']==m['finite_run/predictions.npz']['sha256'] and s['shared_phase_sha256']==shared[node]
   assert s['frozen_tensor_sha_before']==s['frozen_tensor_sha_after'] and s['selected_prediction_replay_max_error']==0
   arrays=np.load(io.BytesIO(z.read('finite_run/predictions.npz')));pred=arrays['valid_pred'];y=arrays['valid_y'];assert pred.shape==y.shape==(229,) and np.isfinite(pred).all()
   metrics={'mae':float(np.mean(np.abs(pred-y))),'mse229':float(np.mean((pred-y)**2))};assert abs(metrics['mae']-h[best-1]['dev_mae_229'])<1e-6
   if 'exit' in inv:assert inv['exit']['exit_code']==0
   state='100_COMPLETE_SELECTED_ADDON_FULL_MODEL_SEPARATE'
  elif 'exit' in inv:
   assert inv['exit']['exit_code']!=0 and inv['actual_process_argv'] in [None,[]] and len(h)<100
   state='FAILED_PARTIAL_NOT_COMPLETE'
  else:
   assert inv['actual_process_argv']==inv['launch']['argv'] and inv['actual_process_argv'][1:]==pr['argv'] and uuid in inv['compute']
   assert 'State:\tZ' not in inv['actual_process_status']
  reports[node]=dict(state=state,history_epochs=len(h),best_epoch=best,metrics=metrics,capture_utc=inv['utc'],gpu=inv['gpu'],compute=inv['compute'],argv=inv['actual_process_argv'],exit=inv.get('exit'),data_disk_free=inv['data_disk_free'],snapshot_sha256=r['sha256'])
for k,v in common.items():assert len(set(v))==1,k
assert len(set(shared.values()))==1
for e in range(10):
 for k in ['train_task_mse','train_residual_loss','dev_author_batch_mean_mse','dev_mse_229','dev_mae_229']:assert len({h[e][k] for h in histories.values()})==1,(e,k)
out=dict(status='FINITE_SNAPSHOTS_NODE_SPECIFIC_LIVE_COMPLETE_FAILURE_INDEPENDENT_AUDIT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reports=reports,shared_phase_sha256=shared['a'],common={k:v[0] for k,v in common.items()},scope='Failure remains failed. A selected addon is not a full checkpoint. Fresh timestamps are inside snapshot; stamp is only an ID. No TEST access.')
a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(reports))
