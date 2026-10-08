"""Full final100 CPU state audit. No model construction, forward or task labels."""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--prior-run',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.run);out=root/'out';prior=Path(a.prior_run)
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
def tensors_sha(s):
 h=hashlib.sha256()
 for n,v in sorted(s.items()):
  assert v.device.type=='cpu' and torch.isfinite(v).all()
  v=v.contiguous();h.update(n.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.numpy().tobytes())
 return h.hexdigest()
r=read(out/'actual_training_receipt.json');ex=read(root/'natural_exit.json');old=read(prior/'out/actual_training_receipt.json')
assert ex['exit_code']==0 and ex['natural_wait_verified'] and ex['child_pid']==r['pid'] and ex['child_full_argv'][1:]==r['argv']
assert r['phase']=='continue100' and r['epochs']==100 and r['optimizer_steps']==2200 and r['formal100_complete']
assert not r['OUTER_input_or_labels_used'] and not r['head_training_started'] and not r['fresh_instance_replay_pending']
assert r['strict_same_instance_disk_replay_error']<=1e-6 and r['fresh_instance_strict_full_disk_replay_error']<=1e-6 and r['dummy_label_replacement_error']==0
assert len(r['history'])==100 and r['history'][:10]==old['history']
proof=old['next_update_continuation_check'];assert proof['continuous']==proof['restored'] and proof['original_boundary_restored'] and proof['formal_steps_before_and_after']==220
budget=r['final_budget'];assert max(budget['peak_allocated_bytes'],budget['peak_reserved_bytes'])<=6*1024**3
start=read(out/'actual_training_start.json');assert start['start_epoch']==10 and start['optimizer_steps']==220 and not start['initial_optimizer_empty']
assert start['plan_sha256']==r['plan_sha256']==old['plan_sha256']
fresh=read(root/'fresh_actual_evidence.json');assert fresh['original_stage10_D_B_CPU_verified'] and fresh['stage10_resume_full_sha256']==old['complete_resume_full']['sha256']
states={};resume_best_sha=None
for name,key in [('complete_resume_full.pt','complete_resume_full'),('selected_best_full.pt','selected_best_full')]:
 file=out/name;record=r[key];assert sha(file)==record['sha256'] and file.stat().st_size==record['bytes']
 cp=torch.load(file,map_location='cpu',weights_only=True);m=cp['metadata'];s=cp['model']
 assert len(s)==374 and tensors_sha(s)==m['state_sha256'] and m['fold']==0 and m['seed']==91819 and m['plan_sha256']==r['plan_sha256']
 assert tensors_sha({n:v for n,v in s.items() if n.startswith(('dberta.v6_audio_','dberta.v6_visual_'))})==m['fit_statistics_sha256']==r['fit_statistics_sha256']
 if name=='complete_resume_full.pt':
  assert m['scope']=='FORMAL100_COMPLETE_RESUME' and m['epoch']==100 and m['optimizer_steps']==2200
  assert cp['history']==r['history'] and cp['scheduler']['last_epoch']==2200 and len(cp['optimizer']['state'])==364
  assert sum(len(g['params']) for g in cp['optimizer']['param_groups'])==364
  for st in cp['optimizer']['state'].values():
   assert set(st)=={'step','exp_avg','exp_avg_sq'} and float(st['step'])==2200
   for v in st.values():assert v.device.type=='cpu' and torch.isfinite(v).all()
  assert cp['rng']['torch'].dtype==torch.uint8 and len(cp['rng']['cuda'])==1 and cp['rng']['cuda'][0].dtype==torch.uint8
  assert 'python_random' in cp['rng_extra'] and 'numpy_global_random' in cp['rng_extra']
  assert len(cp['best_state'])==374 and cp['best_epoch']==r['best_epoch'] and cp['best_metric']==r['INNER_video_equal_MSE_selection_only']
  best_prediction=np.asarray(cp['best_prediction'],dtype=np.float32);assert best_prediction.shape==(153,) and np.isfinite(best_prediction).all()
  resume_best_sha=tensors_sha(cp['best_state'])
 else:
  assert m['scope']=='FORMAL100_EARLIEST_INNER_BEST' and m['epoch']==r['best_epoch'] and m['optimizer_steps']==r['best_epoch']*22
  assert m['state_sha256']==resume_best_sha
 states[name]={'file_sha256':sha(file),'bytes':file.stat().st_size,'tensors':374,'elements':sum(v.numel() for v in s.values()),'state_sha256':m['state_sha256']}
 del cp,s
inner=np.load(root.parent/'source/inner_0.npy',allow_pickle=False);orders=np.load(root.parent/'source/fit_orders_seed91819_100.npy',allow_pickle=False)
assert inner.shape==(153,) and orders.shape==(100,695)
steps=[json.loads(s) for s in (out/'actual_fit_steps.jsonl').read_text().splitlines()];assert len(steps)==1980
for i,s in enumerate(steps):
 e=10+i//22;b=i%22
 assert s['epoch']==e+1 and s['optimizer_steps']==221+i and s['rows']==orders[e,b*32:min(b*32+32,695)].tolist()
 assert s['batch_size']==(23 if b==21 else 32) and s['all_finite_nonNone_tensors']==364
best=float('inf');best_epoch=None
for h in r['history']:
 e=h['epoch'];file=(prior if e<=10 else root)/f'out/inner_epoch_{e:03d}.npz'
 assert sha(file)==h['inner_prediction_file_sha256'] and h['optimizer_steps']==e*22 and not h['OUTER_or_head_labels_read']
 with np.load(file,allow_pickle=False) as z:
  assert np.array_equal(z['row_ids'],inner) and z['prediction'].shape==(153,) and np.isfinite(z['prediction']).all()
  assert str(z['model_state_sha256'])==h['state_sha256']
 if h['INNER_video_equal_MSE_selection_only']<best:best=h['INNER_video_equal_MSE_selection_only'];best_epoch=e
 assert h['best_epoch']==best_epoch and h['best_metric']==best
assert best_epoch==r['best_epoch'] and best==r['INNER_video_equal_MSE_selection_only']
for name in ('selected_best_inner_replay.npz','fresh_selected_best_inner_replay.npz'):
 with np.load(out/name,allow_pickle=False) as z:
  assert np.array_equal(z['row_ids'],inner) and np.max(np.abs(z['prediction']-best_prediction))<=1e-6
  assert str(z['model_state_sha256'])==resume_best_sha
record={'status':'ORIGINAL_REFERENCE100_COMPLETE_MODEL_ADAM_RNG_SOURCE_ORDER_ARRAY_CPU_AUDIT_PASSED_NOT_MODEL_FORWARD',
 'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),
 'original_training_receipt_sha256':sha(out/'actual_training_receipt.json'),'original_natural_exit_sha256':sha(root/'natural_exit.json'),
 'prior10_original_receipt_sha256':sha(prior/'out/actual_training_receipt.json'),'states':states,
 'complete_Adam_parameter_states':364,'steps':2200,'epochs':100,'best_epoch':best_epoch,
 'INNER_video_MSE_is_selection_only':True,'continuation_not_restarted':True,'GPU_used':False,'CPU_model_forward':False,'OUTER_used':False}
Path(a.out).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
