"""Original whole best/model/Adam/RNG and source audit, no model forward."""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.run);out=root/'out'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def tensors_sha(s):
 h=hashlib.sha256()
 for n,v in sorted(s.items()):
  assert v.device.type=='cpu' and torch.isfinite(v).all();v=v.contiguous();h.update(n.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.numpy().tobytes())
 return h.hexdigest()
r=json.loads((out/'actual_training_receipt.json').read_text());ex=json.loads((root/'natural_exit.json').read_text())
assert ex['exit_code']==0 and ex['natural_wait_verified'] and ex['child_pid']==r['pid'] and ex['child_full_argv'][1:]==r['argv']
assert r['epochs']==10 and r['optimizer_steps']==220 and not r['formal100_complete'] and not r['OUTER_input_or_labels_used']
assert not r['fresh_instance_replay_pending'] and r['fresh_instance_strict_full_disk_replay_error']<=1e-6 and r['dummy_label_replacement_error']==0
assert r['next_update_continuation_check']['exact_complete_model_Adam_scheduler_RNG_compared'] and r['next_update_continuation_check']['formal_steps_before_and_after']==220
states={}
for n,key in [('complete_resume_full.pt','complete_resume_full'),('selected_best_full.pt','selected_best_full')]:
 path=out/n;assert sha(path)==r[key]['sha256'] and path.stat().st_size==r[key]['bytes']
 cp=torch.load(path,map_location='cpu',weights_only=True);m=cp['metadata'];s=cp['model'];assert len(s)==374 and tensors_sha(s)==m['state_sha256']
 assert m['fold']==0 and m['seed']==91819 and m['plan_sha256']==r['plan_sha256']
 stats={n:v for n,v in s.items() if n.startswith(('dberta.v6_audio_','dberta.v6_visual_'))};assert tensors_sha(stats)==m['fit_statistics_sha256']==r['fit_statistics_sha256']
 if n=='complete_resume_full.pt':
  assert m['scope']=='STAGED_SHARED10_RESUME_NOT_COMPLETE100' and m['epoch']==10 and m['optimizer_steps']==220 and len(cp['history'])==10
  assert len(cp['optimizer']['state'])==364 and cp['scheduler']['last_epoch']==220
  assert sum(len(g['params']) for g in cp['optimizer']['param_groups'])==364
  for st in cp['optimizer']['state'].values():
   assert float(st['step'])==220 and set(st)=={'step','exp_avg','exp_avg_sq'}
   for v in st.values():assert v.device.type=='cpu' and torch.isfinite(v).all()
  assert cp['rng']['torch'].dtype==torch.uint8 and len(cp['rng']['cuda'])==1 and cp['rng']['cuda'][0].dtype==torch.uint8
  assert len(cp['best_state'])==374 and cp['best_epoch']==r['best_epoch'] and cp['best_metric']==r['INNER_video_equal_MSE_selection_only']
  assert np.asarray(cp['best_prediction']).shape==(153,)
  resume_best_sha=tensors_sha(cp['best_state'])
 else:
  assert m['scope']=='SHARED10_BEST_NOT_COMPLETE100' and m['epoch']==r['best_epoch'] and m['state_sha256']==resume_best_sha
 states[n]={'file_sha256':sha(path),'tensors':len(s),'elements':sum(v.numel() for v in s.values()),'state_sha256':m['state_sha256']}
 del cp,s
for name in ('selected_best_inner_replay.npz','fresh_selected_best_inner_replay.npz'):
 with np.load(out/name,allow_pickle=False) as z:
  assert z['row_ids'].shape==(153,) and z['prediction'].shape==(153,) and np.isfinite(z['prediction']).all()
  assert str(z['model_state_sha256'])==resume_best_sha
proof={'status':'ORIGINAL_SHARED10_FULL_MODEL_ADAM_RNG_ARRAY_CPU_AUDIT_PASSED_NOT_MODEL_FORWARD','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),'original_training_receipt_sha256':sha(out/'actual_training_receipt.json'),'original_natural_exit_sha256':sha(root/'natural_exit.json'),'states':states,'complete_Adam_parameter_states':364,'steps':220,'GPU_used':False,'CPU_model_forward':False}
Path(a.out).write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof),flush=True)
