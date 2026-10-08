"""Draft only: refuses launch without a separately frozen post-GPU formal plan."""
import os
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,json,datetime,math,time,subprocess,shutil,zipfile
import numpy as np
import torch
from group_teacher_runtime_v2 import setup,forward,row_batch,collect,sha,tensor_sha,norm_free_sha,source_checks,write_json

p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--fold',type=int,choices=[0,1,2],required=True);p.add_argument('--uuid',required=True)
a=p.parse_args();now=lambda:datetime.datetime.now(datetime.timezone.utc)
plan_path=a.root/'group_teacher_plan_v3.json'
formal_path=a.root/'group_teacher_formal_plan_v4.json'
if not formal_path.is_file():raise RuntimeError('No formal plan: GPU mechanism, time and preservation dependencies must pass first')
formal=json.loads(formal_path.read_text(encoding='utf-8'))
assert formal['status']=='FORMAL_TRAINING_PLAN_FROZEN_AFTER_THREE_GPU_MECHANISM_AUDITS'
assert formal['parent_plan_sha256']==sha(plan_path) and formal['trainer_sha256']==sha(__file__)
audit_path=a.root/'three_fold_gpu_mechanism_audit.json'
assert sha(audit_path)==formal['mechanism_audit_sha256']
audit=json.loads(audit_path.read_text(encoding='utf-8'))
assert audit['status']=='THREE_ORIGINAL_GPU_MECHANISM_RECEIPTS_INDEPENDENT_CPU_AUDIT_PASSED'
assert audit['plan_sha256']==sha(plan_path)
assert audit['fresh_remote_and_local_disk_check_still_required_before_launch'] is True
space_path=a.root/'formal_space_authorization.json'
assert sha(space_path)==formal['space_authorization_sha256']
space=json.loads(space_path.read_text(encoding='utf-8'))
assert space['status']=='FRESH_PERMANENT_LOCAL_AND_THREE_REMOTE_SPACES_VERIFIED'
assert 0<=(now()-datetime.datetime.fromisoformat(space['utc'])).total_seconds()<600
assert space['all_three_weights_and_preservation_budget_bytes']<=space['permanent_local_free_bytes']
assert space['deletion_required'] is False
lease=datetime.datetime.fromisoformat(formal['estimated_lease_end_utc'])
assert formal['preservation_margin_seconds']>=7200
estimate=audit['folds'][a.fold]['seconds_conservative_pilot_estimate']
assert (lease-now()).total_seconds()>estimate+formal['preservation_margin_seconds']
plan=json.loads(plan_path.read_text(encoding='utf-8'));source_checks(a.base,a.root,plan)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader'],text=True).strip()
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip()
assert gpu.split(',')[0].strip()==a.uuid and not compute
local_receipt=a.root/'precheck_v2/receipt.json'
assert sha(local_receipt)==audit['folds'][a.fold]['original_receipt_sha256']
pre=json.loads(local_receipt.read_text(encoding='utf-8'))
assert pre['gpu'].split(',')[0].strip()==a.uuid
assert shutil.disk_usage(a.root).free>3*pre['initial_checkpoint_bytes']+1024**3
out=a.root/'run_v1';assert not out.exists();out.mkdir()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
model,opt,sched,data,fit,inner,stats,matched,arguments=setup(a.base,a.root,a.fold)
assert tensor_sha(model.state_dict())==pre['initial_full_tensor_sha256']
assert norm_free_sha(model)==audit['common_non_normalization_initial_tensor_sha256']
orders=np.load(a.root/f'orders_{a.fold}.npy')
assert orders.shape==(100,len(fit))
for name,reference in plan['folds'][a.fold]['files'].items():assert sha(a.root/name)==reference['sha256']
assert all(np.array_equal(np.sort(row),data.ids['fit']) for row in orders)
protocol={'utc':now().isoformat(),'argv':__import__('sys').argv,'gpu':gpu,'initial_compute':compute,'fold':a.fold,'seed':91818,'epochs':100,'plan_sha256':sha(plan_path),'formal_plan_sha256':sha(formal_path),'trainer_sha256':sha(__file__),'mechanism_audit_sha256':sha(audit_path),'space_authorization_sha256':sha(space_path),'initial_non_normalization_tensor_sha256':norm_free_sha(model),'trainable_parameters':sum(x.numel() for x in model.parameters() if x.requires_grad),'public_pretrained_matched_tensors':matched,'initial_full_tensor_sha256':tensor_sha(model.state_dict()),'orders_sha256':sha(a.root/f'orders_{a.fold}.npy'),'runtime_sha256':sha(a.root/'group_teacher_runtime_v2.py'),'fit_rows':len(fit),'inner_rows':len(inner),'outer_rows':len(data.ids['outer']),'retained_parameter_tensors':len(list(model.parameters())),'unused_pooler_removed':True,'all_retained_gradients_required_every_update':True,'selection':'Earliest strict minimum sample-weighted inner TRAIN MSE after all 100 epochs','outer_labels_read':False,'dev_requested':False,'test_requested':False}
write_json(out/'protocol.json',protocol)
history=[];best=float('inf');best_epoch=None
checkpoint=out/'selected_full_checkpoint.pt'
for epoch in range(1,101):
 tick=time.monotonic();model.train();squared_sum=0.;count=0
 for offset in range(0,len(fit),32):
  ids=orders[epoch-1,offset:offset+32];batch=row_batch(fit,data.ids['fit'],ids)
  opt.zero_grad(set_to_none=True);prediction=forward(model,batch);loss=(prediction-batch[3].view(-1)).square().mean()
  if not torch.isfinite(loss):raise FloatingPointError('Nonfinite fit loss')
  loss.backward()
  if not all(x.requires_grad and x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters()):raise FloatingPointError('Nonfinite fit gradient')
  opt.step();sched.step();squared_sum+=float(loss.detach())*len(ids);count+=len(ids)
 assert count==len(fit)
 prediction,labels=collect(model,inner)
 mse=float(np.mean((prediction.astype(np.float64)-labels.astype(np.float64))**2))
 assert np.isfinite(mse)
 if mse<best:
  best=mse;best_epoch=epoch
  temporary=out/'selected_full_checkpoint.pending.pt'
  torch.save(model.state_dict(),temporary);temporary.replace(checkpoint)
  np.savez(out/'selected_inner_predictions.npz',prediction=prediction,label=labels,row_ids=data.ids['inner'])
 history.append({'epoch':epoch,'fit_sample_mse':squared_sum/count,'inner_sample_mse':mse,'best_epoch':best_epoch,'utc':now().isoformat(),'elapsed_seconds':time.monotonic()-tick})
 write_json(out/'history.json',history)
 print('GROUP_TEACHER_EPOCH',epoch,mse,best_epoch,flush=True)
assert len(history)==100 and best_epoch==min(range(1,101),key=lambda e:history[e-1]['inner_sample_mse'])
assert all(torch.isfinite(value).all() for value in model.state_dict().values())
model.load_state_dict(torch.load(checkpoint,map_location='cpu'),strict=True)
assert all(torch.isfinite(value).all() for value in model.state_dict().values())
selected_sha=tensor_sha(model.state_dict())
with zipfile.ZipFile(checkpoint) as archive:assert archive.testzip() is None
prediction,labels=collect(model,inner)
with np.load(out/'selected_inner_predictions.npz',allow_pickle=False) as stored:
 assert np.array_equal(stored['row_ids'],data.ids['inner']) and np.array_equal(labels,stored['label'])
 replay_error=float(np.max(np.abs(prediction-stored['prediction'])))
 assert replay_error<=1e-6
# No outer label tensor exists; the adapter passes zero dummy labels only after
# fitting and inner selection are finished. The real outer labels stay unread.
data.outer_prediction_authorized=True
outer=data.outer_inputs();outer_prediction,dummy_labels=collect(model,outer)
assert np.isfinite(outer_prediction).all() and np.array_equal(dummy_labels,np.zeros_like(dummy_labels))
assert tensor_sha(model.state_dict())==selected_sha
np.savez(out/'outer_scalar_predictions.npz',row_ids=data.ids['outer'],mu=outer_prediction)
assert [entry['role'] for entry in data.journal]==['fit','inner','outer_prediction_only_after_selected100']
completion={'status':'GROUP_TEACHER_100_INNER_SELECTED_FULL_RELOADED_OUTER_SCALAR_PREDICTIONS_COMPLETE','utc':now().isoformat(),'fold':a.fold,'completed_epochs':100,'best_epoch':best_epoch,'best_inner_sample_mse':best,'inner_strict_disk_replay_max_error':replay_error,'full_checkpoint_sha256':sha(checkpoint),'full_checkpoint_bytes':checkpoint.stat().st_size,'selected_model_tensor_sha256':selected_sha,'outer_predictions_sha256':sha(out/'outer_scalar_predictions.npz'),'outer_rows':len(outer),'outer_labels_read':False,'dev_requested':False,'test_requested':False,'data_access_journal':data.journal,'protocol_sha256':sha(out/'protocol.json'),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'local_permanent_or_independent_CPU_preservation_complete':False}
write_json(out/'completion.json',completion)
print('GROUP_TEACHER_100_COMPLETE',a.fold,best_epoch,flush=True)
