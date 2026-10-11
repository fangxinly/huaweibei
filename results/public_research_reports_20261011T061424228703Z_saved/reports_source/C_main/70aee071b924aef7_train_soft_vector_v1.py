"""Matched frozen-coordinate feedback experiment; TRAIN/dev only."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time
import numpy as np
import torch
from soft_vector_runtime_v1 import FrozenCoordinateLearner

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensorsha(values):
 h=hashlib.sha256()
 for n,v in sorted(values.items()):
  h.update(n.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(path,data):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2),encoding='utf-8');temp.replace(path)
def inventory():
 q=lambda a:subprocess.check_output(['nvidia-smi',*a],text=True).strip()
 return {'gpu':q(['--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader,nounits']),'compute':q(['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),'disk_free':__import__('shutil').disk_usage('/data').free}
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['fixed','scalar','soft_projected'],required=True);p.add_argument('--expected-gpu-uuid',required=True);p.add_argument('--phase',choices=['preflight','train'],required=True);p.add_argument('--seed',type=int,default=91815);p.add_argument('--kappa',type=float,default=.1);p.add_argument('--epochs',type=int,default=100);c=p.parse_args()
assert c.epochs==100 and c.seed==91815 and c.kappa==.1
assert not c.out.exists();c.out.mkdir()
inv=inventory();assert inv['gpu'].split(',')[0]==c.expected_gpu_uuid and not inv['compute']
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(c.seed);np.random.seed(c.seed)
start=time.monotonic();learner=FrozenCoordinateLearner(c.root,c.mode,c.kappa).cuda()
learner.eval();initial=tensorsha(learner.state_dict());frozen_before=tensorsha({n:v for n,v in learner.state_dict().items() if n.startswith(('flow.','fusion.','predictor.'))})
train={k:v for k,v in np.load(c.root/'teacher_cache_v1/train_cache.npz',allow_pickle=False).items()};dev={k:v for k,v in np.load(c.root/'teacher_cache_v1/dev_cache.npz',allow_pickle=False).items()}
assert len(train['y'])==1281 and len(dev['y'])==229 and 'teacher_gradient' not in dev
orders=np.stack([np.random.RandomState(c.seed+1327+epoch).permutation(1281) for epoch in range(100)])
np.save(c.out/'orders.npy',orders,allow_pickle=False)
trainable=[p for p in learner.parameters() if p.requires_grad];parameter_count=sum(p.numel() for p in trainable)
optimizer=torch.optim.AdamW(trainable,lr=1e-4,weight_decay=.01)
def batch(data,indices):return {k:torch.as_tensor(v[indices],device='cuda') for k,v in data.items() if k!='row_id'}
def step(b,step_index,mode):
 optimizer.zero_grad(set_to_none=True)
 for group in optimizer.param_groups:group['lr']=1e-4*min((step_index+1)/400, max(0.,(4000-step_index)/3600))
 pred,normalized,obs=learner(b,mode);task=(pred-b['y']).square().mean();aux=learner.feedback.gradient_regression_loss(normalized,b['teacher_gradient'])
 loss=task+.01*aux;assert torch.isfinite(loss);loss.backward()
 assert all(p.grad is None for module in (learner.flow,learner.fusion,learner.predictor) for p in module.parameters())
 torch.nn.utils.clip_grad_norm_(trainable,1.);optimizer.step()
 return float(task.detach()),float(aux.detach())
protocol={'scope':'Frozen encoder/first-flow/terminal; train six donor networks and three matched gradient heads; cached deterministic coordinates, not end-to-end encoder training.','seed':c.seed,'requested_mode':c.mode,'epochs':100,'shared_fixed_epochs':10,'kappa':.1,'risk_scale':.05,'auxiliary_gradient_loss_weight':.01,'lr':1e-4,'warmup_steps':400,'total_steps':4000,'weight_decay':.01,'clip_norm':1.,'batch_size':32,'drop_last':True,'train_rows':1281,'dev_rows':229,'test_requested':False,'initial_tensor_sha256':initial,'orders_sha256':sha(c.out/'orders.npy'),'trainable_parameter_count':parameter_count,'teacher_collection_sha256':sha(c.root/'teacher_cache_v1/collection.json'),'source_sha256':{n:sha(c.root/n) for n in ['train_soft_vector_v1.py','soft_vector_runtime_v1.py','task_gradient_vector_candidate_v3.py']},'argv':sys.argv,'gpu_uuid':c.expected_gpu_uuid,'started_utc':stamp(),'initial_inventory':inv}
write(c.out/'protocol.json',protocol)
if c.phase=='preflight':
 b=batch(train,orders[0,:32]);base=learner.addon()
 with torch.no_grad():
  p0=learner.terminal(b['state'],b['mask'],.5*b['old_context'])
  baseline_error=float((p0-b['reference_prediction']).abs().max());assert baseline_error<2e-5
  predictions={m:learner(b,m)[0].clone() for m in ['fixed','scalar','soft_projected']}
  assert torch.equal(predictions['fixed'],predictions['scalar']) and torch.equal(predictions['fixed'],predictions['soft_projected'])
  modified=dict(b,y=b['y'].flip(0)+10,teacher_gradient=b['teacher_gradient']+20)
  assert torch.equal(learner(b,'fixed')[0],learner(modified,'fixed')[0])
 optimizer.zero_grad(set_to_none=True);pred,n,obs=learner(b,'fixed');(pred-b['y']).square().mean().backward()
 assert all(x.grad is None for x in learner.feedback.parameters())
 donor_gradient=sum(float(x.grad.abs().sum()) for x in learner.donor.parameters() if x.grad is not None);assert donor_gradient>0
 optimizer.zero_grad(set_to_none=True);pred,n,obs=learner(b,'fixed');learner.feedback.gradient_regression_loss(n,b['teacher_gradient']).backward()
 assert all(x.grad is None for x in learner.donor.parameters())
 head_gradient=sum(float(x.grad.abs().sum()) for x in learner.feedback.parameters() if x.grad is not None);assert head_gradient>0
 optimizer.zero_grad(set_to_none=True)
 for index in range(20):step(batch(train,orders[0,index*32:(index+1)*32]),index,'fixed')
 assert frozen_before==tensorsha({n:v for n,v in learner.state_dict().items() if n.startswith(('flow.','fusion.','predictor.'))})
 torch.save(learner.addon(),c.out/'after20_addon.pt')
 report=dict(protocol,status='FULL_FROZEN_TERMINAL_20_TRAIN_UPDATES_CHECKED',baseline_cache_replay_max_error=baseline_error,label_replacement_max_error=0.,main_task_head_gradients_none=True,auxiliary_donor_gradients_none=True,donor_task_gradient_sum=donor_gradient,head_aux_gradient_sum=head_gradient,frozen_tensor_sha_before=frozen_before,frozen_tensor_sha_after=frozen_before,after20_tensor_sha256=tensorsha(learner.state_dict()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start,inventory=inventory(),completed_utc=stamp())
 write(c.out/'preflight.json',report);print('PREFLIGHT_COMPLETE',flush=True)
else:
 plan=json.loads((c.root/'formal_plan_v1.json').read_text());assert plan['seed']==c.seed and plan['kappa']==c.kappa and plan['epochs']==100
 protocol['formal_plan_sha256']=sha(c.root/'formal_plan_v1.json');write(c.out/'protocol.json',protocol)
 history=[];best=float('inf');best_epoch=None
 for epoch in range(1,101):
  mode='fixed' if epoch<=10 else c.mode;losses=[];epoch_start=time.monotonic()
  for index in range(40):losses.append(step(batch(train,orders[epoch-1,index*32:(index+1)*32]),(epoch-1)*40+index,mode))
  if epoch==10:torch.save(learner.addon(),c.out/'shared_phase_addon.pt')
  predictions=[];normal=[];observations={}
  with torch.no_grad():
   for index in range(0,229,128):
    pred,n,obs=learner(batch(dev,np.arange(index,min(index+128,229))),mode)
    predictions.append(pred.cpu().numpy());normal.append(n.cpu().numpy())
    for key,value in obs.items():observations.setdefault(key,[]).append(value.cpu().numpy())
  pred=np.concatenate(predictions);y=dev['y'];batch_mse=float(np.mean([np.mean((pred[:128]-y[:128])**2),np.mean((pred[128:]-y[128:])**2)]))
  row={'epoch':epoch,'effective_mode':mode,'train_task_mse':float(np.mean(losses,axis=0)[0]),'train_gradient_loss':float(np.mean(losses,axis=0)[1]),'dev_author_batch_mean_mse':batch_mse,'dev_mse_229':float(np.mean((pred-y)**2)),'dev_mae_229':float(np.mean(np.abs(pred-y))),'seconds':time.monotonic()-epoch_start,'utc':stamp()}
  assert np.isfinite(list(v for v in row.values() if isinstance(v,float))).all();history.append(row)
  if batch_mse<best:
   best=batch_mse;best_epoch=epoch;torch.save(learner.addon(),c.out/'best_addon.pt')
   np.savez(c.out/'predictions.npz',valid_y=y,valid_pred=pred,normalized_gradient=np.concatenate(normal),**{k:np.concatenate(v) for k,v in observations.items()})
  write(c.out/'history.json',history);print(json.dumps(row),flush=True)
 learner.restore_addon(torch.load(c.out/'best_addon.pt',map_location='cpu'))
 assert frozen_before==tensorsha({n:v for n,v in learner.state_dict().items() if n.startswith(('flow.','fusion.','predictor.'))})
 selected_mode='fixed' if best_epoch<=10 else c.mode
 with torch.no_grad():replay=np.concatenate([learner(batch(dev,np.arange(i,min(i+128,229))),selected_mode)[0].cpu().numpy() for i in range(0,229,128)])
 with np.load(c.out/'predictions.npz') as a:assert np.array_equal(replay,a['valid_pred'])
 selection={'status':'100_EPOCHS_SELECTED_ADDON_COMPLETE_NOT_FULL_CHECKPOINT','best_epoch':best_epoch,'best_dev_author_batch_mean_mse':best,'effective_mode':selected_mode,'epochs':100,'addon_sha256':sha(c.out/'best_addon.pt'),'prediction_sha256':sha(c.out/'predictions.npz'),'shared_phase_sha256':sha(c.out/'shared_phase_addon.pt'),'selected_prediction_replay_max_error':0.,'frozen_tensor_sha_before':frozen_before,'frozen_tensor_sha_after':frozen_before,'completed_utc':stamp(),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-start,'inventory':inventory()}
 write(c.out/'selection.json',selection);print('100_EPOCH_ADDON_COMPLETE',flush=True)
