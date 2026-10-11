"""Frozen17-condition DEV mechanism diagnostic, never a TRAIN target."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,datetime,hashlib,json
import numpy as np
import torch
from soft_vector_runtime_v1 import FrozenCoordinateLearner,PAIRS
def tensorsha(values):
 h=hashlib.sha256()
 for n,v in sorted(values.items()):h.update(n.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert not a.out.exists();a.out.mkdir();torch.set_num_threads(2)
protocol=json.loads((a.run/'protocol.json').read_text());selection=json.loads((a.run/'selection.json').read_text());assert selection['epochs']==100
mode=selection['effective_mode'];learner=FrozenCoordinateLearner(a.root,protocol['requested_mode'],.1).cuda();learner.restore_addon(torch.load(a.run/'best_addon.pt',map_location='cpu'));learner.eval()
before=tensorsha(learner.state_dict());cache=np.load(a.root/'teacher_cache_v1/dev_cache.npz');assert 'teacher_gradient' not in cache.files
names=['default','alloff']+[f'off{i}' for i in range(6)]+[f'swap{i}' for i in range(6)]+['fixed','scalar','soft_projected']+[f'receiver_off{i}' for i in range(3)]
chunks={};saved=np.load(a.run/'predictions.npz');default_error=0.;label_error=0.
for start in range(0,229,128):
 indices=np.arange(start,min(start+128,229));b={k:torch.as_tensor(cache[k][indices],device='cuda') for k in cache.files if k!='row_id'};pool=b['pooled_state'];messages=[];swaps=[]
 for i,(m,j) in enumerate(PAIRS):
  def donor(pj):return learner.donor[i](torch.cat([pool[:,m],pj,.5*(pool[:,m]-pj),.5*(pool[:,m]+pj)],-1).detach())
  messages.append(donor(pool[:,j]));swaps.append(donor(pool[:,j].roll(1,0)))
 messages=torch.stack(messages,1).detach()
 context=(.5*b['old_context']).detach().requires_grad_();p0=learner.terminal(b['state'],b['mask'],context)
 true_gradient=torch.autograd.grad((p0-b['y']).square().sum(),context)[0].detach()
 with torch.no_grad():
  pred,normal,obs=learner(b,mode);default_error=max(default_error,float((pred-torch.as_tensor(saved['valid_pred'][indices],device='cuda')).abs().max()))
  changed=dict(b,y=b['y'].flip(0)+10);label_error=max(label_error,float((learner(changed,mode)[0]-pred).abs().max()))
  values={'valid_y':b['y'],'reference_prediction':p0.detach(),'true_reference_gradient':true_gradient,'predicted_gradient':obs['predicted_gradient'],'message':messages,'estimated_raw_message_dot':obs['estimated_raw_message_dot'],'estimated_transformed_dot':obs['estimated_transformed_dot'],'scalar_weights':obs['scalar_weights'],'positive_parallel_retention_ratio':obs['positive_parallel_retention_ratio'],'true_raw_message_dot':(true_gradient.repeat_interleave(2,1)*messages).sum(-1)}
  for name in names:
   current=messages.clone();effective=mode
   if name=='default':output=pred
   elif name=='alloff':output=p0.detach()
   else:
    if name.startswith('off'):current[:,int(name[3:])]=0
    elif name.startswith('swap'):i=int(name[4:]);current[:,i]=swaps[i]
    elif name.startswith('receiver_off'):i=int(name[-1]);current[:,2*i:2*i+2]=0
    else:effective=name
    learner.feedback.mode=effective;cx,_,_=learner.feedback(b['old_context'],pool,b['reference_prediction'],current);output=learner.terminal(b['state'],b['mask'],cx)
   values['prediction_'+name]=output.detach()
  for k,v in values.items():chunks.setdefault(k,[]).append(v.detach().cpu().numpy())
arrays={k:np.concatenate(v) for k,v in chunks.items()};assert default_error==0 and label_error==0
assert before==tensorsha(learner.state_dict()) and all(p.grad is None for p in learner.parameters())
np.savez(a.out/'diagnostics.npz',**arrays)
r={'status':'FROZEN20_DEV_CONDITIONS_ARRAYS_SAVED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'conditions':names,'rows':229,'default_cached_replay_max_error':default_error,'label_replacement_max_error':label_error,'tensor_sha_before':before,'tensor_sha_after':before,'addon_sha256':hashlib.sha256((a.run/'best_addon.pt').read_bytes()).hexdigest(),'diagnostics_sha256':hashlib.sha256((a.out/'diagnostics.npz').read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'selection':selection,'teacher_scope':'DEV labels only for frozen diagnostic gradient calibration and loss evaluation, never training, structure or parameter selection. Swap shifts only donor pooled representation within each official128/101batch, same label-free head coordinates. Actual effects are task-head counterfactuals, not semantic truths.','test_requested':False}
(a.out/'diagnostics_receipt.json').write_text(json.dumps(r,indent=2));print('DIAGNOSTICS_COMPLETE',flush=True)
