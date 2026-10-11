"""Frozen20 primary DEV interventions and explicit finite-risk calibration."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,datetime,hashlib,json,sys,subprocess,time
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--base-root',type=Path,required=True);p.add_argument('--formal',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
sys.path[:0]=[str(a.formal),str(a.base_root)]
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner,PAIRS
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tsha(values):
 h=hashlib.sha256()
 for n,v in sorted(values.items()):h.update(n.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
assert not a.out.exists();a.out.mkdir();torch.set_num_threads(2);start=time.monotonic()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip();assert gpu.split(',')[0]==a.expected_uuid and not compute
protocol=json.loads((a.run/'protocol.json').read_text());selection=json.loads((a.run/'selection.json').read_text());assert selection['epochs']==100
plan=json.loads((a.formal/'finite_diagnostics_plan_v1.json').read_text());assert plan['source_sha256']==sha(__file__)
mode=selection['effective_mode'];scales=json.loads((a.formal/'train_scales.json').read_text());learner=FrozenCoordinateLearner(a.base_root,mode,scales).cuda();learner.restore_addon(torch.load(a.run/'best_addon.pt',map_location='cpu'));learner.eval();before=tsha(learner.state_dict())
cache=np.load(a.base_root/'teacher_cache_v1/dev_cache.npz');saved=np.load(a.run/'predictions.npz');names=['default','alloff']+[f'off{i}' for i in range(6)]+[f'swap{i}' for i in range(6)]+['fixed','finite_scalar','finite_vector']+[f'receiver_off{i}' for i in range(3)];assert plan['conditions']==names and len(names)==20
chunks={};replay=0.;label_error=0.;p0error=0.;identity_error=0.;context_error=0.
with torch.no_grad():
 for startrow in range(0,229,128):
  ids=np.arange(startrow,min(startrow+128,229));b={n:torch.as_tensor(cache[n][ids],device='cuda') for n in ['state','mask','old_context','pooled_state','reference_prediction','y']};pool=b['pooled_state'];messages=[];swaps=[]
  for i,(receiver,donor) in enumerate(PAIRS):
   def message(q):return learner.donor[i](torch.cat([pool[:,receiver],q,.5*(pool[:,receiver]-q),.5*(pool[:,receiver]+q)],-1))
   messages.append(message(pool[:,donor]));swaps.append(message(pool[:,donor].roll(1,0)))
  messages=torch.stack(messages,1);terminal=lambda cx:learner.terminal(b['state'],b['mask'],cx)
  pred,normal,obs=learner(b,mode);replay=max(replay,float((pred-torch.as_tensor(saved['valid_pred'][ids],device='cuda')).abs().max()));changed=learner(dict(b,y=b['y'].flip(0)+20),mode)[0];label_error=max(label_error,float((pred-changed).abs().max()))
  p0=terminal(.5*b['old_context']);p0error=max(p0error,float((p0-b['reference_prediction']).abs().max()));rho=b['reference_prediction']-b['y'];rhohat=obs['estimated_residual']
  def controlled(current,effective):
   cx,o=learner.feedback.control(b['old_context'],current,b['reference_prediction'],rhohat,terminal,effective)
   tx=o['controlled_message'] if effective=='finite_vector' else (current*2*o['actual_weights'][:,:,None])
   assert torch.allclose(cx,learner.feedback.context(b['old_context'],tx),atol=2e-6,rtol=1e-6)
   return cx,tx,o
  cx,transmitted,cobs=controlled(messages,mode);context_error=max(context_error,float((terminal(cx)-pred).abs().max()))
  values=dict(valid_y=b['y'],reference_prediction=b['reference_prediction'],reference_recomputed=p0,estimated_residual=rhohat,true_residual=rho,normalized_residual=normal,message=messages,transmitted_message=transmitted,actual_weights=cobs['actual_weights'])
  raw_with=terminal(learner.feedback.context(b['old_context'],messages));raw_without=[]
  for i in range(6):
   current=messages.clone();current[:,i]=0;raw_without.append(terminal(learner.feedback.context(b['old_context'],current)))
  raw_without=torch.stack(raw_without,1);withdelta=raw_with-b['reference_prediction'];withoutdelta=raw_without-b['reference_prediction'][:,None]
  utilityhat=2*rhohat[:,None]*(withoutdelta-withdelta[:,None])+withoutdelta.square()-withdelta[:,None].square()
  utilitytrue=(raw_without-b['y'][:,None]).square()-(raw_with-b['y'])[:,None].square()
  trueformula=2*rho[:,None]*(withoutdelta-withdelta[:,None])+withoutdelta.square()-withdelta[:,None].square();identity_error=max(identity_error,float((trueformula-utilitytrue).abs().max()))
  values.update(raw_fixed_prediction=raw_with,raw_without_predictions=raw_without,estimated_raw_finite_utility=utilityhat,true_raw_finite_utility=utilitytrue)
  for name in names:
   if name=='default':output=pred
   elif name=='alloff':output=p0
   elif name.startswith('off'):
    tx=transmitted.clone();tx[:,int(name[3:])]=0;output=terminal(learner.feedback.context(b['old_context'],tx))
   elif name.startswith('receiver_off'):
    tx=transmitted.clone();i=int(name[-1]);tx[:,2*i:2*i+2]=0;output=terminal(learner.feedback.context(b['old_context'],tx))
   elif name.startswith('swap'):
    current=messages.clone();i=int(name[4:]);current[:,i]=swaps[i];output=terminal(controlled(current,mode)[0])
   else:output=terminal(controlled(messages,name)[0])
   assert torch.isfinite(output).all();values['prediction_'+name]=output
  values['true_final_transmission_gain']=torch.stack([(values['prediction_off'+str(i)]-b['y']).square()-(pred-b['y']).square() for i in range(6)],1)
  for n,v in values.items():chunks.setdefault(n,[]).append(v.cpu().numpy())
assert replay==label_error==context_error==0 and p0error<2e-5 and identity_error<2e-6
assert tsha(learner.state_dict())==before and all(p.grad is None for p in learner.parameters())
np.savez(a.out/'diagnostics.npz',**{k:np.concatenate(v) for k,v in chunks.items()})
r=dict(status='FINITE_FROZEN20_DEV_PRIMARY_INTERVENTIONS_AND_RAW_CALIBRATION_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=gpu,conditions=names,rows=229,default_cached_replay_error=replay,label_replacement_error=label_error,context_replay_error=context_error,reference_replay_error=p0error,finite_squared_identity_error=identity_error,tensor_sha_before=before,tensor_sha_after=before,addon_sha256=sha(a.run/'best_addon.pt'),selection=selection,diagnostics_sha256=sha(a.out/'diagnostics.npz'),source_sha256=sha(__file__),plan_sha256=sha(a.formal/'finite_diagnostics_plan_v1.json'),elapsed_seconds=time.monotonic()-start,test_requested=False,scope=plan['scope'])
(a.out/'diagnostics_receipt.json').write_text(json.dumps(r,indent=2));print('FINITE20_DIAGNOSTICS_COMPLETE',flush=True)
