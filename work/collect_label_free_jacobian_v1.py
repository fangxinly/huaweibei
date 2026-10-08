"""Reference context sensitivity is label-free; no labels requested."""
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess
import numpy as np
import torch
from soft_vector_runtime_v1 import FrozenCoordinateLearner
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
assert not a.out.exists();a.out.mkdir();torch.set_num_threads(2)
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==a.expected_uuid
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
model=FrozenCoordinateLearner(a.root,'fixed',.1).cuda();model.requires_grad_(False);model.eval();reports={}
for split,count in [('train',1281),('dev',229)]:
 cache=np.load(a.root/'teacher_cache_v1'/f'{split}_cache.npz');jac=[];pred=[]
 # Deliberately request only label-free covariates from the unchanged container.
 for start in range(0,count,128):
  end=min(start+128,count);state=torch.as_tensor(cache['state'][start:end],device='cuda');mask=torch.as_tensor(cache['mask'][start:end],device='cuda');old=torch.as_tensor(cache['old_context'][start:end],device='cuda')
  context=(.5*old).detach().requires_grad_();prediction=model.terminal(state,mask,context)
  derivative=torch.autograd.grad(prediction.sum(),context)[0].detach();assert torch.isfinite(derivative).all()
  jac.append(derivative.cpu().numpy());pred.append(prediction.detach().cpu().numpy())
 arrays={'reference_jacobian':np.concatenate(jac),'reference_prediction':np.concatenate(pred),'row_id':np.arange(count)}
 assert np.max(np.abs(arrays['reference_prediction']-cache['reference_prediction']))<2e-5
 dest=a.out/f'{split}_jacobian.npz';np.savez(dest,**arrays);reports[split]={'rows':count,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size,'fields':sorted(arrays)}
assert all(p.grad is None for p in model.parameters())
r={'status':'LABEL_FREE_REFERENCE_CONTEXT_JACOBIAN_COLLECTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_uuid':uuid,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'splits':reports,'labels_requested':False,'teacher_targets_requested':False,'test_requested':False,'scope':'Frozen reference Jacobian dp0/dcontext; computed without y. Known sensitivity direction is separate from unknown prediction residual sign. No new training.'}
(a.out/'jacobian_receipt.json').write_text(json.dumps(r,indent=2));print('JACOBIAN_COLLECTION_COMPLETE',flush=True)
