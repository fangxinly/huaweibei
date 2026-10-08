"""Head-label holdout integrity, not whole-pipeline validation."""
from pathlib import Path
import argparse,datetime,hashlib,json,sys
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--base-root',type=Path,required=True);p.add_argument('--checks',type=Path,required=True);a=p.parse_args()
sys.path.append(str(a.base_root));from finite_task_risk_runtime_v1 import FrozenCoordinateLearner
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(91817)
scales=json.loads((a.checks/'train_scales.json').read_text());model=FrozenCoordinateLearner(a.base_root,'fixed',scales).cuda()
model.restore_addon(torch.load(a.checks/'after20_addon.pt',map_location='cpu'));cache=np.load(a.base_root/'teacher_cache_v1/train_cache.npz',allow_pickle=False)
rows=np.load(a.checks/'head_fit_rows.npy');fit=np.zeros(1281,dtype=bool);fit[rows]=True
features={k:torch.as_tensor(cache[k],device='cuda') for k in ['old_context','pooled_state','reference_prediction']}
norm,hat=model.feedback.estimate_residual(features['old_context'],features['pooled_state'],features['reference_prediction'])
target=torch.as_tensor(cache['reference_prediction']-cache['y'],device='cuda');mask=torch.as_tensor(fit,device='cuda')
loss=model.feedback.residual_loss(norm,target,mask)
params=list(model.feedback.parameters());g1=torch.autograd.grad(loss,params,retain_graph=True)
changed=target.clone();changed[~mask]+=100
other=model.feedback.residual_loss(norm,changed,mask);g2=torch.autograd.grad(other,params)
assert torch.equal(loss,other) and all(torch.equal(x,y) for x,y in zip(g1,g2))
pred=hat.detach().cpu().numpy();truth=target.cpu().numpy()
assert np.isfinite(pred).all()
dest=a.checks/'TRAIN_head_holdout_arrays.npz';assert not dest.exists();np.savez(dest,row_id=np.arange(1281),fit_mask=fit,predicted_residual=pred,residual_target=truth)
metrics={}
for label,m in [('head_fit',fit),('head_label_holdout',~fit)]:
    metrics[label]=dict(rows=int(m.sum()),mse=float(np.mean((pred[m]-truth[m])**2)),zero_baseline_mse=float(np.mean(truth[m]**2)),sign_agreement=float(np.mean((pred[m]>0)==(truth[m]>0))))
r=dict(status='TRAIN_HEAD_HELDOUT_LABEL_REPLACEMENT_LOSS_AND_ALL_GRADIENTS_EXACTLY_UNCHANGED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),metrics=metrics,aux_loss_error=0.,aux_gradient_error=0.,arrays_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),dev_requested=False,test_requested=False,limits='20updatepilot descriptive head-label holdout only; main donors see allTRAIN, frozen encoder fullTRAINfit and DEVselected. Not independent pipeline validation or evidence of efficacy.')
path=a.checks/'TRAIN_head_holdout_receipt.json';assert not path.exists();path.write_text(json.dumps(r,indent=2));print('FINITE_HEAD_HOLDOUT_INTEGRITY_COMPLETE',flush=True)
