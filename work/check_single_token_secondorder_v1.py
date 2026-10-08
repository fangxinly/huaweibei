"""Actual TRAIN witness-specific double FD, label and gradient isolation."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,copy,datetime,hashlib,json,subprocess,sys,time
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
base=Path('/data/coding/soft_vector_research_20261005T1220Z');old=Path('/data/coding/finite_task_risk_v1_deployment_20261005T1450Z');pre=Path('/data/coding/finite_single_token_precheck_20261005T1525Z');sys.path[:0]=[str(pre),str(old),str(base)]
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner
from finite_single_token_reader_v1 import install_for_flow
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not a.out.exists();a.out.mkdir()
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip();assert gpu.split(',')[0]=='GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa' and not compute
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(91817);start=time.monotonic()
scales=json.loads((old/'train_scales.json').read_text());m=FrozenCoordinateLearner(base,'finite_vector',scales).cuda();m.restore_addon(torch.load(old/'run/shared_phase_addon.pt',map_location='cpu'));install_for_flow(m.flow);m.eval()
frozen=lambda:{k:v.detach().cpu().clone() for k,v in m.state_dict().items() if k.startswith(('flow.','fusion.','predictor.'))};before=frozen()
cache=np.load(base/'teacher_cache_v1/train_cache.npz');ids=np.array([620,0]);fields=['state','mask','old_context','pooled_state','reference_prediction','y'];b={n:torch.as_tensor(cache[n][ids],device='cuda') for n in fields};assert int(b['mask'][0].sum())==1
for n in ['state','old_context','pooled_state']:b[n].requires_grad_()
changed=dict(b,y=b['y'].flip(0)+20)
pred,n,obs=m(b,'finite_vector');assert torch.equal(pred,m(changed,'finite_vector')[0]);task=(pred-b['y']).square().mean();task.backward()
assert all(p.grad is None for p in m.feedback.parameters()) and all(p.grad is None for mod in [m.flow,m.fusion,m.predictor] for p in mod.parameters())
assert all(b[k].grad is None for k in ['state','old_context','pooled_state'])
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.donor.parameters());dg=sum(float(p.grad.abs().sum()) for p in m.donor.parameters() if p.grad is not None);assert dg>0
with torch.no_grad():infer=m(b,'finite_vector')[0]
assert torch.equal(pred.detach(),infer)
m.zero_grad(set_to_none=True);_,normal,_=m(b,'finite_vector');m.feedback.residual_loss(normal,b['reference_prediction']-b['y'],torch.ones(2,dtype=torch.bool,device='cuda')).backward()
assert all(p.grad is None for p in m.donor.parameters());hg=sum(float(p.grad.abs().sum()) for p in m.feedback.parameters() if p.grad is not None);assert hg>0
assert all(b[k].grad is None for k in ['state','old_context','pooled_state'])
m.zero_grad(set_to_none=True)
double=copy.deepcopy(m).double();dp={k:(v.detach().double() if v.dtype.is_floating_point else v.detach()) for k,v in b.items()}
with torch.no_grad():_,_,ob=double(dp,'fixed');f0=ob['message'].clone()
rhohat=double.feedback.estimate_residual(dp['old_context'],dp['pooled_state'],dp['reference_prediction'])[1].detach();terminal=lambda ctx:double.terminal(dp['state'],dp['mask'],ctx)
def evaluate(f):
 ctx,_=double.feedback.control(dp['old_context'],f,dp['reference_prediction'],rhohat,terminal,'finite_vector');return terminal(ctx).square().sum()
f=f0.detach().requires_grad_();value=evaluate(f);grad=torch.autograd.grad(value,f)[0];assert torch.isfinite(grad).all();direction=torch.randn_like(f);direction/=direction.norm();eps=1e-5
numeric=(evaluate((f0+eps*direction).requires_grad_()).item()-evaluate((f0-eps*direction).requires_grad_()).item())/(2*eps);analytic=float((grad*direction).sum());error=abs(numeric-analytic);assert error<max(2e-6,2e-3*abs(analytic)),(analytic,numeric,error)
# Frozen terminal Hessian-vector product itself, including zero-centered slots.
cx=(.5*dp['old_context']).detach().requires_grad_();v=terminal(cx).square().sum();first=torch.autograd.grad(v,cx,create_graph=True)[0];hvp=torch.autograd.grad((first*torch.ones_like(first)).sum(),cx)[0];assert torch.isfinite(first).all() and torch.isfinite(hvp).all()
with torch.no_grad():
 zero=torch.zeros((2,6,100),device='cuda');p0=m.terminal(b['state'].detach(),b['mask'],.5*b['old_context'].detach());zctx,_=m.feedback.control(b['old_context'],zero,p0,torch.zeros_like(p0),lambda c:m.terminal(b['state'].detach(),b['mask'],c),'finite_vector');assert torch.equal(zctx,.5*b['old_context']) and torch.isfinite(zctx).all()
assert all(torch.equal(v,before[k]) for k,v in frozen().items())
r=dict(status='SINGLE_TOKEN_REAL_DOUBLE_FD_HVP_LABEL_MAIN_AUX_ISOLATION_PASSED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=gpu,initial_compute=compute,source_sha256=sha(__file__),reader_source_sha256=sha(pre/'finite_single_token_reader_v1.py'),checkpoint_sha256=sha(old/'run/shared_phase_addon.pt'),train_rows=ids.tolist(),double_unrolled_finite_difference=dict(analytic=analytic,numerical=numeric,error=error,epsilon=eps),terminal_hvp_finite=True,main_head_gradients_none=True,aux_donor_gradients_none=True,frozen_gradients_none=True,coordinate_gradients_none=True,main_donor_gradient_l1=dg,aux_head_gradient_l1=hg,label_replacement_error=0.,train_no_grad_error=0.,inference_zero_signal_exact=True,frozen_parameters_unchanged=True,dev_requested=False,test_requested=False,elapsed_seconds=time.monotonic()-start,argv=sys.argv)
(a.out/'receipt.json').write_text(json.dumps(r,indent=2));print('SINGLE_TOKEN_SECONDORDER_COMPLETE',flush=True)
