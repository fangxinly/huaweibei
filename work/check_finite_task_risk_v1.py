"""TRAIN-only real frozen-terminal preflight; never launches formal training."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import argparse,copy,datetime,hashlib,json,subprocess,sys,time
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--base-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
sys.path.append(str(a.base_root))
from soft_vector_runtime_v1 import FrozenCoordinateLearner as Old,PAIRS
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
def tensor_sha(values):
    h=hashlib.sha256()
    for n,v in sorted(values.items()):
        h.update(n.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def inv():
    run=lambda args:subprocess.check_output(['nvidia-smi',*args],text=True).strip()
    return dict(gpu=run(['--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader,nounits']),compute=run(['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),disk_free=__import__('shutil').disk_usage('/data').free)
assert not a.out.exists();a.out.mkdir()
initial_inv=inv();assert initial_inv['gpu'].split(',')[0]==a.expected_uuid and not initial_inv['compute']
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(91817);np.random.seed(91817)
start=time.monotonic();source=Path(__file__).resolve().parent
cache=np.load(a.base_root/'teacher_cache_v1/train_cache.npz',allow_pickle=False)
fields=['state','mask','old_context','pooled_state','reference_prediction','y','row_id']
train={n:cache[n] for n in fields};assert len(train['y'])==1281
fit_order=np.random.RandomState(91817+173).permutation(1281);fit_rows=np.sort(fit_order[:1153]);heldout=np.sort(fit_order[1153:])
fitmask=np.zeros(1281,dtype=bool);fitmask[fit_rows]=True
np.save(a.out/'head_fit_rows.npy',fit_rows);np.save(a.out/'head_heldout_rows.npy',heldout)
rho=train['reference_prediction']-train['y'];residual_rms=float(np.sqrt(np.mean(rho[fit_rows].astype(np.float64)**2)))
old=Old(a.base_root,'fixed',.1).cuda();old.requires_grad_(False);old.eval();sums=np.zeros(6,dtype=np.float64)
with torch.no_grad():
    for i in range(0,len(fit_rows),64):
        pool=torch.as_tensor(train['pooled_state'][fit_rows[i:i+64]],device='cuda')
        for j,(m,n) in enumerate(PAIRS):
            f=torch.cat([pool[:,m],pool[:,n],.5*(pool[:,m]-pool[:,n]),.5*(pool[:,m]+pool[:,n])],1)
            sums[j]+=old.donor[j](f).double().square().sum().item()
message_rms=np.sqrt(sums/(len(fit_rows)*100))
jac=np.load(a.base_root/'label_free_jacobian_v1/train_jacobian.npz',allow_pickle=False)
assert np.array_equal(jac['row_id'],train['row_id'])
j=jac['reference_jacobian'][fit_rows].astype(np.float64)
sensitivity=.125**2*np.sum(np.repeat(np.sum(j*j,axis=2),2,axis=1)*message_rms[None,:]**2,axis=1)
median=float(np.median(sensitivity));assert median>1e-12
scales=dict(residual_rms=residual_rms,message_rms=message_rms.tolist(),beta=residual_rms**2/(2*median),median_reference_normalized_message_sensitivity_norm2=median,head_fit_rows=1153,head_heldout_rows=128,train_rows=1281,split_seed=91817+173,dev_requested=False,test_requested=False,limits='Head-label holdout only, main donor loss sees allTRAIN and frozen teacher fitted fullTRAIN/DEVselected; not whole-pipeline crossfit or independent validation.')
(a.out/'train_scales.json').write_text(json.dumps(scales,indent=2))
del old;torch.manual_seed(91817)
model=FrozenCoordinateLearner(a.base_root,'fixed',scales).cuda();model.eval()
initial=tensor_sha(model.state_dict());frozen_keys=lambda:{n:v for n,v in model.state_dict().items() if n.startswith(('flow.','fusion.','predictor.'))};frozen_before=tensor_sha(frozen_keys())
orders=np.stack([np.random.RandomState(91817+1327+e).permutation(1281) for e in range(100)]);np.save(a.out/'orders.npy',orders)
def batch(ids):
    b={n:torch.as_tensor(train[n][ids],device='cuda') for n in fields if n!='row_id'}
    b['head_fit_mask']=torch.as_tensor(fitmask[ids],device='cuda');return b
b=batch(orders[0,:32]);b['state'].requires_grad_();b['old_context'].requires_grad_()
pred,n,obs=model(b,'fixed');baseline_error=float((model.terminal(b['state'].detach(),b['mask'],.5*b['old_context'].detach())-b['reference_prediction']).abs().max())
assert baseline_error<2e-5
(pred-b['y']).square().mean().backward()
assert all(x.grad is None for x in model.feedback.parameters())
assert b['state'].grad is None and b['old_context'].grad is None
assert all(x.grad is None for x in model.flow.parameters())
donor_gradient=sum(float(x.grad.abs().sum()) for x in model.donor.parameters() if x.grad is not None);assert donor_gradient>0
model.zero_grad(set_to_none=True)
pred,n,obs=model(b,'fixed');model.feedback.residual_loss(n,b['reference_prediction']-b['y'],b['head_fit_mask']).backward()
assert all(x.grad is None for x in model.donor.parameters())
head_gradient=sum(float(x.grad.abs().sum()) for x in model.feedback.parameters() if x.grad is not None);assert head_gradient>0
model.zero_grad(set_to_none=True)
donor_params=list(model.donor.parameters());head_params=list(model.feedback.parameters())
optimizers=[torch.optim.AdamW(ps,lr=1e-4,weight_decay=.01) for ps in [donor_params,head_params]]
twenty_start=time.monotonic()
for i in range(20):
    b=batch(orders[0,i*32:(i+1)*32]);model.zero_grad(set_to_none=True)
    for opt in optimizers:
        for g in opt.param_groups:g['lr']=1e-4*(i+1)/400
    pred,n,obs=model(b,'fixed');task=(pred-b['y']).square().mean();aux=model.feedback.residual_loss(n,b['reference_prediction']-b['y'],b['head_fit_mask'])
    loss=task+.01*aux;assert torch.isfinite(loss);loss.backward()
    assert all(x.grad is None for x in model.flow.parameters())
    # Separate clipping avoids donor labels affecting head updates through global norm.
    for ps,opt in zip([donor_params,head_params],optimizers):
        torch.nn.utils.clip_grad_norm_(ps,1.);opt.step()
twenty_seconds=time.monotonic()-twenty_start
assert tensor_sha(frozen_keys())==frozen_before
torch.save(model.addon(),a.out/'after20_addon.pt')
probe=batch(np.arange(8));modified=dict(probe,y=probe['y'].flip(0)+20,head_fit_mask=~probe['head_fit_mask'])
mode_checks={}
for mode in ['fixed','finite_scalar','finite_vector']:
    tick=time.monotonic();model.zero_grad(set_to_none=True)
    pred,n,obs=model(probe,mode);changed=model(modified,mode)[0]
    assert torch.equal(pred,changed)
    (pred-probe['y']).square().mean().backward()
    assert all(x.grad is None for x in model.feedback.parameters())
    assert all(x.grad is None for module in [model.flow,model.fusion,model.predictor] for x in module.parameters())
    assert all(x.grad is None or torch.isfinite(x.grad).all() for x in model.donor.parameters())
    with torch.no_grad():infer=model(probe,mode)[0]
    assert torch.allclose(infer,pred.detach(),atol=2e-6,rtol=2e-6)
    mode_checks[mode]=dict(label_replacement_max_error=0.,train_vs_no_grad_max_error=float((infer-pred.detach()).abs().max()),forward_backward_seconds=time.monotonic()-tick,main_head_gradients_none=True,frozen_gradients_none=True)
    if mode=='finite_vector':
        assert float(obs['trust_relative_change'].max())<=.25001
        mode_checks[mode]['trust_max']=float(obs['trust_relative_change'].max())
# Real decoder double-precision directional difference of the unrolled controller.
double=copy.deepcopy(model).double();dp={k:v[:2].detach().to(torch.float64) if v.dtype.is_floating_point else v[:2].detach() for k,v in probe.items()}
with torch.no_grad():_,_,ob=double(dp,'fixed');f0=ob['message'].clone()
rhohat=double.feedback.estimate_residual(dp['old_context'],dp['pooled_state'],dp['reference_prediction'])[1].detach()
terminal=lambda ctx:double.terminal(dp['state'],dp['mask'],ctx)
def evaluate(f):
    ctx,_=double.feedback.control(dp['old_context'],f,dp['reference_prediction'],rhohat,terminal,'finite_vector')
    return terminal(ctx).square().sum()
f=f0.detach().requires_grad_();value=evaluate(f);grad=torch.autograd.grad(value,f)[0]
assert torch.isfinite(grad).all();direction=torch.randn_like(f);direction/=direction.norm()
epsilon=1e-5
numerical=(evaluate((f0+epsilon*direction).requires_grad_()).item()-evaluate((f0-epsilon*direction).requires_grad_()).item())/(2*epsilon)
analytic=float((grad*direction).sum());fd_error=abs(numerical-analytic);assert fd_error<max(2e-6,2e-3*abs(analytic)),(numerical,analytic,fd_error)
# Exact finite squared-loss identity on TRAIN only, plus explicit no-signal control.
with torch.no_grad():
    predictions=model(probe,'finite_vector')[0];delta=predictions-probe['reference_prediction'];true_rho=probe['reference_prediction']-probe['y']
    identity=(predictions-probe['y']).square()-(probe['reference_prediction']-probe['y']).square()
    identity_error=float((identity-model.feedback.risk_change(delta,true_rho)).abs().max());assert identity_error<2e-6
    zero=torch.zeros((8,6,100),device='cuda');p0=model.terminal(probe['state'],probe['mask'],.5*probe['old_context'])
    zctx,zobs=model.feedback.control(probe['old_context'],zero,p0,torch.zeros_like(p0),lambda ctx:model.terminal(probe['state'],probe['mask'],ctx),'finite_vector')
    assert torch.equal(zctx,.5*probe['old_context']) and torch.isfinite(zctx).all()
assert tensor_sha(frozen_keys())==frozen_before
report=dict(status='FINITE_RISK_REAL_TRAIN_MECHANISMS_CHECKED_NOT_FORMAL_TRAINING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_uuid=a.expected_uuid,initial_inventory=initial_inv,inventory=inv(),scales=scales,scales_sha256=sha(a.out/'train_scales.json'),head_fit_rows_sha256=sha(a.out/'head_fit_rows.npy'),head_heldout_rows_sha256=sha(a.out/'head_heldout_rows.npy'),orders_sha256=sha(a.out/'orders.npy'),initial_tensor_sha256=initial,after20_tensor_sha256=tensor_sha(model.state_dict()),after20_addon_sha256=sha(a.out/'after20_addon.pt'),frozen_tensor_sha_before=frozen_before,frozen_tensor_sha_after=tensor_sha(frozen_keys()),trainable_parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),head_parameters=sum(v.numel() for v in model.feedback.parameters()),baseline_cache_replay_max_error=baseline_error,twenty_fixed_updates_seconds=twenty_seconds,mode_checks=mode_checks,unrolled_real_decoder_double_finite_difference=dict(analytic=analytic,numerical=numerical,absolute_error=fd_error,epsilon=epsilon),exact_TRAIN_squared_risk_identity_max_error=identity_error,peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start,source_sha256={n:sha(source/n) for n in ['finite_task_risk_feedback_v1.py','finite_task_risk_runtime_v1.py','check_finite_task_risk_v1.py']},base_source_sha256={n:sha(a.base_root/n) for n in ['soft_vector_runtime_v1.py','task_gradient_vector_candidate_v3.py']},teacher_collection_sha256=sha(a.base_root/'teacher_cache_v1/collection.json'),train_jacobian_sha256=sha(a.base_root/'label_free_jacobian_v1/train_jacobian.npz'),dev_requested=False,test_requested=False,limits=scales['limits'],argv=sys.argv)
(a.out/'preflight.json').write_text(json.dumps(report,indent=2));print('FINITE_REAL_TRAIN_PREFLIGHT_COMPLETE',flush=True)
