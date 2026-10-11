"""Actual CUDA synthetic module check; does not certify full-model or TRAIN/dev."""
import argparse,copy,datetime,hashlib,inspect,json,os,subprocess,time
from pathlib import Path
import torch
from task_gradient_vector_candidate_v3 import TaskGradientVectorFeedback,soft_halfspace_proximal
p=argparse.ArgumentParser();p.add_argument('--expected-gpu-uuid',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
assert not a.out.exists()
query=lambda args:subprocess.check_output(['nvidia-smi',*args],text=True).strip()
initial=query(['--query-gpu=uuid,name,memory.used','--format=csv,noheader,nounits'])
assert initial.split(',')[0].strip()==a.expected_gpu_uuid and int(initial.split(',')[-1])<=16,initial
assert not query(['--query-compute-apps=pid','--format=csv,noheader,nounits'])
assert torch.cuda.is_available()
torch.set_num_threads(2);torch.manual_seed(20261005);torch.cuda.manual_seed_all(20261005)
device=torch.device('cuda:0');b,d=4,100
template=TaskGradientVectorFeedback('fixed',relative_norm2=.01).to(device)
old=torch.randn(b,3,d,device=device);pool=torch.randn_like(old);ref=torch.randn(b,device=device);donor=torch.randn(b,6,d,device=device)*.005+.025
try:template(old,pool,ref,donor)
except AssertionError:unfitted_rejected=True
else:raise AssertionError('Unfitted scale accepted')
template.set_train_fitted_gradient_rms(torch.ones(3,device=device))
def state_sha(m):
 h=hashlib.sha256()
 for name,value in sorted(m.state_dict().items()):
  h.update(name.encode());h.update(str((tuple(value.shape),str(value.dtype))).encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
models={mode:copy.deepcopy(template) for mode in ('fixed','scalar','soft_projected')}
for mode,m in models.items():m.mode=mode
hashes={mode:state_sha(m) for mode,m in models.items()};assert len(set(hashes.values()))==1
with torch.no_grad():contexts={mode:m(old,pool,ref,donor)[0] for mode,m in models.items()}
assert all(torch.equal(contexts['fixed'],v) for v in contexts.values())
assert list(inspect.signature(template.forward).parameters)==['old_context','pooled_state','reference_prediction','donor_messages']
rows={}
for mode,m in models.items():
 for head in m.gradient_heads:
  with torch.no_grad():head[-1].bias.fill_(1.)
 old_i=old.clone().requires_grad_();pool_i=pool.clone().requires_grad_();ref_i=ref.clone().requires_grad_();f=donor.clone().requires_grad_()
 context,prediction,obs=m(old_i,pool_i,ref_i,f)
 assert (obs['estimated_raw_message_dot']>0).all()
 context.sum().backward()
 assert all(v.grad is None for v in m.gradient_heads.parameters())
 assert pool_i.grad is None and ref_i.grad is None
 assert torch.allclose(old_i.grad,torch.full_like(old_i,.5))
 g=obs['predicted_gradient'].repeat_interleave(2,1)
 if mode=='fixed':expected=torch.full_like(f,.125)
 elif mode=='scalar':
  w=obs['scalar_weights'][:,:,None]
  expected=.25*(w-2.5*w*(1-w)*f.detach().sum(-1,keepdim=True)*g)
 else:
  lam=obs['projection_norm2_regularization'];n2=g.square().sum(-1,keepdim=True)
  expected=.125*(1-g*g.sum(-1,keepdim=True)/(n2+lam))
 error=(f.grad-expected).abs().max().item();assert error<1e-6,error
 teacher=torch.randn_like(prediction,requires_grad=True)
 aux=m.gradient_regression_loss(prediction,teacher);aux.backward()
 assert teacher.grad is None and any(v.grad is not None and v.grad.abs().sum()>0 for v in m.gradient_heads.parameters())
 with torch.no_grad():
  c1=m(old,pool,ref,donor)[0];c2=m(old,pool,ref,donor)[0]
 assert torch.equal(c1,c2)
 optimizer=torch.optim.AdamW(m.gradient_heads.parameters(),lr=1e-4)
 torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
 for _ in range(20):
  optimizer.zero_grad(set_to_none=True)
  _,normalized,_=m(old,pool,ref,donor)
  loss=m.gradient_regression_loss(normalized,teacher.detach())
  assert torch.isfinite(loss);loss.backward();optimizer.step()
 torch.cuda.synchronize()
 rows[mode]={'task_message_gradient_max_error':error,'message_gradient_l1':float(f.grad.abs().sum()),'auxiliary_teacher_grad_none':True,'task_gradient_heads_grad_none':True,'feature_grad_detached':True,'synthetic_20_updates_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'initial_sha':hashes[mode]}
# Double-precision local CUDA gradcheck checks the pure vector function only.
f64=torch.tensor([[1.,.3,-.2]],device=device,dtype=torch.float64,requires_grad=True)
g64=torch.tensor([[1.,.1,.1]],device=device,dtype=torch.float64)
assert torch.autograd.gradcheck(lambda f:soft_halfspace_proximal(f,g64,.7),(f64,),eps=1e-6,atol=1e-5,rtol=1e-3)
compute=query(['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'])
compute_rows=[line for line in compute.splitlines() if line.strip()]
assert len(compute_rows)==1 and a.expected_gpu_uuid in compute_rows[0],compute
namespace_pid_record=[x for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('NSpid')]
r={'status':'CUDA_SYNTHETIC_MODULE_FORWARD_BACKWARD_VERIFIED_NOT_FULL_MODEL','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':os.sys.argv,'pid':os.getpid(),'namespace_pid_record':namespace_pid_record,'host_compute_pid':compute_rows[0].split(',')[0].strip(),'pid_equality_claimed':False,'binding_basis':'Initially empty expected UUID, own CUDA allocation, exactly one compute entry on same UUID; post-exit emptiness requires separate observation','gpu_uuid':a.expected_gpu_uuid,'gpu_initial':initial,'compute_during_check':compute,'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'source_sha256':hashlib.sha256(Path(__file__).with_name('task_gradient_vector_candidate_v3.py').read_bytes()).hexdigest(),'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'initial_state_sha':hashes['fixed'],'actual_parameters':sum(v.numel() for v in template.parameters()),'unfitted_scale_rejected':True,'zero_head_initial_controls_equal':True,'pure_message_cuda_gradcheck':True,'mode_checks':rows,'scale_source':'SYNTHETIC_ONES_ONLY_NOT_TRAIN','relative_norm2':.01,'hyperparameter_selected_for_experiment':False,'full_model_integrated':False,'full_model_label_isolation_verified':False,'actual_train_teacher_collected':False,'formal_training_started':False,'train_dev_test_accessed':False}
a.out.write_text(json.dumps(r,indent=2),encoding='utf-8')
encoded=__import__('base64').b64encode(json.dumps(r).encode()).decode()
print('REPORT_BEGIN')
for i in range(0,len(encoded),64):print(encoded[i:i+64])
print('REPORT_END')

