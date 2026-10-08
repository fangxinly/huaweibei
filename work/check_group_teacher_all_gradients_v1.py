import os
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,json,datetime,subprocess,time
import torch,numpy as np
from group_teacher_runtime_v1 import setup,forward,row_batch,sha,tensor_sha,norm_free_sha,source_checks,write_json
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--uuid',required=True);a=p.parse_args();tick=time.monotonic()
out=a.root/'all_gradient_precheck_v1';assert not out.exists();out.mkdir()
plan=json.loads((a.root/'group_teacher_plan_v2.json').read_text());source_checks(a.base,a.root,plan)
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip();assert gpu==a.uuid and not compute
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
model,opt,sched,data,fit,inner,stats,matched,args=setup(a.base,a.root,a.fold)
initial=tensor_sha(model.state_dict());pre=json.loads((a.root/'precheck_v1/receipt.json').read_text());assert initial==pre['initial_full_tensor_sha256']
batch=row_batch(fit,data.ids['fit'],np.load(a.root/f'orders_{a.fold}.npy')[0,:32]);model.train();opt.zero_grad(set_to_none=True)
pred=forward(model,batch);loss=(pred-batch[3].view(-1)).square().mean();assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize()
grads={};missing=[];zero=[]
for name,v in model.named_parameters():
 assert v.requires_grad
 if v.grad is None:missing.append(name);continue
 assert torch.isfinite(v.grad).all(),name
 l1=float(v.grad.abs().sum());grads[name]={'numel':v.numel(),'l1':l1}
 if l1==0:zero.append(name)
assert tensor_sha(model.state_dict())==initial
result=dict(status='ALL_RETAINED_PARAMETER_GRADIENT_INSPECTION_COMPLETE_NOT_PARAMETER_UPDATE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_uuid=gpu,fold=a.fold,source_sha256=sha(__file__),runtime_sha256=sha(a.root/'group_teacher_runtime_v1.py'),initial_tensor_sha256=initial,parameter_tensors=len(list(model.parameters())),parameters=sum(v.numel() for v in model.parameters()),finite_gradient_tensors=len(grads),missing_gradient_names=missing,zero_gradient_names=zero,gradients=grads,parameters_updated=False,outer_labels_read=False,dev_requested=False,test_requested=False,elapsed_seconds=time.monotonic()-tick)
write_json(out/'receipt.json',result);print('ALL_PARAMETER_GRADIENT_INSPECTION_COMPLETE',len(grads),len(missing),len(zero),flush=True)
assert not missing,missing
