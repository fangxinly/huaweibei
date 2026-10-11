import os
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,json,subprocess,datetime,time,zipfile
import torch,numpy as np
from group_teacher_runtime_v1 import setup,forward,row_batch,norm_free_sha,tensor_sha,sha,source_checks,write_json
p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--uuid',required=True);a=p.parse_args();start=time.monotonic();out=a.root/'precheck_v1';assert not out.exists();out.mkdir()
plan=json.loads((a.root/'group_teacher_plan_v2.json').read_text());source_checks(a.base,a.root,plan);info=plan['folds'][a.fold]
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True).strip();assert gpu.split(',')[0]==a.uuid and not compute
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
model,opt,sched,data,fit,inner,stats,matched,arguments=setup(a.base,a.root,a.fold)
orders=np.load(a.root/f'orders_{a.fold}.npy');assert orders.shape==(100,len(fit)) and all(np.array_equal(np.sort(x),data.ids['fit']) for x in orders)
for n,v in info['files'].items():assert sha(a.root/n)==v['sha256']
initial=tensor_sha(model.state_dict());shared=norm_free_sha(model)
try:data.dataset([data.ids['outer'][0]],'fit');raise AssertionError('outerguard failed')
except PermissionError:outer_denied=True
from encoder_adapter import content_mask
valid=content_mask(fit.tensors[4]);witness={}
for name,index in [('audio',2),('visual',1)]:
 values=fit.tensors[index].squeeze(1)[valid].cpu().numpy();witness[name+'_raw_fit_content_tokens']=values
 for key,value in stats[name].items():witness[name+'_'+key]=value.cpu().numpy()
np.savez(out/'normalization_witness.npz',**witness)
batch=row_batch(fit,data.ids['fit'],orders[0,:32]);model.eval()
with torch.no_grad():
 pred=forward(model,batch);changed=list(batch);changed[3]=changed[3].flip(0)+20;label_error=float((pred-forward(model,tuple(changed))).abs().max());assert label_error==0
cp=out/'initial_full_checkpoint.pt';torch.save(model.state_dict(),cp)
model.load_state_dict(torch.load(cp,map_location='cpu'),strict=True);model.eval()
with torch.no_grad():reload_error=float((pred-forward(model,batch)).abs().max());assert reload_error==0
with zipfile.ZipFile(cp) as z:assert z.testzip() is None
np.savez(out/'initial_replay_predictions.npz',prediction=pred.cpu().numpy(),fit_row_ids=orders[0,:32])
targets={'text':model.dberta.model.embeddings.word_embeddings.weight,'audio':model.dberta.proj_a.weight,'visual':model.dberta.proj_v.weight,'audio_transformer':model.dberta.transa.layers[0].self_attn.in_proj_weight,'visual_transformer':model.dberta.transv.layers[0].self_attn.in_proj_weight,'fusion':model.dberta.fusion[0].weight,'decoder':model.dberta.predictor[-1].weight}
before=model.dberta.fusion[0].weight.detach().clone();update_seconds=[];gradients=[]
for step in range(4):
 tick=time.monotonic();model.train();ids=orders[0,step*32:(step+1)*32];b=row_batch(fit,data.ids['fit'],ids);opt.zero_grad(set_to_none=True);prediction=forward(model,b);loss=(prediction-b[3].view(-1)).square().mean();assert torch.isfinite(loss);loss.backward()
 current={}
 for n,v in targets.items():assert v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum()>0;current[n]=float(v.grad.abs().sum())
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());opt.step();sched.step();torch.cuda.synchronize();update_seconds.append(time.monotonic()-tick);gradients.append(current)
assert not torch.equal(before,model.dberta.fusion[0].weight)
assert all(torch.isfinite(v).all() for v in model.state_dict().values())
tick=time.monotonic();model.eval()
with torch.no_grad():
 b=tuple(x[:128].cuda() for x in inner.tensors);prediction=forward(model,b);changed=list(b);changed[3]=changed[3].flip(0)-20;inner_label_error=float((prediction-forward(model,tuple(changed))).abs().max());assert inner_label_error==0
torch.cuda.synchronize();inner_pair_seconds=time.monotonic()-tick
assert not any(x['role'].startswith('outer') for x in data.journal)
r=dict(status='GROUP_TEACHER_FIT_ONLY_NORMALIZATION_OUTER_GUARD_PUBLIC_INIT_FULL_DISK_RELOAD_AND_FOUR_REAL_UPDATES_PASSED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=gpu,initial_compute=compute,fold=a.fold,seed=91818,fit_rows=len(fit),inner_rows=len(inner),outer_rows=len(data.ids['outer']),initial_full_tensor_sha256=initial,initial_non_normalization_tensor_sha256=shared,initial_checkpoint_sha256=sha(cp),initial_checkpoint_bytes=cp.stat().st_size,initial_full_state_tensor_count=len(model.state_dict()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),pretrained_matched_tensors=matched,outer_access_denied=outer_denied,fit_label_replacement_error=label_error,inner_label_replacement_error=inner_label_error,strict_disk_reload_error=reload_error,update_seconds=update_seconds,gradient_l1=gradients,real_parameter_update=True,inner128_two_forwards_seconds=inner_pair_seconds,peak_allocated_bytes=torch.cuda.max_memory_allocated(),normalization_witness_sha256=sha(out/'normalization_witness.npz'),data_access_journal=data.journal,outer_predictions_generated=False,dev_requested=False,test_requested=False,source_sha256=sha(__file__),runtime_sha256=sha(a.root/'group_teacher_runtime_v1.py'),plan_sha256=sha(a.root/'group_teacher_plan_v2.json'),elapsed_seconds=time.monotonic()-start,scope=plan['scope'])
write_json(out/'receipt.json',r);print('GROUP_TEACHER_PRECHECK_COMPLETE',a.fold,r['trainable_parameters'],update_seconds,flush=True)
