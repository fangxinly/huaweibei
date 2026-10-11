"""Matched100 epochs after real mechanisms; frozen backbone TRAIN/dev study."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import argparse,datetime,hashlib,json,subprocess,sys,time
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--base-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--mode',choices=['fixed','finite_scalar','finite_vector'],required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
sys.path.append(str(a.base_root));from finite_task_risk_runtime_v1 import FrozenCoordinateLearner
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def tensor_sha(values):
    h=hashlib.sha256()
    for n,v in sorted(values.items()):
        h.update(n.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def write_json(path,obj):
    temp=path.with_suffix('.pending.json');temp.write_text(json.dumps(obj,indent=2));temp.replace(path)
def save_addon(path,obj):
    temp=path.with_suffix('.pending.pt');torch.save(obj,temp);temp.replace(path)
def stamp():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def inventory():
    q=lambda args:subprocess.check_output(['nvidia-smi',*args],text=True).strip()
    return dict(utc=stamp(),gpu=q(['--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader,nounits']),compute=q(['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),disk_free=__import__('shutil').disk_usage('/data').free)
plan=json.loads((a.root/'finite_formal_plan_v1.json').read_text())
assert plan['seed']==91817 and plan['epochs']==100 and plan['shared_fixed_epochs']==10
for filename,h in plan['source_sha256'].items():assert sha(a.root/filename)==h
for filename,h in plan['base_source_sha256'].items():assert sha(a.base_root/filename)==h
for filename,h in plan['teacher_file_sha256'].items():assert sha(a.base_root/'teacher_cache_v1'/filename)==h
scales=json.loads((a.root/'train_scales.json').read_text());assert sha(a.root/'train_scales.json')==plan['scales_sha256']
fit=np.load(a.root/'head_fit_rows.npy');assert sha(a.root/'head_fit_rows.npy')==plan['head_fit_rows_sha256']
assert not a.out.exists();a.out.mkdir()
first_inventory=inventory();assert first_inventory['gpu'].split(',')[0]==a.expected_uuid and not first_inventory['compute']
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(91817);np.random.seed(91817)
start=time.monotonic();model=FrozenCoordinateLearner(a.base_root,a.mode,scales).cuda();model.eval()
initial=tensor_sha(model.state_dict());assert initial==plan['initial_tensor_sha256']
frozen=lambda:{n:v for n,v in model.state_dict().items() if n.startswith(('flow.','fusion.','predictor.'))}
frozen_sha=tensor_sha(frozen());orders=np.stack([np.random.RandomState(91817+1327+i).permutation(1281) for i in range(100)])
np.save(a.out/'orders.npy',orders);assert sha(a.out/'orders.npy')==plan['orders_sha256']
fields=['state','mask','old_context','pooled_state','reference_prediction','y']
train_np=np.load(a.base_root/'teacher_cache_v1/train_cache.npz',allow_pickle=False);dev_np=np.load(a.base_root/'teacher_cache_v1/dev_cache.npz',allow_pickle=False)
train={n:train_np[n] for n in fields};dev={n:dev_np[n] for n in fields};assert len(train['y'])==1281 and len(dev['y'])==229
fit_mask=np.zeros(1281,dtype=bool);fit_mask[fit]=True
donor_parameters=list(model.donor.parameters());head_parameters=list(model.feedback.parameters())
optimizers=[torch.optim.AdamW(group,lr=1e-4,weight_decay=.01) for group in [donor_parameters,head_parameters]]
def batch(data,ids,training=False):
    b={n:torch.as_tensor(v[ids],device='cuda') for n,v in data.items()}
    if training:b['head_fit_mask']=torch.as_tensor(fit_mask[ids],device='cuda')
    return b
protocol=dict(scope=plan['scope'],seed=91817,epochs=100,requested_mode=a.mode,shared_fixed_epochs=10,train_rows=1281,dev_rows=229,test_requested=False,head_fit_rows=1153,head_heldout_rows=128,head_aux_weight=.01,initial_tensor_sha256=initial,orders_sha256=sha(a.out/'orders.npy'),scales_sha256=sha(a.root/'train_scales.json'),head_fit_rows_sha256=sha(a.root/'head_fit_rows.npy'),formal_plan_sha256=sha(a.root/'finite_formal_plan_v1.json'),source_sha256=plan['source_sha256'],base_source_sha256=plan['base_source_sha256'],teacher_collection_sha256=sha(a.base_root/'teacher_cache_v1/collection.json'),initial_inventory=first_inventory,gpu_uuid=a.expected_uuid,argv=sys.argv,started_utc=stamp(),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),head_parameters=sum(p.numel() for p in head_parameters),scales=scales,head_holdout_limits=plan['head_holdout_limits'])
assert protocol['trainable_parameters']==520506 and protocol['head_parameters']==214506
write_json(a.out/'protocol.json',protocol)
history=[];best=float('inf');best_epoch=None
for epoch in range(1,101):
    mode='fixed' if epoch<=10 else a.mode;losses=[];tick=time.monotonic()
    for i in range(40):
        step=(epoch-1)*40+i;b=batch(train,orders[epoch-1,i*32:(i+1)*32],True)
        model.zero_grad(set_to_none=True)
        for opt in optimizers:
            for group in opt.param_groups:group['lr']=1e-4*min((step+1)/400,max(0.,(4000-step)/3600))
        pred,n,obs=model(b,mode);task=(pred-b['y']).square().mean()
        aux=model.feedback.residual_loss(n,b['reference_prediction']-b['y'],b['head_fit_mask'])
        loss=task+.01*aux;assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None for module in [model.flow,model.fusion,model.predictor] for p in module.parameters())
        for parameters,opt in zip([donor_parameters,head_parameters],optimizers):
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters)
            torch.nn.utils.clip_grad_norm_(parameters,1.);opt.step()
        if step==19:assert tensor_sha(model.state_dict())==plan['preflight_after20_tensor_sha256']
        losses.append([float(task.detach()),float(aux.detach())])
    if epoch==10:save_addon(a.out/'shared_phase_addon.pt',model.addon())
    predictions=[];normalized=[];observations={}
    with torch.no_grad():
        for i in range(0,229,128):
            pred,n,obs=model(batch(dev,np.arange(i,min(i+128,229))),mode)
            predictions.append(pred.cpu().numpy());normalized.append(n.cpu().numpy())
            for key,value in obs.items():observations.setdefault(key,[]).append(value.detach().cpu().numpy())
    pred=np.concatenate(predictions);y=dev['y'];score=float(np.mean([np.mean((pred[:128]-y[:128])**2),np.mean((pred[128:]-y[128:])**2)]))
    row=dict(epoch=epoch,effective_mode=mode,train_task_mse=float(np.mean(losses,axis=0)[0]),train_residual_loss=float(np.mean(losses,axis=0)[1]),dev_author_batch_mean_mse=score,dev_mse_229=float(np.mean((pred-y)**2)),dev_mae_229=float(np.mean(np.abs(pred-y))),seconds=time.monotonic()-tick,utc=stamp())
    assert all(np.isfinite(v) for v in row.values() if isinstance(v,float));history.append(row)
    if score<best:
        best=score;best_epoch=epoch;save_addon(a.out/'best_addon.pt',model.addon())
        pending=a.out/'predictions.pending.npz';np.savez(pending,valid_pred=pred,valid_y=y,normalized_residual=np.concatenate(normalized),**{k:np.concatenate(v) for k,v in observations.items()});pending.replace(a.out/'predictions.npz')
    write_json(a.out/'history.json',history);print(json.dumps(row),flush=True)
model.restore_addon(torch.load(a.out/'best_addon.pt',map_location='cpu'));assert tensor_sha(frozen())==frozen_sha
effective='fixed' if best_epoch<=10 else a.mode
with torch.no_grad():replay=np.concatenate([model(batch(dev,np.arange(i,min(i+128,229))),effective)[0].cpu().numpy() for i in range(0,229,128)])
with np.load(a.out/'predictions.npz') as z:assert np.array_equal(replay,z['valid_pred'])
selection=dict(status='FINITE_100_EPOCH_SELECTED_ADDON_NOT_FULL_CHECKPOINT',epochs=100,best_epoch=best_epoch,best_dev_author_batch_mean_mse=best,requested_mode=a.mode,effective_mode=effective,addon_sha256=sha(a.out/'best_addon.pt'),prediction_sha256=sha(a.out/'predictions.npz'),shared_phase_sha256=sha(a.out/'shared_phase_addon.pt'),selected_prediction_replay_max_error=0.,frozen_tensor_sha_before=frozen_sha,frozen_tensor_sha_after=tensor_sha(frozen()),peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start,inventory=inventory(),completed_utc=stamp())
write_json(a.out/'selection.json',selection);print('FINITE_100_EPOCH_ADDON_COMPLETE',flush=True)
