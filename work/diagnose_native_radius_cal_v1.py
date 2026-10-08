"""CAL-only zero-label real-message coverage precheck. No EVAL or fitting."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,datetime,hashlib,json,importlib.util,shutil,subprocess,sys,time
import numpy as np
import torch

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);a=ap.parse_args()
    start=time.monotonic();p=json.loads(a.plan.read_text());r=Path(p['new_root']);out=r/'mechanism';out.mkdir(exist_ok=False)
    assert sha(__file__)==p['source_sha256']
    for f,h in p['pinned_files'].items():assert sha(f)==h,f
    q=lambda c:subprocess.check_output(c,text=True).strip()
    gpu=q(['nvidia-smi','--query-gpu=uuid,name,memory.used','--format=csv,noheader,nounits'])
    compute=q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'])
    assert gpu.split(',')[0].strip()==p['expected_uuid'] and not compute
    remain=(datetime.datetime.fromisoformat(p['estimated_lease_end_utc'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    assert remain>p['maximum_seconds']+p['preservation_reserve_seconds'] and shutil.disk_usage(r).free>=p['minimum_remote_free_bytes']
    write(out/'inventory.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),argv=sys.argv,gpu=gpu,compute=compute,
        full_process_argv=q(['ps','-ww','-eo','pid,ppid,args']),free_bytes=shutil.disk_usage(r).free,estimated_lease_remaining_seconds=remain,platform_lease_verified=False))
    sys.path[:0]=[p['formal_root'],p['base_root']]
    from finite_task_risk_runtime_v2 import FrozenCoordinateLearner
    spec=importlib.util.spec_from_file_location('helpers',p['helper_source']);helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    torch.set_num_threads(2);torch.manual_seed(91817);torch.use_deterministic_algorithms(True);torch.cuda.reset_peak_memory_stats()
    scaleplan=json.loads((Path(p['formal_root'])/'train_scales.json').read_text())
    model=FrozenCoordinateLearner(Path(p['base_root']),'finite_vector',scaleplan).cuda()
    model.restore_addon(torch.load(Path(p['formal_root'])/'run/best_addon.pt',map_location='cpu'));model.eval();model.requires_grad_(False)
    before=helpers.tensor_sha(model.state_dict());assert before==p['expected_model_state']
    z=np.load(p['prior_predictions'],allow_pickle=False)
    rows=np.flatnonzero(z['role']=='calibration');assert len(rows)==418 and len(np.unique(z['video'][rows]))==18
    assert np.array_equal(rows,np.load(r/'authorized_cal_rows.npy',allow_pickle=False))
    inputs=np.load(Path(p['base_root'])/'teacher_cache_v1/train_cache.npz',allow_pickle=False)
    # Only original row IDs explicitly authorized as CAL enter any terminal forward.
    cache={k:inputs[k][rows] for k in ['state','mask','old_context']}
    fields={k:z[k][rows].copy() for k in ['original_messages','message_scale','old_shift','teacher_unit','pf','native_prediction','target_old','target_cal','lambda_value','cap','video','fold','candidate_valid']}
    def inspect(pos,poison=False):
        b={k:torch.as_tensor(v[pos],device='cuda') for k,v in cache.items()}
        if poison:b['y']=torch.arange(len(pos),device='cuda').float()+99
        f={k:torch.as_tensor(v[pos],device='cuda') for k,v in fields.items() if k not in ['video','candidate_valid']}
        m=f['original_messages'];sc=f['message_scale'];old=b['old_context']
        term=lambda c:model.terminal(b['state'],b['mask'],model.feedback.context(old,c))
        # Reconstruct the exact coordinate convention of the saved old endpoint.
        # The old amp1 candidate is this saved native message, not a renormalized round-trip.
        native_message=(m/sc+f['old_shift'])*sc
        raw_shift=native_message-m;norm_shift=raw_shift/sc
        radius=norm_shift.norm(dim=(1,2),keepdim=True);den=radius.clamp(min=1e-30)
        unit=torch.where(radius>0,norm_shift/den,torch.zeros_like(norm_shift))
        teacher=f['teacher_unit'];candidates=[m]
        for amp in [.125,.25,.5,1.]:candidates.append(native_message if amp==1. else m+amp*raw_shift)
        for amp in [.125,.25,.5,1.]:candidates.append(m+sc*radius*amp*teacher)
        for amp in [.125,.25,.5,1.]:candidates.append(m-amp*raw_shift)
        messages=torch.stack(candidates,1)
        with torch.no_grad():pred=torch.stack([term(c) for c in candidates],1)
        assert torch.isfinite(messages).all() and torch.isfinite(pred).all()
        assert float((pred[:,0]-f['pf']).abs().max())<=1e-6
        assert float((pred[:,4]-f['native_prediction']).abs().max())<=1e-6
        assert torch.equal(messages[:,0],m) and torch.equal(messages[:,4],native_message)
        shift=(messages-m[:,None])/sc[:,None]
        rel=shift.flatten(2).norm(dim=2)/(m/sc).norm(dim=(1,2)).clamp(min=1e-30)[:,None]
        assert (rel<=.25001).all()
        cap=f['cap'].double();valid=(pred.double()-pred[:,0].double()[:,None]).abs()<=cap[:,None];valid[:,0]=True
        # Only zero-label eligibility accounting; no acceptance selection or new EVAL score.
        unique=torch.ones(len(pos),13,dtype=torch.bool,device='cuda')
        for i in range(1,13):unique[:,i]=~torch.stack([(messages[:,i]==messages[:,j]).flatten(1).all(1) for j in range(i)],1).any(1)
        qs=[]
        for target in [f['target_old'],f['target_cal']]:
            delta=pred.double()-pred[:,0].double()[:,None];qs.append(2*(pred[:,0].double()-target.double())[:,None]*delta+delta.square())
        v={'row':rows[pos],'fold':fields['fold'][pos],'video':fields['video'][pos],'valid_lengths':b['mask'].sum(-1),'original_messages':m,
            'message_scale':sc,'saved_old_shift':f['old_shift'],'native_messages':native_message,'old_unit':unit,'teacher_unit':teacher,
            'normalized_native_radius':radius.flatten(),'candidate_messages':messages,'candidate_prediction':pred,'candidate_valid':valid,'candidate_unique':unique,
            'relative_message_shift':rel,'target_old':f['target_old'],'target_cal':f['target_cal'],'lambda_value':f['lambda_value'],'cap':cap,
            'proxy_risk_change':torch.stack(qs,1),'prior_grid_candidate_valid':fields['candidate_valid'][pos],'saved_pf':f['pf'],'saved_native_prediction':f['native_prediction']}
        return {k:t.detach().cpu().numpy() if isinstance(t,torch.Tensor) else t for k,t in v.items()}
    sentinel=np.flatnonzero(np.isin(rows,[454,620]));assert len(sentinel)==2
    a0=inspect(sentinel);a1=inspect(sentinel,True);assert all(np.array_equal(a0[k],a1[k]) for k in a0)
    assert (a0['valid_lengths']==1).all()
    singles=[inspect(np.asarray([int(pos)])) for pos in sentinel]
    coupling=max(float(np.max(np.abs(a0['candidate_prediction'][i]-s['candidate_prediction'][0]))) for i,s in enumerate(singles));assert coupling<=1e-6
    chunks={}
    for first in range(0,418,32):
        pos=np.arange(first,min(first+32,418));v=inspect(pos)
        for k,t in v.items():chunks.setdefault(k,[]).append(t)
        assert torch.cuda.max_memory_allocated()<=p['maximum_peak_allocated_bytes'] and time.monotonic()-start<p['maximum_seconds']
    full={k:np.concatenate(v) for k,v in chunks.items()};path=out/'cal_candidate_arrays.npz';np.savez_compressed(path,**full)
    assert path.stat().st_size<=p['maximum_new_array_bytes']
    assert before==helpers.tensor_sha(model.state_dict()) and all(t.grad is None for t in model.parameters())
    write(out/'receipt.json',dict(status='CAL_ONLY_NATIVE_RADIUS_ZERO_LABEL_GPU_MECHANISM_COMPLETED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),argv=sys.argv,
        source_sha256=sha(__file__),plan_sha256=sha(a.plan),original_arrays_sha256=sha(p['prior_predictions']),prediction_sha256=sha(path),array_bytes=path.stat().st_size,
        rows=418,videos=18,model_state_before=before,model_state_after=before,parameter_gradients_all_None=True,optimizer_created=False,optimizer_steps=0,
        saved_native_terminal_replay_max_error=float(np.max(np.abs(full['candidate_prediction'][:,4]-full['saved_native_prediction']))),
        saved_F_terminal_replay_max_error=float(np.max(np.abs(full['candidate_prediction'][:,0]-full['saved_pf']))),single_token_cal_rows=[454,620],poisoned_label_error=0,
        single_token_single_batch_max_error=coupling,seconds=time.monotonic()-start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        real_labels_read=False,CAL_fit=False,EVAL_forward_rows=0,EVAL_metrics=False,DEV_read=False,TEST_read=False,path_solver_rerun=False,selected_predictions_generated=False,
        scope='Real finite-message CAL-only mechanical coverage; not new control benefit or original DeBERTa replay. Prior labels remain outside accessed NPZ fields.'))
    print('CAL_NATIVE_RADIUS_MECHANISM_COMPLETE',418,flush=True)
if __name__=='__main__':main()
