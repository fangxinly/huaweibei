"""Label-free finite real-message candidates; immutable C2, no student fitting."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import argparse, copy, datetime, hashlib, importlib.util, json, shutil, subprocess, sys, time
import numpy as np
import torch

POOLS = [[0,1,2,3,4], [0,5,6,7,8], [0,1,2,3,4,5,6,7,8], [0,1,2,3,4,9,10,11,12]]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def selector(pred,pf,target,valid,ids,force_f=None):
    d=pred.double()-pf.double()[:,None]
    q=2*(pf.double()-target.double())[:,None]*d+d.square()
    score=q[:,ids].clone();score[~valid[:,ids]]=float('inf');score[:,0]=0
    index=torch.as_tensor(ids,device=pred.device)[score.argmin(1)]
    best=score.min(1).values
    index[~(best<0)]=0
    if force_f is not None:index[force_f]=0
    return index,q
def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--phase',choices=['precheck','execute'],required=True);a=p.parse_args()
    began=time.monotonic();plan=json.loads(a.plan.read_text());root=Path(plan['new_root'])
    assert plan['status']=='FROZEN_MATCHED_REAL_MESSAGE_POOLS_V1'
    assert sha(__file__)==plan['source_sha256']
    for name,value in plan['pinned_files'].items():assert sha(name)==value,name
    out=root/a.phase;out.mkdir(exist_ok=False)
    query=lambda argv:subprocess.check_output(argv,text=True).strip()
    gpu=query(['nvidia-smi','--query-gpu=uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'])
    compute=query(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits'])
    assert gpu.split(',')[0].strip()==plan['expected_uuid'] and not compute
    remaining=(datetime.datetime.fromisoformat(plan['estimated_lease_end_utc'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    assert remaining>plan['maximum_seconds']+plan['preservation_reserve_seconds']
    assert shutil.disk_usage(root).free>=plan['minimum_remote_free_bytes']
    write(out/'inventory.json',dict(utc=utc(),pid=os.getpid(),argv=sys.argv,gpu=gpu,compute=compute,
        processes=query(['ps','-ww','-eo','pid,ppid,args']),free_bytes=shutil.disk_usage(root).free,
        estimated_remaining_seconds=remaining,platform_lease_verified=False))
    if a.phase=='execute':
        auth=json.loads((root/'execution_authorization.json').read_text())
        assert auth['source_sha256']==sha(__file__) and auth['plan_sha256']==sha(a.plan)
        assert auth['precheck_sha256']==sha(root/'precheck/receipt.json')
        assert auth['precheck_audit_sha256']==sha(root/'precheck/independent_audit.json')
        assert auth['actual_precheck_capture_verified'] is True
        assert json.loads((root/'precheck_exit.json').read_text())['exit_code']==0
    base=Path(plan['base_root']);formal=Path(plan['formal_root']);sys.path[:0]=[str(formal),str(base)]
    from finite_task_risk_runtime_v2 import FrozenCoordinateLearner,PAIRS
    spec=importlib.util.spec_from_file_location('immutable_helpers',plan['helper_source'])
    helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(helpers)
    torch.set_num_threads(2);torch.manual_seed(91817);torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats() # Includes construction/loading, both paths and all terminal calls.
    scales=json.loads((formal/'train_scales.json').read_text())
    selection=json.loads((formal/'run/selection.json').read_text())
    assert selection['epochs']==100 and selection['best_epoch']==37 and scales['beta']==plan['beta']
    model=FrozenCoordinateLearner(base,'finite_vector',scales).cuda()
    model.restore_addon(torch.load(formal/'run/best_addon.pt',map_location='cpu'))
    model.eval();model.requires_grad_(False);before=helpers.tensor_sha(model.state_dict());fb=model.feedback
    assert fb.inner_steps==3 and fb.step_size==.25 and fb.trust_fraction==.25
    cache=np.load(base/'teacher_cache_v1/train_cache.npz',allow_pickle=False)
    inputs={k:cache[k] for k in ['state','mask','old_context','pooled_state','reference_prediction']}
    with np.load(plan['old_diagnostic_arrays'],allow_pickle=False) as z:
        native=z['original_prediction'].copy();old_path=z['learned_prediction_path'].copy()
        old_raw=z['raw_prediction'].copy();old_rho=z['estimated_residual'].copy()
        assert np.array_equal(z['p0'],inputs['reference_prediction'])
    with np.load(plan['teacher_path_arrays'],allow_pickle=False) as z:teacher_path=z['prediction_path'].copy()
    with np.load(root/'mu_input.npz',allow_pickle=False) as z:
        assert set(z.files)=={'row_ids','fold','mu'} and np.array_equal(z['row_ids'],np.arange(1281))
        mu=z['mu'].copy();fold=z['fold'].copy()
    with np.load(root/'roles.npz',allow_pickle=False) as z:
        assert set(z.files)=={'row_ids','fold','video','role'}
        assert np.array_equal(z['row_ids'],np.arange(1281)) and np.array_equal(z['fold'],fold)
        videos=z['video'].copy();roles=z['role'].copy()
    params=json.loads((root/'cal_parameters.json').read_text())
    lambdas=np.asarray(params['lambda_by_fold']);caps=np.asarray(params['cap_by_fold'])
    assert np.isfinite(lambdas).all() and ((lambdas>=0)&(lambdas<=1)).all() and (caps>=0).all()
    scale=fb.train_message_rms[None,:,None]
    def bounded_path(old,messages,p0,rho,terminal):
        # Fixed F anchor across all updates; exact frozen numeric update, no new fitting.
        u0=(messages/scale).detach();u=u0.clone().requires_grad_()
        radius=.25*u0.norm(dim=(1,2),keepdim=True);predictions=[]
        for step in range(4):
            with torch.enable_grad():
                pred=terminal(fb.context(old,u*scale))
                objective=.5*(u-u0).square().sum((1,2))+fb.finite_beta*fb.risk_change(pred-p0,rho)/fb.train_residual_rms.square()
                if step<3:g=torch.autograd.grad(objective.sum(),u)[0]
            assert torch.isfinite(pred).all() and torch.isfinite(objective).all()
            predictions.append(pred.detach())
            if step<3:
                assert torch.isfinite(g).all()
                proposed=u.detach()-.25*g.detach();shift=proposed-u0
                factor=(radius/shift.norm(dim=(1,2),keepdim=True).clamp(min=1e-12)).clamp(max=1)
                u=(u0+factor*shift).detach().requires_grad_()
        return u.detach()-u0,torch.stack(predictions,1)
    def inspect(rows,poison=False):
        b={k:torch.as_tensor(v[rows],device='cuda') for k,v in inputs.items()}
        if poison:b['y']=torch.arange(len(rows),device='cuda').float()+99 # Deliberately ignored.
        pool=b['pooled_state'].detach();old=b['old_context'].detach();p0=b['reference_prediction'].detach()
        terminal=lambda cx:model.terminal(b['state'].detach(),b['mask'],cx)
        with torch.no_grad():
            m=torch.stack([model.donor[i](torch.cat([pool[:,r],pool[:,d],.5*(pool[:,r]-pool[:,d]),.5*(pool[:,r]+pool[:,d])],-1)) for i,(r,d) in enumerate(PAIRS)],1)
            pf=terminal(fb.context(old,m)).detach()
            rho=fb.estimate_residual(old,pool,p0)[1].detach()
            p0new=terminal(.5*old)
        target=torch.as_tensor(mu[rows],dtype=pf.dtype,device='cuda').detach()
        so,po=bounded_path(old,m,p0,rho,terminal)
        st,pt=bounded_path(old,m,p0,(p0-target).detach(),terminal)
        def unit(v):
            n=v.norm(dim=(1,2),keepdim=True);return torch.where(n>0,v/n.clamp(min=1e-12),torch.zeros_like(v))
        do,dt=unit(so),unit(st);u0=(m/scale).detach();radius=.25*u0.norm(dim=(1,2),keepdim=True)
        candidates=[m] # Original exact F; no normalization round-trip on fallback.
        for direction in [do,dt,-do]:
            for amp in [.125,.25,.5,1.]:candidates.append((m+scale*radius*amp*direction).detach())
        messages=torch.stack(candidates,1)
        with torch.no_grad():pred=torch.stack([terminal(fb.context(old,c)) for c in candidates],1)
        pred[:,0]=pf
        cap=torch.as_tensor(caps[fold[rows]],device='cuda',dtype=torch.float64)
        delta=pred.double()-pf.double()[:,None]
        finite=torch.isfinite(messages).flatten(2).all(2)&torch.isfinite(pred)
        valid=finite&(delta.abs()<=cap[:,None]);valid[:,0]=True
        lam=torch.as_tensor(lambdas[fold[rows]],device='cuda',dtype=torch.float64)
        told=(p0-rho).double().detach();tcal=(pf.double()+lam*(target.double()-pf.double())).detach()
        indices=[];selected=[];replays=[];scores=[]
        for tar,force in [(told,None),(tcal,lam==0)]:
            for ids in POOLS:
                idx,q=selector(pred,pf,tar,valid,ids,force)
                chosen=messages[torch.arange(len(rows),device='cuda'),idx]
                with torch.no_grad():replay=terminal(fb.context(old,chosen))
                chosen_pred=pred.gather(1,idx[:,None])[:,0]
                assert float((replay-chosen_pred).abs().max())<=1e-6
                assert torch.equal(chosen[idx==0],m[idx==0])
                assert ((chosen_pred.double()-pf.double()).abs()<=cap).all()
                indices.append(idx);selected.append(chosen_pred);replays.append(replay);scores.append(q)
        duplicates=torch.zeros(len(rows),13,dtype=torch.bool,device='cuda')
        for i in range(1,13):duplicates[:,i]=torch.stack([torch.eq(messages[:,i],messages[:,j]).flatten(1).all(1) for j in range(i)],1).any(1)
        relative=((messages-m[:,None])/scale[:,None]).flatten(2).norm(dim=2)/u0.norm(dim=(1,2)).clamp(min=1e-12)[:,None]
        assert (relative<=.25001).all()
        replay_errors=dict(F=float(np.max(np.abs(pf.cpu().numpy()-old_raw[rows]))),rho=float(np.max(np.abs(rho.cpu().numpy()-old_rho[rows]))),
            p0=float((p0new-p0).abs().max()),old_path=float(np.max(np.abs(po.cpu().numpy()-old_path[rows]))),
            teacher_path=float(np.max(np.abs(pt.cpu().numpy()-teacher_path[rows]))),native=float(np.max(np.abs(po[:,-1].cpu().numpy()-native[rows]))))
        assert max(replay_errors.values())<=1e-6,replay_errors
        values=dict(row=np.asarray(rows),fold=fold[rows],video=videos[rows],role=roles[rows],valid_lengths=b['mask'].sum(-1),pf=pf,p0=p0,mu=target,
            rho_old=rho,native_prediction=po[:,-1],old_path=po,teacher_path=pt,lambda_value=lam,cap=cap,target_old=told,target_cal=tcal,
            original_messages=m,message_scale=scale.expand(len(rows),-1,-1),normalized_radius=radius.flatten(),old_shift=so,teacher_shift=st,
            old_unit=do,teacher_unit=dt,candidate_messages=messages,candidate_prediction=pred,candidate_valid=valid,candidate_duplicate=duplicates,
            relative_message_shift=relative,selected_index=torch.stack(indices,1),selected_prediction=torch.stack(selected,1),selected_replay=torch.stack(replays,1))
        return {k:v.detach().cpu().numpy() if isinstance(v,torch.Tensor) else v for k,v in values.items()},replay_errors
    evidence=[];checks=[];max_errors={}
    for rows in [np.arange(32),np.asarray([620,621]),np.asarray([1280])]:
        t=time.monotonic();values,errors=inspect(rows);changed,_=inspect(rows,True)
        assert all(np.array_equal(values[k],changed[k]) for k in values),'poisoned label access'
        if 620 in rows:assert values['valid_lengths'][list(rows).index(620)]==1
        evidence.append(dict(rows=rows.tolist(),label_replacement_max_error=0,seconds=time.monotonic()-t,replay_errors=errors))
        checks.append(values)
    pair,_=inspect(np.asarray([620,621]));single,_=inspect(np.asarray([620]))
    coupling=float(np.max(np.abs(pair['candidate_prediction'][0]-single['candidate_prediction'][0])))
    assert coupling<=1e-6
    pp=torch.tensor([[0.,.2,-.2]],device='cuda');ff=pp[:,0];vv=torch.ones_like(pp,dtype=torch.bool)
    for vv0,force in [(vv,torch.ones(1,device='cuda',dtype=torch.bool)),(torch.tensor([[True,False,False]],device='cuda'),None)]:
        assert selector(pp,ff,torch.tensor([.5],device='cuda'),vv0,[0,1,2],force)[0].item()==0
    assert selector(torch.zeros_like(pp),ff,ff,vv,[0,1,2])[0].item()==0
    zero_cap=(pp.double()-ff.double()[:,None]).abs()<=0
    assert selector(pp,ff,torch.tensor([.5],device='cuda'),zero_cap,[0,1,2])[0].item()==0
    # One-token double clone FD-HVP on the anchored new acceptance target; no model update.
    dm=copy.deepcopy(model).double();dm.requires_grad_(False);rows=np.asarray([620,621])
    bd={k:torch.as_tensor(v[rows],device='cuda').double() if np.issubdtype(v.dtype,np.floating) else torch.as_tensor(v[rows],device='cuda') for k,v in inputs.items()}
    with torch.no_grad():
        pool=bd['pooled_state'];mm=torch.stack([dm.donor[i](torch.cat([pool[:,r],pool[:,d],.5*(pool[:,r]-pool[:,d]),.5*(pool[:,r]+pool[:,d])],-1)) for i,(r,d) in enumerate(PAIRS)],1)
        sc=dm.feedback.train_message_rms[None,:,None];u0=(mm/sc).detach()
        pf0=dm.terminal(bd['state'],bd['mask'],dm.feedback.context(bd['old_context'],mm)).detach()
        lm=torch.as_tensor(lambdas[fold[rows]],device='cuda');tar=(pf0+lm*(torch.as_tensor(mu[rows],device='cuda')-pf0)).detach()
    direction=torch.arange(1,u0.numel()+1,device='cuda',dtype=torch.float64).reshape_as(u0);direction/=direction.norm()
    def dg(u,hessian=False):
        u=u.detach().requires_grad_();pr=dm.terminal(bd['state'],bd['mask'],dm.feedback.context(bd['old_context'],u*sc))
        loss=.5*(u-u0).square().sum()+dm.feedback.finite_beta*((pr-tar).square()/dm.feedback.train_residual_rms.square()).sum()
        g=torch.autograd.grad(loss,u,create_graph=hessian)[0]
        return (g,torch.autograd.grad((g*direction).sum(),u)[0]) if hessian else g
    _,h=dg(u0,True);fd=[]
    for step in [1e-4,1e-5]:
        estimate=(dg(u0+step*direction)-dg(u0-step*direction))/(2*step)
        err=float((estimate-h).norm()/(1+h.norm()));assert torch.isfinite(estimate).all() and err<1e-4
        fd.append(dict(step=step,relative_error=err))
    del dm
    chunks={}
    def append(v):
        for k,x in v.items():chunks.setdefault(k,[]).append(x)
    if a.phase=='precheck':
        for v in checks:append(v)
    else:
        for first in range(0,1281,32):
            rows=np.arange(first,min(first+32,1281));v,err=inspect(rows);append(v)
            for k,x in err.items():max_errors[k]=max(max_errors.get(k,0),x)
            assert torch.cuda.max_memory_allocated()<=plan['maximum_peak_allocated_bytes']
            assert time.monotonic()-began<plan['maximum_seconds']
            write(out/'progress.json',dict(utc=utc(),rows=int(rows[-1])+1,pid=os.getpid(),complete=False))
    merged={k:np.concatenate(x) for k,x in chunks.items()}
    path=out/'predictions_frozen.npz';np.savez_compressed(path,**merged)
    assert path.stat().st_size<=plan['maximum_new_array_bytes']
    after=helpers.tensor_sha(model.state_dict());assert before==after and all(p.grad is None for p in model.parameters())
    peak=torch.cuda.max_memory_allocated();seconds=time.monotonic()-began
    assert peak<=plan['maximum_peak_allocated_bytes'] and seconds<plan['maximum_seconds']
    pessimistic=max(x['seconds'] for x in evidence)*41*2
    assert pessimistic<plan['maximum_seconds']
    write(out/'prediction_freeze.json',dict(utc=utc(),sha256=sha(path),bytes=path.stat().st_size,real_labels_read=False,all_eight_arms_frozen=True))
    write(out/'receipt.json',dict(passed=True,phase=a.phase,utc=utc(),pid=os.getpid(),argv=sys.argv,source_sha256=sha(__file__),plan_sha256=sha(a.plan),
        model_state_before=before,model_state_after=after,no_parameter_gradients=True,optimizer_steps=0,optimizer_created=False,
        rows=len(merged['row']),videos=len(np.unique(merged['video'])),prediction_sha256=sha(path),array_bytes=path.stat().st_size,
        actual_peak_allocated_bytes=peak,seconds=seconds,pessimistic_seconds=pessimistic,evidence=evidence,maximum_replay_errors=max_errors,
        selected_message_terminal_replay_max_error=float(np.max(np.abs(merged['selected_prediction']-merged['selected_replay']))),
        single_row_batch_replay_max_error=coupling,lambda_zero_and_all_invalid_and_zero_cap_tie_exact_F=True,double_finite_difference_hvp=fd,
        shared_nominal_candidate_terminal_calls=13,duplicate_messages_identified_exactly=True,duplicates_forwarded_at_fixed_batch_shape=True,
        real_labels_read=False,dev_requested=False,test_requested=False,whole_pipeline_crossfit=False))
    print('MATCHED_MESSAGE_POOLS_NATURAL_COMPLETE',a.phase,len(merged['row']),seconds,peak,flush=True)
if __name__=='__main__':main()
