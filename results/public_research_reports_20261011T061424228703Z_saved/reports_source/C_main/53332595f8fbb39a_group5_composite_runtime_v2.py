"""Prepared runner with separately gated complete-epoch composite recovery.

No historical parent/cache is accepted. Cache labels are FIT-only. A separate
native/CPU/full-original transport qualification is required before training.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from group5_release_transport_v1 import digest,write


def utc():return dt.datetime.now(dt.timezone.utc)


def validate(a):
    if digest(a.plan)!=a.plan_sha:raise PermissionError('Exact composite plan differs')
    p=json.loads(a.plan.read_bytes())
    wanted='GROUP5_COMPOSITE_NATIVE_PRECHECK_FROZEN' if a.stage=='precheck' else 'GROUP5_COMPOSITE_TRAIN_FROZEN'
    if p.get('status')!=wanted or not p.get('execution_enabled'):
        raise PermissionError('Composite stage is not separately frozen for execution')
    if a.method not in ('anchored_message20','old_fixed_A') or p['method']!=a.method or p['fold']!=a.fold or p['old_task_weight_reuse']:
        raise PermissionError('Fresh same-fold composite scope differs')
    if sys.executable!=p['python'] or {n:importlib.metadata.version(n) for n in p['runtime_versions']}!=p['runtime_versions']:
        raise PermissionError('Exact native runtime differs')
    if a.out.exists():raise FileExistsError('Fresh stage root required')
    for base,mapping in ((a.bundle,p['source_SHA']),(a.assets,p['asset_SHA'])):
        for rel,h in mapping.items():
            if digest(base/rel)!=h:raise PermissionError('Exact source/assets differ: '+rel)
    for key in ('parent_plan','parent_qualification','parent_checkpoint'):
        if digest(p[key]['path'])!=p[key]['SHA']:raise PermissionError('Exact fresh parent provenance differs')
    if digest(a.bundle/'split.json')!=p['split_SHA'] or digest(a.bundle/f'orders_fold{a.fold}.npy')!=p['orders_SHA']:
        raise PermissionError('Exact split/order differs')
    capture=dict(actual_UTC=utc().isoformat(),GPU_UUID=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),
                 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True),
                 fullargv=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True),meminfo=Path('/proc/meminfo').read_text(),remote_free_bytes=shutil.disk_usage(a.out.parent).free)
    available=int(next(l.split()[1] for l in capture['meminfo'].splitlines() if l.startswith('MemAvailable:')))*1024
    if capture['GPU_UUID']!=p['GPU_UUID'] or capture['compute'].strip() or available<6*1024**3:
        raise PermissionError('Fresh physical compute/CPU RAM gate failed')
    if capture['remote_free_bytes']<p['remote_required_bytes'] or p['local_C_free_bytes']<200*1024**2 or p['local_D_free_bytes']<40*1024**2:
        raise PermissionError('Original preservation space absent')
    if not 0<=(utc()-dt.datetime.fromisoformat(p['local_space_capture_UTC'])).total_seconds()<=300:
        raise PermissionError('Actual local staging observation stale')
    if (dt.datetime.fromisoformat(p['lease_end_UTC'])-utc()).total_seconds()<p['stage_budget_seconds']+max(7200,p['saving_budget_seconds']):
        raise PermissionError('Measured stage plus saving reserve exceeds lease')
    if a.stage=='train' and not all(p.get(k) for k in ('same_method_fold_native_CPU_qualified','Release_range_restore_qualified','fresh_addon_initial_state_qualified')):
        raise PermissionError('Same-method/fold native/CPU/full transport qualification absent')
    if a.stage=='train' and digest(p['native_precheck_receipt'])!=p['native_precheck_receipt_SHA']:
        raise PermissionError('Original native-precheck receipt changed')
    from group5_composite_resume_contract_v2 import qualify
    recovery=qualify(p,a.method,a.fold,a.stage)  # must pass before token creation and scientific imports
    token=Path(p['new_once_token']);token.parent.mkdir(parents=True,exist_ok=True)
    with token.open('x',encoding='utf8') as f:json.dump(dict(actual_UTC=utc().isoformat(),plan_SHA=a.plan_sha,method=a.method,fold=a.fold,stage=a.stage),f)
    a.out.mkdir();write(a.out/'fresh_physical.json',capture)
    return p,recovery


def run(a):
    p,recovery=validate(a)
    from group5_composite_resume_contract_v2 import copy_completed_epoch_evidence
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from group5_fresh_direct_session_v1 import construct
    from group5_composite_components_v1 import qualified_parent,restore_parent,cache_message,cache_old_A,fit_old_A_scales,make_old_A_learner
    from fixed_flow_components_candidate import tensor_sha
    from fold_contract import batches,inner_mse
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    parent_method='anchored_parent' if a.method=='anchored_message20' else 'old_A_teacher'
    session=construct(parent_method,a.bundle,a.assets,p['parent_plan']['path'],p['parent_plan']['SHA'],a.fold)
    fold=session.fold;parent=qualified_parent(a.method,a.fold,fold,p['parent_qualification']['path'],p['parent_checkpoint']['path'])
    restore_parent(session,parent,p['parent_checkpoint']['path'])
    parent_SHA=tensor_sha(session.model.state_dict());device=session.author.DEVICE
    if a.stage=='precheck':
        if a.method=='anchored_message20':
            unused,cache=cache_message(session,a.out);del unused
            cache_names=('first_cache.npz','tail.pt')
        else:
            cache,rms,error=cache_old_A(session,a.out);scales=fit_old_A_scales(session.model.dberta,cache['fit'])
            write(a.out/'FIT_scales.json',dict(scales,first_coordinate_replay_max_error=error))
            cache_names=('fit_cache.npz','inner_cache.npz','outer_cache.npz','FIT_gradient_rms.npy','FIT_scales.json')
    else:
        reference=p['cache_reference'];cache_names=tuple(reference['files_SHA'])
        if reference['parent_checkpoint_SHA']!=parent['checkpoint']['SHA'] or not reference.get('original_CPU_and_GitHub_restoration_verified'):
            raise PermissionError('Original cache parent/CPU/transport scope differs')
        for name,h in reference['files_SHA'].items():
            origin=Path(reference['root'])/name
            if digest(origin)!=h:raise PermissionError('Original cache bytes differ')
            shutil.copyfile(origin,a.out/name)
        if a.method=='anchored_message20':
            with np.load(a.out/'first_cache.npz',allow_pickle=False) as z:cache={k:z[k].copy() for k in z.files}
        else:
            cache={}
            for role in ('fit','inner','outer'):
                with np.load(a.out/(role+'_cache.npz'),allow_pickle=False) as z:cache[role]={k:z[k].copy() for k in z.files}
            rms=np.load(a.out/'FIT_gradient_rms.npy',allow_pickle=False);scales=json.loads((a.out/'FIT_scales.json').read_bytes())
    if tensor_sha(session.model.state_dict())!=parent_SHA:raise ValueError('Cache stage mutated selected parent')
    parent_state={k:v.detach().cpu().clone() for k,v in session.model.state_dict().items()}
    if a.method=='anchored_message20':
        for role in ('fit','inner','outer'):
            if cache[role+'_ids'].tolist()!=fold['row_ids'][role]:raise PermissionError('Cache exact role IDs differ')
    else:
        for role in ('fit','inner','outer'):
            if cache[role]['row_ids'].tolist()!=fold['row_ids'][role] or (role!='fit' and 'y' in cache[role]):raise PermissionError('Cache row/target scope differs')
        if not np.array_equal(cache['fit']['y'],session.fit.tensors[3].numpy().reshape(-1)):raise ValueError('Fresh FIT cache targets differ')
    # The historical message constructor consumes RNG before the addon. Preserve
    # that sequence, while the human-authorized new CV common seed is128.
    random.seed(128);np.random.seed(128);torch.manual_seed(128);torch.cuda.manual_seed_all(128)
    if a.method=='anchored_message20':
        from incremental_message import OriginalTail,IncrementalMessage,objective
        tail=OriginalTail();saved_tail=torch.load(a.out/'tail.pt',map_location='cpu')
        if saved_tail['parent_selected_state_SHA']!=parent_SHA:raise PermissionError('Frozen tail parent differs')
        tail.load_state_dict(saved_tail['state'],strict=True);tail.to(device);model=IncrementalMessage().to(device)
        parameters=[[v for v in model.parameters() if v.requires_grad]]
        optimizers=[torch.optim.AdamW(parameters[0],lr=.001,weight_decay=.01)]
        frozen_SHA=tensor_sha(tail.state_dict())
    else:
        model=make_old_A_learner(session.model.dberta,rms,scales);model.eval()
        parameters=[list(model.donor.parameters()),list(model.feedback.parameters())]
        optimizers=[torch.optim.AdamW(v,lr=1e-4,weight_decay=.01) for v in parameters]
        frozen_SHA=tensor_sha({k:v for k,v in model.state_dict().items() if k.startswith(('flow.','fusion.','predictor.'))})
    session.model.cpu();torch.cuda.empty_cache()
    epochs=20 if a.method=='anchored_message20' else 100
    orders=np.load(a.bundle/f'orders_fold{a.fold}.npy',allow_pickle=False)[:epochs].copy()
    if orders.shape!=(epochs,fold['rows']['fit']):raise ValueError('Complete order shape differs')
    for order in orders:batches(order.tolist())
    np.save(a.out/'FIT_orders.npy',orders,allow_pickle=False)
    target=session.fit.tensors[3].numpy().reshape(-1).copy();np.save(a.out/'FIT_targets.npy',target,allow_pickle=False)
    fit_mask=np.zeros(len(target),dtype=bool)
    if a.method=='old_fixed_A':fit_mask[scales['fit_rows']]=True
    def cpu(v):
        if torch.is_tensor(v):return v.detach().cpu().clone()
        if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return type(v)(cpu(x) for x in v)
        return v
    def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()])
    def rng_SHA():
        r=rng();r['torch']=r['torch'].tolist();r['cuda']=[v.tolist() for v in r['cuda']]
        return hashlib.sha256(pickle.dumps(r,protocol=4)).hexdigest()
    def combined(state=None):
        return {**{'parent.'+k:v for k,v in parent_state.items()},**{'addon.'+k:v for k,v in (state or model.state_dict()).items()}}
    initial_SHA=tensor_sha(combined());initial_rng=rng_SHA();history=[];updates=0
    mapping={id(v):n for n,v in model.named_parameters()};ownership=[]
    for opt in optimizers:
        ownership.append({str(i):mapping[id(v)] for g,sg in zip(opt.param_groups,opt.state_dict()['param_groups']) for v,i in zip(g['params'],sg['params'])})
    owned=[id(v) for group in parameters for v in group]
    if len(owned)!=len(set(owned)) or set(owned)!={id(v) for v in model.parameters() if v.requires_grad}:raise ValueError('Complete addon optimizer ownership differs')
    receipt=dict(method=a.method,fold=a.fold,exact_plan_SHA=recovery['origin_plan_SHA'] if recovery else a.plan_sha,exact_dispatch_plan_SHA=a.plan_sha,source_SHA=p['source_SHA'],split_SHA=p['split_SHA'],
                 public_pretraining_fresh_start=True,historical_task_weights_used=False,outer_labels_decoded=False,fit_ids=fold['row_ids']['fit'],inner_ids=fold['row_ids']['inner'],
                 parent_checkpoint_SHA=parent['checkpoint']['SHA'],parent_selected_state_SHA=parent_SHA,clean_initial_state_SHA=initial_SHA,initial_rng_SHA=initial_rng,
                 cache_SHA={n:digest(a.out/n) for n in cache_names},frozen_tail_or_teacher_SHA=frozen_SHA,argv=[sys.executable]+sys.argv,pid=os.getpid(),
                 CV_adaptations=['common seed128','frozen whole-FIT common tail-inclusive orders','whole-INNER FP64 strict-earliest MSE for old_fixed_A; message fixed20 has no INNER target access'])
    write(a.out/'construction.json',receipt)
    def fixed_state():
        return tensor_sha(tail.state_dict()) if a.method=='anchored_message20' else tensor_sha({k:v for k,v in model.state_dict().items() if k.startswith(('flow.','fusion.','predictor.'))})
    def forward(role,rows,dummy=0):
        if a.method=='anchored_message20':
            first,mask,base,slots=[torch.as_tensor(cache[role+'_'+k][rows],device=device) for k in ('first','mask','base','slots')]
            cx=model(slots);return tail(first,mask,base,cx),cx
        names=('state','mask','old_context','pooled_state','reference_prediction')
        b={k:torch.as_tensor(cache[role][k][rows],device=device) for k in names}
        b['y']=torch.full((len(rows),),dummy,device=device)  # inference target argument is intentionally unused
        if role=='fit':b['head_fit_mask']=torch.as_tensor(fit_mask[rows],device=device)
        return model(b,'fixed')[:2]
    def predict(role,dummy=0):
        model.eval();parts=[]
        with torch.no_grad():
            for start in range(0,fold['rows'][role],128):parts.append(forward(role,np.arange(start,min(start+128,fold['rows'][role])),dummy)[0].cpu().numpy())
        values=np.concatenate(parts).astype(np.float32)
        if values.shape!=(fold['rows'][role],) or not np.isfinite(values).all():raise ValueError('Invalid composite prediction')
        return values
    def step(rows):
        # Keep the original fixed teacher/tail in eval mode throughout addon fit.
        model.eval()
        for opt in optimizers:opt.zero_grad(set_to_none=True)
        if a.method=='old_fixed_A':
            total=epochs*len(batches(orders[0].tolist()));warmup=int(total*.1)
            for opt in optimizers:
                for g in opt.param_groups:g['lr']=1e-4*min((updates+1)/warmup,max(0.,(total-updates)/(total-warmup)))
        torch.cuda.synchronize();tick=time.perf_counter();pred,context=forward('fit',rows)
        y=torch.as_tensor(target[rows],device=device)
        if a.method=='anchored_message20':value,parts=objective(pred,y,context)
        else:
            main=(pred-y).square().mean();aux=model.feedback.residual_loss(context,torch.as_tensor(cache['fit']['reference_prediction'][rows],device=device)-y,torch.as_tensor(fit_mask[rows],device=device))
            value=main+.01*aux;parts=dict(main_mse=main.detach(),residual_aux=aux.detach())
        if not torch.isfinite(value):raise ValueError('Nonfinite FIT addon objective')
        value.backward()
        for group,opt in zip(parameters,optimizers):
            if any(v.grad is not None and not torch.isfinite(v.grad).all() for v in group):raise ValueError('Nonfinite addon gradient')
            if a.method=='anchored_message20' and any(v.grad is None for v in group):raise ValueError('Missing message gradient')
            norm=torch.nn.utils.clip_grad_norm_(group,1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
            opt.step();opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize();peak=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())
        if peak>6*1024**3:raise PermissionError('Prospective GPU gate failed')
        return dict(seconds=time.perf_counter()-tick,objective=float(value.detach()),parts={k:float(v) for k,v in parts.items()},peak_bytes=peak)
    def save(best_state=None,best_p=None):
        path=a.out/'complete_composite_full.pt';tmp=path.with_name(path.name+'.tmp')
        torch.save(dict(metadata=dict(receipt,updates=updates,history=history),parent_model=parent_state,model=cpu(model.state_dict()),
                        frozen_tail=cpu(tail.state_dict()) if a.method=='anchored_message20' else None,optimizers=[cpu(o.state_dict()) for o in optimizers],
                        optimizer_index_to_name=ownership,rng=cpu(rng()),orders=orders,statistics=cpu(session.stats),selected_model=best_state,
                        selected_inner_prediction=best_p,guard_journal=session.guard.journal,completed_epoch_recovery_only=True),tmp)
        tmp.replace(path);return dict(path=str(path),SHA=digest(path),bytes=path.stat().st_size)
    if a.method=='anchored_message20':
        values=predict('fit')
        if np.max(np.abs(values-cache['fit_p0']))>2e-5:
            raise ValueError('Zero-initial message does not reproduce the fresh selected parent cache')
        with torch.no_grad():
            slots=torch.as_tensor(cache['fit_slots'][:7],device=device)
            if not torch.equal(model(slots),torch.zeros_like(model(slots))):raise ValueError('Message initialization is not exactly zero')
    if a.stage=='precheck':
        first=batches(orders[0].tolist());records=[]
        for rows in (first[0],first[1],first[-1]):records.append(step(rows));updates+=1
        before=tensor_sha(combined());before_rng=rng_SHA();p0=predict('inner');p7=predict('inner',7)
        if not np.array_equal(p0,p7) or tensor_sha(combined())!=before or rng_SHA()!=before_rng or fixed_state()!=frozen_SHA:
            raise ValueError('Precheck dummy/state/RNG/frozen-parent invariance failed')
        np.savez(a.out/'INNER_dummy_predictions.npz',row_ids=np.asarray(fold['row_ids']['inner']),dummy0=p0,dummy7=p7,model_state_sha256=np.asarray(before))
        saved=save();loaded=torch.load(saved['path'],map_location='cpu');model.load_state_dict(loaded['model'],strict=True)
        for opt,state in zip(optimizers,loaded['optimizers']):opt.load_state_dict(state)
        if tensor_sha(combined())!=before or not np.array_equal(predict('inner'),p0):raise ValueError('Full composite reload differs')
        write(a.out/'actual_stage_receipt.json',dict(receipt,status='COMPOSITE_NATIVE_PRECHECK3_EXIT_PENDING_CPU_TRANSPORT',actual_UTC=utc().isoformat(),updates=updates,
              after_state_SHA=before,checkpoint=saved,steps=records,inner_labels_decoded=False,projected_fit_seconds=epochs*len(first)*max(r['seconds'] for r in records)*1.5,guard_journal=session.guard.journal))
        return
    pre=json.loads(Path(p['native_precheck_receipt']).read_bytes())
    if any(pre[k]!=receipt[k] for k in ('method','fold','source_SHA','split_SHA','clean_initial_state_SHA','initial_rng_SHA','parent_checkpoint_SHA','cache_SHA')):
        raise PermissionError('Fresh composite differs from qualified native initialization/cache')
    best=float('inf');best_epoch=0;best_state=None;best_p=None;start_epoch=0
    if recovery:
        from group5_composite_epoch_resume_v2 import restore
        original=torch.load(p['resume']['checkpoint']['path'],map_location='cpu')
        restored=restore(model,optimizers,tail if a.method=='anchored_message20' else None,session,
                         original,p['source_SHA'],recovery['origin_plan_SHA'],orders,ownership,parent_SHA,
                         receipt['cache_SHA'],receipt)
        if restored['updates']!=recovery['restored_updates'] or restored['completed_epochs']!=recovery['completed_epochs']:
            raise PermissionError('Original CPU evidence and loaded complete epoch disagree')
        history=restored['history'];updates=restored['updates'];start_epoch=restored['completed_epochs']
        best=restored['best_MSE'];best_epoch=restored['best_epoch']
        best_state=restored['selected_model'];best_p=restored['selected_inner_prediction']
        copy_completed_epoch_evidence(p['resume'],history,a.method,a.out)
        receipt['recovery']=dict(recovery,reconstruction_journal=restored['reconstruction_journal'],
                                 reconstruction_is_extra_cost=True,prior_token_not_reused=True)
        write(a.out/'construction.json',receipt);write(a.out/'history.json',history)
        del original,restored
    for epoch in range(start_epoch,epochs):
        for number,rows in enumerate(batches(orders[epoch].tolist()),1):
            row=step(rows);updates+=1
            with (a.out/'FIT_steps.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(row,epoch=epoch+1,batch=number,updates=updates,rows=rows))+'\n')
        state=tensor_sha(combined());row=dict(epoch=epoch+1,updates=updates,state_SHA=state)
        if a.method=='old_fixed_A':
            before_rng=rng_SHA();values=predict('inner')
            if tensor_sha(combined())!=state or rng_SHA()!=before_rng:raise ValueError('INNER eval mutated state/RNG')
            path=a.out/f'INNER_epoch{epoch+1:03d}.npz';np.savez(path,row_ids=np.asarray(fold['row_ids']['inner']),prediction=values,model_state_sha256=np.asarray(state))
            frozen=digest(path);write(a.out/f'INNER_epoch{epoch+1:03d}_freeze.json',dict(actual_UTC=utc().isoformat(),SHA=frozen,state_SHA=state))
            score=inner_mse(values,session.guard.inner_labels(path,frozen,state))
            if score<best:best,best_epoch,best_state,best_p=score,epoch+1,cpu(model.state_dict()),values.copy()
            row.update(inner_MSE=score,best_epoch=best_epoch,best_MSE=best,prediction_SHA=frozen)
        elif epoch+1==epochs:best_epoch=epochs;best_state=cpu(model.state_dict())
        if fixed_state()!=frozen_SHA:raise ValueError('Frozen teacher/tail mutated during addon fit')
        history.append(row);write(a.out/'history.json',history);save(best_state,best_p);print(json.dumps(row),flush=True)
    if updates!=p['updates']:raise ValueError('Fixed composite budget differs')
    model.load_state_dict(best_state,strict=True);state=tensor_sha(combined());before_rng=rng_SHA();p0=predict('outer');p7=predict('outer',7)
    if not np.array_equal(p0,p7) or tensor_sha(combined())!=state or rng_SHA()!=before_rng:raise ValueError('OUTER dummy/state/RNG invariance failed')
    path=a.out/'OUTER_prediction_only.npz';np.savez(path,row_ids=np.asarray(fold['row_ids']['outer']),prediction=p0,model_state_sha256=np.asarray(state))
    write(a.out/'actual_stage_receipt.json',dict(receipt,status='COMPOSITE_COMPLETE_OUTER_UNSCORED_CPU_TRANSPORT_PENDING',actual_UTC=utc().isoformat(),updates=updates,
          best_epoch=best_epoch,best_inner_MSE=best if a.method=='old_fixed_A' else None,selected_state_SHA=state,checkpoint=dict(path=str(a.out/'complete_composite_full.pt'),SHA=digest(a.out/'complete_composite_full.pt'),bytes=(a.out/'complete_composite_full.pt').stat().st_size),
          outer_prediction_SHA=digest(path),guard_journal=session.guard.journal))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','bundle','assets','out'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--method',required=True);p.add_argument('--fold',type=int,required=True);p.add_argument('--stage',choices=('precheck','train'),required=True)
    a=p.parse_args();started=utc()
    try:run(a)
    except BaseException as exc:
        if a.out.exists():write(a.out/'failed_natural_exit.json',dict(actual_UTC=utc().isoformat(),started_UTC=started.isoformat(),exit=1,error_type=type(exc).__name__,message=str(exc)))
        raise
    else:write(a.out/'natural_exit.json',dict(actual_UTC=utc().isoformat(),started_UTC=started.isoformat(),exit=0))
