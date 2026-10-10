"""Fresh direct-stage native precheck and fixed100 training; OUTER remains unscored.

Exact frozen source/assets/runtime/resource plans precede scientific imports.
Prechecks cannot promote themselves into training. Every run consumes a distinct
new token, and interruption leaves a full epoch recovery state and original logs.
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
from group5_release_transport_v1 import digest,write,seal


def utc():return dt.datetime.now(dt.timezone.utc)


def validate(a):
    if digest(a.plan)!=a.plan_sha:raise PermissionError('Exact frozen plan differs')
    p=json.loads(a.plan.read_bytes())
    wanted='GROUP5_DIRECT_NATIVE_PRECHECK_FROZEN' if a.stage=='precheck' else 'GROUP5_DIRECT_TRAIN_FROZEN'
    if p.get('status')!=wanted or not p.get('execution_enabled'):
        raise PermissionError('Stage not explicitly frozen for native execution')
    if a.method not in p['authorized_methods'] or a.fold!=p['fold'] or p['old_task_weight_reuse']:
        raise PermissionError('Method/fold/source parent scope differs')
    if sys.executable!=p['python'] or {n:importlib.metadata.version(n) for n in p['runtime_versions']}!=p['runtime_versions']:
        raise PermissionError('Native runtime differs')
    if a.out.exists():raise FileExistsError('Fresh stage root required')
    for base,mapping in [(a.bundle,p['source_SHA']),(a.assets,p['asset_SHA'])]:
        for rel,h in mapping.items():
            if digest(base/rel)!=h:raise ValueError('Frozen bytes differ: '+rel)
    if digest(a.bundle/'split.json')!=p['split_SHA'] or digest(a.bundle/f'orders_fold{a.fold}.npy')!=p['orders_SHA']:
        raise ValueError('Frozen split/order bytes differ')
    capture=dict(actual_UTC=utc().isoformat(),GPU_UUID=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),
        compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True),
        fullargv=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True),meminfo=Path('/proc/meminfo').read_text(),
        remote_free_bytes=shutil.disk_usage(a.out.parent).free)
    available=int(next(l.split()[1] for l in capture['meminfo'].splitlines() if l.startswith('MemAvailable:')))*1024
    if capture['GPU_UUID']!=p['GPU_UUID'] or capture['compute'].strip() or available<6*1024**3:
        raise PermissionError('Fresh physical compute/CPU resource gate failed')
    if capture['remote_free_bytes']<p['remote_required_bytes'] or p['local_C_free_bytes']<200*1024**2 or p['local_D_free_bytes']<40*1024**2:
        raise PermissionError('Full-original saving space absent')
    captured=dt.datetime.fromisoformat(p['local_space_capture_UTC'])
    if not 0 <= (utc()-captured).total_seconds()<=300:
        raise PermissionError('Local staging observation stale')
    if (dt.datetime.fromisoformat(p['lease_end_UTC'])-utc()).total_seconds()<p['stage_budget_seconds']+max(7200,p['saving_budget_seconds']):
        raise PermissionError('Stage plus saving reserve exceeds conservative lease')
    if a.stage=='train' and not all(p.get(k) for k in ['same_method_fold_native_CPU_qualified','Release_range_restore_qualified','fresh_initial_state_qualified']):
        raise PermissionError('Training native/CPU/transport qualification absent')
    token=Path(p['new_once_token']);token.parent.mkdir(parents=True,exist_ok=True)
    with token.open('x',encoding='utf8') as f:json.dump(dict(actual_UTC=utc().isoformat(),stage=a.stage,method=a.method,fold=a.fold,plan_SHA=a.plan_sha),f)
    a.out.mkdir();write(a.out/'fresh_physical.json',capture)
    return p


def run(a):
    p=validate(a)
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import numpy as np
    import torch
    from group5_fresh_direct_session_v1 import construct,loss,prediction
    from fixed_flow_components_candidate import tensor_sha
    from fold_contract import batches,inner_mse
    torch.set_num_threads(2)
    session=construct(a.method,a.bundle,a.assets,a.plan,a.plan_sha,a.fold)
    orders=np.load(a.bundle/f'orders_fold{a.fold}.npy',allow_pickle=False)
    fold=session.fold
    if orders.shape!=(100,fold['rows']['fit']):raise ValueError('Actual order shape differs')
    for order in orders:batches(order.tolist())
    np.save(a.out/'FIT_orders.npy',orders,allow_pickle=False)
    np.save(a.out/'FIT_targets.npy',session.fit.tensors[3].numpy(),allow_pickle=False)
    def cpu(v):
        if torch.is_tensor(v):return v.detach().cpu().clone()
        if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
        if isinstance(v,(tuple,list)):return type(v)(cpu(x) for x in v)
        return v
    def rng():return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state().clone(),cuda=[x.clone() for x in torch.cuda.get_rng_state_all()])
    def rng_SHA():return hashlib.sha256(pickle.dumps(cpu(rng()),protocol=4)).hexdigest()
    def tensor_rng_SHA():
        r=rng();r['torch']=r['torch'].tolist();r['cuda']=[x.tolist() for x in r['cuda']]
        return hashlib.sha256(pickle.dumps(r,protocol=4)).hexdigest()
    def save(path,obj):
        tmp=path.with_name(path.name+'.tmp');torch.save(obj,tmp);tmp.replace(path)
        return dict(path=str(path),SHA=digest(path),bytes=path.stat().st_size)
    def predict(role,dummy=0):
        session.model.eval();parts=[]
        with torch.no_grad():
            tensors=session.inputs[role].tensors
            for start in range(0,len(tensors[0]),128):
                b=[x[start:start+128].to(session.author.DEVICE) for x in tensors];b[3]=torch.full_like(b[3],dummy)
                parts.append(prediction(session,tuple(b)).detach().cpu().numpy())
        values=np.concatenate(parts).astype(np.float32)
        if values.shape!=(fold['rows'][role],) or not np.isfinite(values).all():raise ValueError('Invalid role prediction')
        return values
    history=[];updates=0
    mapping={id(v):n for n,v in session.model.named_parameters()}
    opt_names={str(i):mapping[id(v)] for group,sg in zip(session.optimizer.param_groups,session.optimizer.state_dict()['param_groups']) for v,i in zip(group['params'],sg['params'])}
    receipt=dict(method=a.method,fold=a.fold,exact_plan_SHA=a.plan_sha,source_SHA=p['source_SHA'],split_SHA=p['split_SHA'],
        clean_initial_state_SHA=session.clean_state_SHA,public_encoder_matched=session.public_encoder_matched,
        initial_rng_SHA=tensor_rng_SHA(),historical_task_weights_used=False,public_pretraining_fresh_start=True,
        fit_ids=fold['row_ids']['fit'],inner_ids=fold['row_ids']['inner'],outer_labels_decoded=False,
        argv=[sys.executable]+sys.argv,pid=os.getpid(),
        exact_dispatch_plan_SHA=a.plan_sha)
    if p.get('resume'):
        if a.stage!='train':raise PermissionError('Precheck cannot resume')
        receipt['exact_plan_SHA']=p['resume']['origin_plan_SHA']
    write(a.out/'construction.json',receipt)
    def checkpoint(best_state=None,best_prediction=None):
        return dict(metadata=dict(receipt,updates=updates,history=history),model=cpu(session.model.state_dict()),
            optimizer=cpu(session.optimizer.state_dict()),scheduler=cpu(session.scheduler.state_dict()),rng=cpu(rng()),
            optimizer_index_to_name=opt_names,orders=orders,statistics=cpu(session.stats),selected_model=best_state,
            selected_inner_prediction=best_prediction,guard_journal=session.guard.journal)
    def step(rows):
        session.model.train();session.optimizer.zero_grad(set_to_none=True)
        batch=tuple(x[rows].to(session.author.DEVICE) for x in session.fit.tensors)
        torch.cuda.synchronize();started=time.perf_counter();value,parts=loss(session,batch)
        if not torch.isfinite(value):raise ValueError('Nonfinite FIT objective')
        value.backward();missing=[]
        for n,v in session.model.named_parameters():
            if not v.requires_grad:continue
            if v.grad is None:missing.append(n)
            elif not torch.isfinite(v.grad).all():raise ValueError('Nonfinite gradient: '+n)
        allowed={'dberta.pooler.dense.weight','dberta.pooler.dense.bias'} if a.method in ('careflow','old_A_teacher') else set()
        if set(missing)!=allowed:raise ValueError('Unexpected missing gradient: '+str(missing))
        # CaReFlow and original counterfactual teacher have no gradient clipping.
        if a.method not in ('careflow','old_A_teacher'):
            norm=torch.nn.utils.clip_grad_norm_(session.model.parameters(),1.)
            if not torch.isfinite(norm):raise ValueError('Nonfinite gradient norm')
        session.optimizer.step();session.scheduler.step();session.optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize()
        peak=max(torch.cuda.max_memory_allocated(),torch.cuda.max_memory_reserved())
        if peak>6*1024**3:raise PermissionError('SixGiB prospective GPU gate failed')
        return dict(seconds=time.perf_counter()-started,objective=float(value.detach()),parts={k:float(v.detach()) for k,v in parts.items()},peak_bytes=peak,missing_gradients=missing)
    if a.stage=='precheck':
        if a.method=='old_A_teacher':session.model.dberta.own_flow.set_epoch(1)
        first=batches(orders[0].tolist());records=[]
        for rows in (first[0],first[1],first[-1]):records.append(step(rows));updates+=1
        before=tensor_sha(session.model.state_dict());before_rng=tensor_rng_SHA()
        p0=predict('inner');p7=predict('inner',7)
        if not np.array_equal(p0,p7) or before!=tensor_sha(session.model.state_dict()) or before_rng!=tensor_rng_SHA():
            raise ValueError('Dummy label/state/RNG invariance failed')
        np.savez(a.out/'INNER_dummy_predictions.npz',row_ids=np.asarray(fold['row_ids']['inner']),dummy0=p0,dummy7=p7,model_state_sha256=np.asarray(before))
        saved=save(a.out/'precheck_full.pt',checkpoint())
        # Model/optimizer/scheduler full reload and exact inference restoration.
        loaded=torch.load(saved['path'],map_location='cpu')
        session.model.load_state_dict(loaded['model'],strict=True);session.optimizer.load_state_dict(loaded['optimizer']);session.scheduler.load_state_dict(loaded['scheduler'])
        if tensor_sha(session.model.state_dict())!=before or not np.array_equal(predict('inner'),p0):raise ValueError('Full checkpoint model restoration differs')
        del loaded
        write(a.out/'actual_stage_receipt.json',dict(receipt,status='NATIVE_PRECHECK3_EXIT_PENDING_CPU_TRANSPORT',actual_UTC=utc().isoformat(),
            steps=records,updates=updates,checkpoint=saved,inner_labels_decoded=False,outer_labels_decoded=False,
            after_state_SHA=before,projected_fit_seconds=p['updates']*max(r['seconds'] for r in records)*1.5))
        return
    pre=json.loads(Path(p['native_precheck_receipt']).read_bytes())
    if pre['method']!=a.method or pre['fold']!=a.fold or pre['clean_initial_state_SHA']!=receipt['clean_initial_state_SHA'] or pre['initial_rng_SHA']!=receipt['initial_rng_SHA']:
        raise PermissionError('Fresh samefold initialization differs from qualified native precheck')
    best=float('inf');best_epoch=0;best_state=None;best_p=None;start_epoch=0
    if p.get('resume'):
        spec=p['resume'];checkpoint_path=Path(spec['checkpoint'])
        if not all(spec.get(k) for k in ('original_checkpoint_CPU_qualified','original_GitHub_restoration_verified','partial_epoch_extra_cost_recorded')):
            raise PermissionError('Resume original preservation/CPU/cost gates absent')
        if digest(checkpoint_path)!=spec['checkpoint_SHA']:
            raise PermissionError('Resume original checkpoint SHA differs')
        from group5_epoch_resume_v1 import restore
        loaded=torch.load(checkpoint_path,map_location='cpu')
        resumed=restore(session,loaded,p['source_SHA'],spec['origin_plan_SHA'],orders,opt_names)
        history=resumed['history'];updates=resumed['updates'];start_epoch=resumed['completed_epochs']
        best=resumed['best_MSE'];best_epoch=resumed['best_epoch'];best_state=resumed['selected_model'];best_p=resumed['selected_inner_prediction']
        receipt['resume_ancestry']=dict(checkpoint_SHA=spec['checkpoint_SHA'],logical_updates_restored=updates,
            extra_partial_epoch_updates=spec['extra_partial_epoch_updates'],origin_root=spec['origin_root'])
        write(a.out/'resume_restoration.json',dict(actual_UTC=utc().isoformat(),**receipt['resume_ancestry'],
            reconstruction_journal=resumed['reconstruction_journal'],model_state_SHA=tensor_sha(session.model.state_dict()),rng_SHA=tensor_rng_SHA()))
        del loaded,resumed
    for epoch in range(start_epoch,100):
        if a.method=='old_A_teacher':session.model.dberta.own_flow.set_epoch(epoch+1)
        records=[]
        for n,rows in enumerate(batches(orders[epoch].tolist()),1):
            record=step(rows);updates+=1;records.append(record)
            with (a.out/'FIT_steps.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(dict(record,epoch=epoch+1,batch=n,rows=rows,updates=updates))+'\n')
        state=tensor_sha(session.model.state_dict());before_rng=tensor_rng_SHA();prediction_values=predict('inner')
        if tensor_sha(session.model.state_dict())!=state or tensor_rng_SHA()!=before_rng:raise ValueError('INNER eval mutated state/RNG')
        path=a.out/f'INNER_epoch{epoch+1:03d}.npz'
        np.savez(path,row_ids=np.asarray(fold['row_ids']['inner']),prediction=prediction_values,model_state_sha256=np.asarray(state))
        freeze=digest(path)
        write(a.out/f'INNER_epoch{epoch+1:03d}_freeze.json',dict(actual_UTC=utc().isoformat(),SHA=freeze,state_SHA=state))
        y=session.guard.inner_labels(path,freeze,state);mse=inner_mse(prediction_values,y)
        if mse<best:best,best_epoch,best_state,best_p=mse,epoch+1,cpu(session.model.state_dict()),prediction_values.copy()
        history.append(dict(epoch=epoch+1,updates=updates,inner_MSE=mse,best_epoch=best_epoch,best_MSE=best,prediction_SHA=freeze,state_SHA=state))
        write(a.out/'history.json',history)
        save(a.out/'resume_selected_full.pt',checkpoint(best_state,best_p))
        print(json.dumps(history[-1]),flush=True)
    if updates!=p['updates'] or session.scheduler.last_epoch!=updates:raise ValueError('Update budget differs')
    session.model.load_state_dict(best_state,strict=True)
    if a.method=='old_A_teacher':session.model.dberta.own_flow.set_epoch(best_epoch)
    before=tensor_sha(session.model.state_dict());before_rng=tensor_rng_SHA();p0=predict('outer');p7=predict('outer',7)
    if not np.array_equal(p0,p7) or before!=tensor_sha(session.model.state_dict()) or before_rng!=tensor_rng_SHA():raise ValueError('OUTER dummy/state/RNG invariance failed')
    path=a.out/'OUTER_prediction_only.npz'
    np.savez(path,row_ids=np.asarray(fold['row_ids']['outer']),prediction=p0,model_state_sha256=np.asarray(before))
    write(a.out/'actual_stage_receipt.json',dict(receipt,status='DIRECT100_COMPLETE_OUTER_UNSCORED_CPU_TRANSPORT_PENDING',actual_UTC=utc().isoformat(),
        updates=updates,best_epoch=best_epoch,best_inner_MSE=best,selected_state_SHA=before,
        checkpoint=dict(path=str(a.out/'resume_selected_full.pt'),SHA=digest(a.out/'resume_selected_full.pt'),bytes=(a.out/'resume_selected_full.pt').stat().st_size),outer_prediction_SHA=digest(path),guard_journal=session.guard.journal))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','bundle','assets','out'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--method',required=True);p.add_argument('--fold',type=int,required=True)
    p.add_argument('--stage',choices=['precheck','train'],required=True)
    a=p.parse_args();started=utc()
    try:
        run(a)
    except BaseException as exc:
        if a.out.exists():write(a.out/'failed_natural_exit.json',dict(actual_UTC=utc().isoformat(),started_UTC=started.isoformat(),exit=1,error_type=type(exc).__name__,message=str(exc)))
        raise
    else:
        write(a.out/'natural_exit.json',dict(actual_UTC=utc().isoformat(),started_UTC=started.isoformat(),exit=0))
