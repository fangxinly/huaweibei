"""Prospective runtime: cannot launch from a preparation protocol.

FIT drives updates, INNER selects strict earliest minimum, OUTER only supplies
dummy-label inputs. All ten saved OUTER arrays must pass storage/CPU gates before
the separate pooled scorer may expose outer labels.
"""
import argparse
import datetime as dt
import json
import importlib.metadata
import math
import os
import pickle
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from fold_contract import sha,batches,inner_mse,validate_folds


def utc():return dt.datetime.now(dt.timezone.utc)


def write(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(path)


def validate_samefold_precheck(stage,method,fold,precheck_fold,original,current):
    """Compare a fresh initial state only with a precheck of the same fold."""
    if stage!='train' or fold!=precheck_fold:
        return False
    if original['method']!=method or original['fold']!=fold:
        raise RuntimeError('Precheck method/fold identity differs')
    for field in ('clean_initial_state_sha256','initial_rng_sha256'):
        if original[field]!=current[field]:
            raise RuntimeError('Fresh training initialization/RNG differs from same-fold precheck')
    return True


def validate(args):
    if sha(args.protocol)!=args.protocol_sha:raise ValueError('Exact protocol SHA required')
    plan=json.loads(args.protocol.read_text(encoding='utf-8'))
    permitted=(plan['status']=='GROUP5_EXECUTION_FROZEN' or
               (plan['status']=='GROUP5_PRECHECK_ONLY_FROZEN' and args.stage=='precheck'))
    if not permitted or not plan['execution_enabled']:
        raise PermissionError('Local candidate has not passed execution/storage gates')
    if plan.get('authorized_stages') and args.stage not in plan['authorized_stages']:
        raise PermissionError('This stage has not been frozen')
    if plan.get('authorized_methods') and args.method not in plan['authorized_methods']:
        raise PermissionError('This method has not been frozen')
    if not plan['human_asset_and_lease_budget_reference'] or not plan['complete_storage_reservation_reference']:
        raise PermissionError('Trusted asset/budget and complete storage reservation absent')
    if sha(args.bundle/'split.json')!=plan['split_sha256']:raise ValueError('Split digest differs')
    split=json.loads((args.bundle/'split.json').read_text(encoding='utf-8'))
    validate_folds(split['canonical_row_ids'],split['folds'])
    for rel,h in plan['order_sha256'].items():
        if sha(args.bundle/rel)!=h:raise ValueError('Frozen order differs: '+rel)
    for rel,h in plan['source_sha256'].items():
        if sha(args.bundle/rel)!=h:raise ValueError('Frozen source differs: '+rel)
    for rel,h in plan['asset_sha256'].items():
        if sha(args.assets/rel)!=h:raise ValueError('Public asset differs: '+rel)
    expected_python=str(args.assets/'.venv/bin/python')
    if sys.executable!=expected_python:raise ValueError('Pinned native runtime required')
    if 'runtime_exact_versions' in plan:
        actual_versions={n:importlib.metadata.version(n) for n in plan['runtime_exact_versions']}
        if actual_versions!=plan['runtime_exact_versions']:raise PermissionError('Native dependencies differ')
    if (args.root.exists() or args.method not in plan['methods'] or args.fold not in range(5)):
        raise ValueError('Fresh root, frozen method/fold required')
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    process=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)
    if uuid!=plan['assigned_gpu_uuid'][args.node] or compute.strip():
        raise PermissionError('Fresh physical identity/empty compute gate failed')
    if shutil.disk_usage(args.root.parent).free<plan['remote_free_floor_bytes']:
        raise PermissionError('Fresh remote saving space absent')
    end=dt.datetime.fromisoformat(plan['conservative_lease_end_UTC'])
    if end>dt.datetime(2026,10,7,13,30,tzinfo=dt.timezone.utc):raise PermissionError('Unconfirmed lease extension')
    remaining=(end-utc()).total_seconds()
    if remaining<plan['remaining_queue_execution_budget_seconds']+7200:
        raise PermissionError('Whole queue plus two-hour saving reserve absent')
    if args.stage=='train':
        for method in plan['methods']:
            parent=plan['precheck_D_other_CPU_joint'][method]
            if sha(parent['path'])!=parent['sha256']:raise ValueError('Precheck storage/CPU parent differs')
            receipt=json.loads(Path(parent['path']).read_text())
            if receipt['status']!='GROUP5_PRECHECK_D_OTHER_CPU_COMPLETE':raise PermissionError('Actual paired precheck missing')
    return plan,{'actual_utc':utc().isoformat(),'uuid':uuid,'compute':compute,'full_process_table':process,
                 'fullargv':[sys.executable]+sys.argv,'remaining_seconds':remaining,
                 'runtime_exact_versions':actual_versions if 'runtime_exact_versions' in plan else None,
                 'remote_free_bytes':shutil.disk_usage(args.root.parent).free}


def run(args):
    plan,physical=validate(args)
    args.root.mkdir();out=args.root/'out';out.mkdir()
    write(out/'actual_physical_preflight.json',physical)
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from fold_session import construct,loss,prediction
    from fixed_flow_components_candidate import tensor_sha
    from anchored_flow import synthetic_check
    torch.set_num_threads(2)
    if args.stage=='precheck':
        write(out/'synthetic_flow_check.json',synthetic_check())
    split=json.loads((args.bundle/'split.json').read_text(encoding='utf-8'))
    session=construct(args.method,args.bundle,args.assets,plan,split,args.fold)
    fold=split['folds'][args.fold]
    orders=np.load(args.bundle/f'orders_fold{args.fold}.npy',allow_pickle=False)
    if orders.shape!=(100,fold['rows']['fit']):raise ValueError('Shared order shape differs')
    for order in orders:batches([int(i) for i in order])
    np.save(out/'original_shared_FIT_orders.npy',orders,allow_pickle=False)
    np.save(out/'original_FIT_supervision.npy',session.fit.tensors[3].numpy(),allow_pickle=False)
    write(out/'original_role_identity.json',fold)
    def cpu(x):
        if torch.is_tensor(x):return x.detach().cpu().clone()
        if isinstance(x,dict):return {k:cpu(v) for k,v in x.items()}
        if isinstance(x,list):return [cpu(v) for v in x]
        if isinstance(x,tuple):return tuple(cpu(v) for v in x)
        return x
    def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),
                      'torch':torch.get_rng_state().clone(),'cuda':[v.clone() for v in torch.cuda.get_rng_state_all()]}
    def rng_digest():
        x=rng();x['torch']=x['torch'].tolist();x['cuda']=[v.tolist() for v in x['cuda']]
        import hashlib
        return hashlib.sha256(pickle.dumps(x,protocol=4)).hexdigest()
    def save(path,value):
        tmp=path.with_suffix(path.suffix+'.tmp');torch.save(value,tmp);tmp.replace(path)
        return {'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    def predict(role,dummy=0):
        session.model.eval();items=[]
        with torch.no_grad():
            tensors=session.inputs[role].tensors
            for start in range(0,len(tensors[0]),128):
                batch=[x[start:start+128].to(session.author.DEVICE) for x in tensors]
                batch[3]=torch.full_like(batch[3],dummy)
                items.append(prediction(session,args.method,tuple(batch)).detach().cpu().numpy())
        p=np.concatenate(items).astype(np.float32)
        if p.shape!=(fold['rows'][role],) or not np.isfinite(p).all():raise ValueError('Invalid prediction')
        return p
    def peak():
        torch.cuda.synchronize()
        p={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        if max(p.values())>6*1024**3:raise RuntimeError('Cumulative six-GiB budget exceeded')
        return p
    def step(rows):
        session.model.train();session.optimizer.zero_grad(set_to_none=True)
        batch=tuple(x[rows].to(session.author.DEVICE) for x in session.fit.tensors)
        torch.cuda.synchronize();start=time.perf_counter();value,parts=loss(session,args.method,batch)
        if not torch.isfinite(value):raise RuntimeError('Nonfinite training loss')
        value.backward()
        missing=[]
        for n,p in session.model.named_parameters():
            if p.grad is None:missing.append(n)
            elif not torch.isfinite(p.grad).all():raise RuntimeError('Nonfinite gradient: '+n)
        permitted={'dberta.pooler.dense.weight','dberta.pooler.dense.bias'} if args.method=='careflow' else set()
        if set(missing)!=permitted:raise RuntimeError('Unexpected missing gradient tensors: '+str(missing))
        torch.nn.utils.clip_grad_norm_(session.model.parameters(),1.)
        session.optimizer.step();session.scheduler.step();torch.cuda.synchronize()
        return {'seconds':time.perf_counter()-start,'objective':float(value.detach()),
                'parts':{k:float(v.detach()) for k,v in parts.items()},'peak':peak(),'missing_gradients':missing}
    receipt={'method':args.method,'fold':args.fold,'rows':fold['rows'],'protocol_sha256':args.protocol_sha,
             'source_sha256':plan['source_sha256'],'split_sha256':plan['split_sha256'],
             'argv':[sys.executable]+sys.argv,'pid':os.getpid(),'public_encoder_matched_tensors':session.public_encoder_matched_tensors,
             'clean_initial_state_sha256':session.clean_state_sha256,'initial_rng_sha256':rng_digest(),
             'old_task_weights_used':False,'outer_labels_read':False}
    receipt['orders_sha256']=sha(out/'original_shared_FIT_orders.npy')
    write(out/'actual_construction.json',receipt)
    history=[];updates=0
    if args.stage=='train' and args.fold==plan['precheck_fold']:
        parent=json.loads(Path(plan['precheck_D_other_CPU_joint'][args.method]['path']).read_text())
        original=parent['precheck_receipt']
        validate_samefold_precheck(args.stage,args.method,args.fold,plan['precheck_fold'],original,receipt)
    if args.stage=='precheck' and args.fold!=plan['precheck_fold']:
        raise ValueError('Only the fixed precheck fold is authorized')
    names={id(p):n for n,p in session.model.named_parameters()}
    optimizer_names={str(index):names[id(p)] for live,saved in zip(session.optimizer.param_groups,session.optimizer.state_dict()['param_groups'])
                     for p,index in zip(live['params'],saved['params'])}
    def checkpoint(best_state=None,best_p=None):
        return {'metadata':dict(receipt,optimizer_steps=updates,history=history),
                'model':cpu(session.model.state_dict()),'optimizer':cpu(session.optimizer.state_dict()),
                'scheduler':cpu(session.scheduler.state_dict()),'rng':rng(),'optimizer_index_to_name':optimizer_names,
                'orders':orders,'fit_ids':fold['row_ids']['fit'],'inner_ids':fold['row_ids']['inner'],
                'statistics':cpu(session.stats),'selected_model':best_state,'selected_inner_prediction':best_p}
    if args.stage=='precheck':
        first=batches(orders[0].tolist())
        records=[step(rows) for rows in [first[0],first[1],first[-1]]];updates=3
        state=tensor_sha(session.model.state_dict());before=rng_digest()
        p0=predict('inner',0);p7=predict('inner',7)
        if not np.array_equal(p0,p7) or tensor_sha(session.model.state_dict())!=state or before!=rng_digest():
            raise RuntimeError('Dummy-label/all-state/RNG invariance failed')
        np.savez(out/'precheck_dummy_predictions.npz',row_ids=np.asarray(fold['row_ids']['inner']),
                 dummy0=p0,dummy7=p7,model_state_sha256=np.asarray(state))
        saved=save(out/'precheck_full.pt',checkpoint())
        session.model.load_state_dict(torch.load(saved['path'],map_location='cpu')['model'],strict=True)
        if not np.array_equal(predict('inner'),p0):raise RuntimeError('Complete-checkpoint replay differs')
        write(out/'actual_stage_receipt.json',dict(receipt,status='GROUP5_PRECHECK3_COMPLETE_CPU_STORAGE_PENDING',
                 actual_utc=utc().isoformat(),steps=records,complete_checkpoint=saved,peak=peak(),
                 inner_labels_read=False,outer_labels_read=False,
                 projection_seconds_per_fold=plan['fold_budgets'][args.fold]['updates']*max(r['seconds'] for r in records)*1.5+1200))
        return
    best=float('inf');best_epoch=0;best_state=None;best_p=None
    for epoch in range(100):
        records=[]
        for n,rows in enumerate(batches(orders[epoch].tolist()),1):
            record=step(rows);updates+=1;records.append(record)
            with (out/'actual_FIT_steps.jsonl').open('a',encoding='utf-8') as f:
                f.write(json.dumps(dict(record,epoch=epoch+1,batch=n,rows=rows,updates=updates))+'\n')
        state=tensor_sha(session.model.state_dict());before=rng_digest();p=predict('inner')
        if tensor_sha(session.model.state_dict())!=state or before!=rng_digest():raise RuntimeError('Evaluation mutated state/RNG')
        path=out/f'INNER_epoch{epoch+1:03d}.npz'
        np.savez(path,row_ids=np.asarray(fold['row_ids']['inner']),prediction=p,model_state_sha256=np.asarray(state))
        digest=sha(path)
        with (out/'prediction_freeze.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps({'actual_utc':utc().isoformat(),'path':str(path),'sha256':digest,'state_sha256':state})+'\n')
        y=session.guard.inner_labels(path,digest,state)
        if epoch==0:
            np.save(out/'original_INNER_selection_labels.npy',y,allow_pickle=False)
        elif not np.array_equal(np.load(out/'original_INNER_selection_labels.npy',allow_pickle=False),y):
            raise RuntimeError('INNER labels changed')
        score=inner_mse(p,y)
        if score<best:
            best,best_epoch,best_state,best_p=score,epoch+1,cpu(session.model.state_dict()),p.copy()
        history.append({'epoch':epoch+1,'updates':updates,'FIT_objective':float(np.mean([r['objective'] for r in records])),
                        'INNER_selection_MSE':score,'best_epoch':best_epoch,'best_mse':best,
                        'inner_prediction_sha256':digest,'state_sha256':state})
        write(out/'history.json',history);write(out/'progress.json',history[-1])
        # Mutable recovery checkpoint every ten epochs; all best states are kept
        # in memory and included in each save. Frozen evidence is never edited.
        if (epoch+1)%10==0:
            save(out/'resume_and_selected_full.pt',checkpoint(best_state,best_p))
        print(json.dumps(history[-1]),flush=True)
    if updates!=plan['fold_budgets'][args.fold]['updates'] or session.scheduler.last_epoch!=updates:
        raise RuntimeError('Update budget/scheduler differs')
    final_resume={'path':str(out/'resume_and_selected_full.pt'),'sha256':sha(out/'resume_and_selected_full.pt'),
                  'bytes':(out/'resume_and_selected_full.pt').stat().st_size}
    session.model.load_state_dict(best_state,strict=True);state=tensor_sha(session.model.state_dict())
    before=rng_digest();outer=predict('outer',0);outer7=predict('outer',7)
    if not np.array_equal(outer,outer7) or state!=tensor_sha(session.model.state_dict()) or before!=rng_digest():
        raise RuntimeError('OUTER label/state/RNG invariance failed')
    outer_path=out/'OUTER_prediction_only.npz'
    np.savez(outer_path,row_ids=np.asarray(fold['row_ids']['outer']),prediction=outer,model_state_sha256=np.asarray(state))
    np.savez(out/'OUTER_dummy_invariance.npz',dummy0=outer,dummy7=outer7,
             model_state_sha256=np.asarray(state))
    write(out/'actual_stage_receipt.json',dict(receipt,status='GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING',
          actual_utc=utc().isoformat(),optimizer_steps=updates,best_epoch=best_epoch,selected_state_sha256=state,
          complete_resume_and_selected=final_resume,outer_prediction_sha256=sha(outer_path),peak=peak(),
          guard_journal=session.guard.journal))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--protocol',type=Path,required=True);p.add_argument('--protocol-sha',required=True)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--node',choices=['A','B','C'],required=True)
    p.add_argument('--method',choices=['anchored_flow','careflow'],required=True)
    p.add_argument('--fold',type=int,required=True);p.add_argument('--stage',choices=['precheck','train'],required=True)
    run(p.parse_args())
