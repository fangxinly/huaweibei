"""Frozen 100-epoch order, gated 10-epoch feasibility prefix and exact continuation.

No runtime launch from local AST checks. This runner only trains one reference;
it does not establish donor superiority or authorize residual/head experiments.
"""
import argparse, datetime, hashlib, json, math, os, random, shutil, sys, time
from pathlib import Path
import numpy as np

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def utc():return datetime.datetime.now(datetime.timezone.utc)
def timestamp(s):return datetime.datetime.fromisoformat(s.replace('Z','+00:00'))
def tree_tuple(x):return tuple(tree_tuple(y) for y in x) if isinstance(x,list) else x

def validate(plan, evidence, now, phase):
    if phase not in ('stage10','continue100'):raise PermissionError('PHASE')
    if evidence.get('scope')!='MINIMAL_FIXED_STAGED_REFERENCE_V1':raise PermissionError('SCOPE')
    if not evidence.get('human_provenance_verified') or evidence.get('lease_source')!='DIRECT_HUMAN_NEW_P4_24H_20261006':raise PermissionError('HUMAN_PROVENANCE')
    if not 0 <= (now-timestamp(evidence['actual_query_utc'])).total_seconds() <= 300:raise PermissionError('FRESH_FIVE_MINUTES')
    if evidence.get('gpu_uuid')!='GPU-53696803-875e-eec8-2231-29db63579891' or evidence.get('compute_processes')!=[]:raise PermissionError('UUID_EMPTY_COMPUTE')
    if not isinstance(evidence.get('python_full_argv'),list):raise PermissionError('FULL_ARGV')
    if not evidence.get('complete_assets_and_source_verified') or not evidence.get('complete_GPU_D_B_precheck_verified'):raise PermissionError('PRECHECK_ASSETS_SOURCE')
    if evidence.get('remote_free_bytes',0)<12*1024**3 or evidence.get('permanent_D_free_bytes',0)<12*1024**3:raise PermissionError('TWELVE_GIB_PRESERVATION_SPACE')
    if evidence.get('plan_sha256')!=plan['_sha256']:raise PermissionError('PLAN_SHA')
    required=plan['conservative_100_projection_seconds']+7200
    if (timestamp(evidence['lease_end_utc'])-now).total_seconds()<required:raise PermissionError('LEASE_PROJECTION_TWO_HOURS_SAVE')
    if phase=='continue100' and not evidence.get('original_stage10_D_B_CPU_verified'):raise PermissionError('STAGE10_ACTUAL_PRESERVATION')

def run(args):
    bundle=Path(args.bundle); plan_path=bundle/'staged_reference_plan.json';plan=read(plan_path);plan['_sha256']=sha(plan_path)
    evidence=read(args.evidence);validate(plan,evidence,utc(),args.phase)
    for name,h in plan['source_sha256'].items():
        if sha(bundle/name)!=h:raise ValueError('SOURCE_SHA: '+name)
    for name,h in plan['role_order_sha256'].items():
        if sha(bundle/name)!=h:raise ValueError('ROLE_ORDER_SHA: '+name)
    parent=Path(plan['actual_precheck_root']); raw=read(parent/'out/actual_precheck_receipt.json');exit_record=read(parent/'natural_exit.json')
    if sha(parent/'out/actual_precheck_receipt.json')!=plan['actual_precheck_receipt_sha256'] or sha(parent/'natural_exit.json')!=plan['actual_precheck_exit_sha256']:raise ValueError('ORIGINAL_PRECHECK_SHA')
    if exit_record['exit_code']!=0 or not exit_record['natural_exit'] or raw['optimizer_steps']!=2 or raw['inner_labels_read']:raise ValueError('PRECHECK_RAW_SCOPE')
    if raw['strict_full_disk_replay_max_error']>1e-6 or raw['inner_dummy_label_replacement_max_error']!=0:raise ValueError('PRECHECK_REPLAY')
    initial=parent/'out/clean_initial_full.pt'
    if sha(initial)!=plan['clean_initial_full_sha256']:raise ValueError('CLEAN_INITIAL_FULL_SHA')
    for n,h in plan['clean_rng_sha256'].items():
        if sha(parent/'out'/n)!=h:raise ValueError('ORIGINAL_CLEAN_RNG_SHA: '+n)
    output=Path(args.out)
    if output.exists():raise FileExistsError('UNIQUE_NEW_RUN_NO_OVERWRITE')
    output.mkdir(parents=True)
    import torch
    from minimal_fixed_runtime_v1 import construct_public_candidate, tensor_sha, forward_batch, fit_objective, require_real_gradients
    import subprocess
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()!=evidence['gpu_uuid']:raise ValueError('DRIVER_UUID_CHANGED')
    started=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    lease=timestamp(evidence['lease_end_utc'])
    def budget(stage):
        elapsed=time.perf_counter()-started;allocated=torch.cuda.max_memory_allocated();reserved=torch.cuda.max_memory_reserved()
        report={'actual_utc':utc().isoformat(),'stage':stage,'elapsed_seconds':elapsed,'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,'remote_free_bytes':shutil.disk_usage('/data').free}
        with (output/'budget_trace.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        limit=1800 if args.phase=='stage10' else 18000
        if elapsed>limit or max(allocated,reserved)>6*1024**3:raise RuntimeError('STAGED_ACTUAL_GPU_TIME_BUDGET: '+stage)
        if (lease-utc()).total_seconds()<7200 or report['remote_free_bytes']<4*1024**3:raise RuntimeError('PRESERVATION_RESERVE: '+stage)
        return report
    session=construct_public_candidate(args.asset_base,bundle)
    text=session.model.dberta.model
    text.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
    assert text.encoder.gradient_checkpointing
    assert text.encoder._gradient_checkpointing_func.keywords=={'use_reentrant':False,'preserve_rng_state':True}
    del text
    assert session.construction_receipt['parameters']==185402807 and session.construction_receipt['parameter_tensors']==364
    assert tensor_sha(session.model.state_dict())==plan['clean_initial_state_sha256']
    assert len(session.optimizer.state)==0
    del session.clean_initial
    payload=torch.load(initial,map_location='cpu',weights_only=True)
    assert payload['metadata']['scope']=='CLEAN_INITIAL_NO_PRECHECK' and payload['metadata']['optimizer_steps']==0
    assert tensor_sha(payload['model'])==plan['clean_initial_state_sha256']
    session.model.load_state_dict(payload['model'],strict=True);del payload
    stats_sha=lambda:tensor_sha({n:v for n,v in session.model.state_dict().items() if n.startswith(('dberta.v6_audio_','dberta.v6_visual_'))})
    assert stats_sha()==plan['fit_statistics_sha256']
    orders=np.load(bundle/'fit_orders_seed91819_100.npy',allow_pickle=False)
    assert orders.shape==(100,695) and all(np.array_equal(np.sort(r),np.sort(session.guard.ids['fit'])) for r in orders)
    local={int(v):i for i,v in enumerate(session.guard.ids['fit'])}
    mapping=read(bundle/'train_row_video_mapping.json');videos=[mapping[int(i)]['video_id'] for i in session.guard.ids['inner']]
    assert len(set(videos))==4
    def rng_restore(t,extra):
        torch.set_rng_state(t['torch']);torch.cuda.set_rng_state_all(t['cuda'])
        random.setstate(tree_tuple(extra['python_random']))
        n=extra['numpy_global_random'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),int(n[2]),int(n[3]),float(n[4])))
    def rng_save():
        n=np.random.get_state()
        return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}, {'python_random':random.getstate(),'numpy_global_random':[n[0],n[1].tolist(),int(n[2]),int(n[3]),float(n[4])]}
    def row_batch(rows):
        index=torch.tensor([local[int(r)] for r in rows],dtype=torch.long)
        return tuple(x[index].to('cuda') for x in session.fit.tensors)
    def predict_inner(dummy=0.0):
        session.model.eval();pieces=[]
        # Deterministic iteration cannot consume training CPU/CUDA RNG.
        with torch.no_grad():
            for start in range(0,153,128):
                batch=[x[start:start+128].to('cuda') for x in session.inner_inputs.tensors]
                batch[3]=torch.full_like(batch[3],dummy);pieces.append(forward_batch(session.model,tuple(batch)).cpu().numpy())
        pred=np.concatenate(pieces)
        assert pred.shape==(153,) and np.isfinite(pred).all()
        return pred
    def cpu_state():return {n:v.detach().cpu().clone() for n,v in session.model.state_dict().items()}
    def cpu_tree(x):
        if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
        if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return type(x)(cpu_tree(v) for v in x)
        return x
    def save(path,data):
        with Path(path).open('xb') as f:torch.save(data,f)
        return {'file':str(path),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
    clean_torch=torch.load(parent/'out/clean_initial_torch_rng.pt',map_location='cpu',weights_only=True)
    clean_extra=read(parent/'out/clean_initial_python_numpy_rng.json')
    start_epoch=0;updates=0;history=[];best_metric=float('inf');best_epoch=None;best_state=None;best_pred=None
    if args.phase=='continue100':
        cp=torch.load(args.resume,map_location='cpu',weights_only=True)
        if sha(args.resume)!=evidence['stage10_resume_full_sha256'] or cp['metadata']['scope']!='STAGED_SHARED10_RESUME_NOT_COMPLETE100':raise ValueError('ACTUAL_STAGE10_SCOPE')
        m=cp['metadata'];assert m['plan_sha256']==plan['_sha256'] and m['epoch']==10 and m['optimizer_steps']==220
        assert tensor_sha(cp['model'])==m['state_sha256']
        session.model.load_state_dict(cp['model'],strict=True);session.optimizer.load_state_dict(cp['optimizer']);session.scheduler.load_state_dict(cp['scheduler'])
        start_epoch=10;updates=220;history=cp['history'];best_metric=cp['best_metric'];best_epoch=cp['best_epoch'];best_state=cp['best_state'];best_pred=np.asarray(cp['best_prediction'],dtype=np.float32)
        assert len(history)==10 and len(session.optimizer.state)==364 and session.scheduler.last_epoch==220
        rng_restore(cp['rng'],cp['rng_extra']);del cp
    else:
        rng_restore(clean_torch,clean_extra)
        assert torch.equal(torch.get_rng_state(),clean_torch['torch'])
        assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),clean_torch['cuda']))
    del clean_torch,clean_extra
    start_proof={'actual_utc':utc().isoformat(),'pid':os.getpid(),'argv':sys.argv,'phase':args.phase,'plan_sha256':plan['_sha256'],'start_epoch':start_epoch,'optimizer_steps':updates,'state_sha256':tensor_sha(session.model.state_dict()),'parameters':185402807,'parameter_tensors':364,'initial_optimizer_empty':args.phase=='stage10','precheck_optimizer_or_rng_not_inherited':True,'shared_order_first10_sha256':plan['shared_first10_order_content_sha256'],'original_guard_journal':session.guard.journal}
    (output/'actual_training_start.json').write_text(json.dumps(start_proof,indent=2)+'\n')
    budget('clean_start_or_exact_resume')
    finish=10 if args.phase=='stage10' else 100
    for epoch in range(start_epoch,finish):
        session.model.train();epoch_t=time.perf_counter();batch_times=[];objectives=[]
        for start in range(0,695,32):
            rows=orders[epoch,start:start+32];batch=row_batch(rows);session.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize();btime=time.perf_counter();loss=fit_objective(session.model,batch)
            if not torch.isfinite(loss):raise RuntimeError('NONFINITE_OBJECTIVE')
            loss.backward();grads=require_real_gradients(session.model)
            assert len(grads)==364
            session.optimizer.step();session.scheduler.step();updates+=1
            torch.cuda.synchronize();seconds=time.perf_counter()-btime
            batch_times.append(seconds);objectives.append(float(loss.detach()))
            with (output/'actual_fit_steps.jsonl').open('a') as f:f.write(json.dumps({'epoch':epoch+1,'step_in_epoch':start//32+1,'optimizer_steps':updates,'rows':rows.tolist(),'batch_size':len(rows),'seconds':seconds,'objective':objectives[-1],'all_finite_nonNone_tensors':len(grads),'optimizer_lr':[g['lr'] for g in session.optimizer.param_groups]})+'\n')
            session.model.dberta.last_losses={};session.model.dberta.last_trace={};session.model.dberta.last_first_prediction=None;session.model.dberta.own_flow._observer_prediction=None
            del batch,loss,grads
            budget('fit_epoch%d_step%d'%(epoch+1,start//32+1))
        assert updates==(epoch+1)*22 and len(batch_times)==22
        torch.cuda.synchronize();train_seconds=time.perf_counter()-epoch_t;it=time.perf_counter();prediction=predict_inner();torch.cuda.synchronize();inner_seconds=time.perf_counter()-it
        state_digest=tensor_sha(session.model.state_dict())
        pfile=output/('inner_epoch_%03d.npz'%(epoch+1))
        np.savez(pfile,row_ids=session.guard.ids['inner'],prediction=prediction,model_state_sha256=np.asarray(state_digest))
        pred_sha=sha(pfile)
        # Exactly INNER153 checkpoint-selection labels, after predictions freeze.
        labels=session.guard.inner_labels_after_frozen_predictions(pfile,pred_sha)
        metric=float(np.mean([np.mean((prediction[np.asarray(videos)==v]-labels[np.asarray(videos)==v])**2) for v in sorted(set(videos))]))
        if not math.isfinite(metric):raise RuntimeError('NONFINITE_INNER_SELECTION_METRIC')
        if metric<best_metric:
            best_metric=metric;best_epoch=epoch+1;best_state=cpu_state();best_pred=prediction.copy()
        assert stats_sha()==plan['fit_statistics_sha256']
        projected=100*(21*max(batch_times[:-1])+batch_times[-1]+inner_seconds)+600
        if projected>18000 or projected+7200>(lease-utc()).total_seconds():raise RuntimeError('ACTUAL_EPOCH100_BUDGET_PROJECTION')
        record={'actual_utc':utc().isoformat(),'epoch':epoch+1,'optimizer_steps':updates,'fit_rows':695,'batch_count':22,'tail_rows':23,'train_seconds':train_seconds,'inner_seconds':inner_seconds,'batch_seconds':batch_times,'mean_training_objective':float(np.mean(objectives)),'INNER_video_equal_MSE_selection_only':metric,'best_epoch':best_epoch,'best_metric':best_metric,'inner_prediction_file_sha256':pred_sha,'state_sha256':state_digest,'conservative_100_seconds_projection':projected,'OUTER_or_head_labels_read':False}
        history.append(record);(output/'progress.json').write_text(json.dumps({'scope':'ACTUAL_REFERENCE_TRAINING_PROGRESS_NOT_COMPLETE100','phase':args.phase,'history':history,'epoch':epoch+1,'optimizer_steps':updates,'formal100_complete':False},indent=2)+'\n')
        print('ACTUAL_REFERENCE_EPOCH '+json.dumps(record),flush=True);budget('epoch_complete_%d'%(epoch+1))
    latest_state=cpu_state();state_digest=tensor_sha(latest_state);rng,rng_extra=rng_save()
    meta={'format':'MINIMAL_FIXED_STAGED_REFERENCE_V1','scope':'STAGED_SHARED10_RESUME_NOT_COMPLETE100' if args.phase=='stage10' else 'FORMAL100_COMPLETE_RESUME','plan_sha256':plan['_sha256'],'runtime_plan_sha256':plan['runtime_plan_sha256'],'seed':91819,'fold':0,'epoch':finish,'optimizer_steps':updates,'state_sha256':state_digest,'fit_statistics_sha256':plan['fit_statistics_sha256'],'best_epoch':best_epoch,'best_metric':best_metric}
    resume=save(output/'complete_resume_full.pt',{'metadata':meta,'model':latest_state,'optimizer':cpu_tree(session.optimizer.state_dict()),'scheduler':session.scheduler.state_dict(),'rng':rng,'rng_extra':rng_extra,'history':history,'best_epoch':best_epoch,'best_metric':best_metric,'best_state':best_state,'best_prediction':best_pred.tolist()})
    best_meta=dict(meta,scope='SHARED10_BEST_NOT_COMPLETE100' if args.phase=='stage10' else 'FORMAL100_EARLIEST_INNER_BEST',epoch=best_epoch,optimizer_steps=best_epoch*22,state_sha256=tensor_sha(best_state))
    best=save(output/'selected_best_full.pt',{'model':best_state,'metadata':best_meta})
    # Full disk replay of own selected checkpoint; no extra optimizer steps.
    disk=torch.load(output/'selected_best_full.pt',map_location='cpu',weights_only=True)
    assert tensor_sha(disk['model'])==best_meta['state_sha256'];session.model.load_state_dict(disk['model'],strict=True);del disk
    replay=predict_inner();error=float(np.max(np.abs(replay-np.asarray(best_pred))))
    if error>1e-6 or stats_sha()!=plan['fit_statistics_sha256']:raise RuntimeError('SELECTED_COMPLETE_DISK_STRICT_REPLAY')
    np.savez(output/'selected_best_inner_replay.npz',row_ids=session.guard.ids['inner'],prediction=replay,model_state_sha256=np.asarray(best_meta['state_sha256']))
    # Save at the planned boundary before diagnostic branch updates. They are
    # isolated and never contribute to the 220/2200 formal optimizer steps.
    continuation_proof=None
    if args.phase=='stage10':
        from resume_next_update_check_v2 import verify_next_update
        continuation_proof=verify_next_update(session,output/'complete_resume_full.pt',
            orders[10,:32],args.asset_base,bundle,output,budget)
        # The diagnostic restores the exact saved last10 model, state and RNG.
        assert tensor_sha(session.model.state_dict())==state_digest
    journal=session.guard.journal
    # Release all GPU ownership before constructing a fresh public instance.
    del session
    torch.cuda.empty_cache()
    session=construct_public_candidate(args.asset_base,bundle)
    session.model.dberta.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
    del session.clean_initial
    disk=torch.load(output/'selected_best_full.pt',map_location='cpu',weights_only=True)
    assert tensor_sha(disk['model'])==best_meta['state_sha256']
    session.model.load_state_dict(disk['model'],strict=True);del disk
    assert tensor_sha(session.model.state_dict())==best_meta['state_sha256']
    fresh_replay=predict_inner();relabel=predict_inner(7.0)
    fresh_error=float(np.max(np.abs(fresh_replay-np.asarray(best_pred))))
    label_error=float(np.max(np.abs(fresh_replay-relabel)))
    if fresh_error>1e-6 or label_error!=0 or stats_sha()!=plan['fit_statistics_sha256']:
        raise RuntimeError('FRESH_INSTANCE_FULL_DISK_REPLAY_OR_RELABEL')
    assert tensor_sha(session.model.state_dict())==best_meta['state_sha256']
    np.savez(output/'fresh_selected_best_inner_replay.npz',row_ids=session.guard.ids['inner'],
        prediction=fresh_replay,model_state_sha256=np.asarray(best_meta['state_sha256']))
    # Existing fixed FIT rows only: do not scan amplitudes, checkpoints or labels.
    from donor_terminal_mechanism_precheck_v1 import donor_terminal_check
    mechanism,arrays=donor_terminal_check(session,row_batch(plan['mechanism_fit_rows']),
                                        'shared10_selected_best' if args.phase=='stage10' else 'formal100_selected_best')
    np.savez(output/'selected_best_donor_mechanism.npz',**arrays)
    if mechanism['seconds']>120:raise RuntimeError('MECHANISM_TWO_MINUTE_BUDGET')
    receipt={'status':'ACTUAL_SHARED10_REFERENCE_COMPLETE_NOT_FORMAL100' if args.phase=='stage10' else 'ACTUAL_REFERENCE100_TRAINING_COMPLETE_OUTER_STILL_DISABLED','actual_utc':utc().isoformat(),'pid':os.getpid(),'argv':sys.argv,'phase':args.phase,'plan_sha256':plan['_sha256'],'epochs':finish,'optimizer_steps':updates,'best_epoch':best_epoch,'INNER_video_equal_MSE_selection_only':best_metric,'strict_same_instance_disk_replay_error':error,'fresh_instance_replay_pending':False,'fresh_instance_strict_full_disk_replay_error':fresh_error,'dummy_label_replacement_error':label_error,'next_update_continuation_check':continuation_proof,'donor_mechanism':mechanism,'fit_statistics_sha256':stats_sha(),'selected_best_full':best,'complete_resume_full':resume,'history':history,'final_budget':budget('complete_state_save_and_replay'),'original_guard_journal':journal,'fresh_instance_guard_journal':session.guard.journal,'OUTER_input_or_labels_used':False,'head_training_started':False,'donor_performance_superiority_claimed':False,'D_B_CPU_preservation_pending':True,'formal100_complete':args.phase=='continue100'}
    (output/'actual_training_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--asset-base',required=True);p.add_argument('--bundle',required=True);p.add_argument('--evidence',required=True);p.add_argument('--out',required=True);p.add_argument('--phase',choices=['stage10','continue100'],required=True);p.add_argument('--resume');args=p.parse_args()
    print(json.dumps(run(args)),flush=True)
