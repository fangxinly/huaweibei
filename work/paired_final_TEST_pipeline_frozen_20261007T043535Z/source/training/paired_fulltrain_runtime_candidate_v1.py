"""Prospective paired PRECHECK/TRAIN runner. A complete execution freeze and actual gates are required."""
import argparse,datetime,hashlib,json,math,os,random,shutil,subprocess,sys,time
from pathlib import Path
from paired_fulltrain_evidence_candidate_v1 import precheck_training_gate

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc)
def write(path,value):
    p=Path(path);tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');tmp.replace(p)

def validate(args):
    planfile=args.bundle/'paired_fulltrain_execution_plan.json'
    if sha(planfile)!=args.plan_sha:raise ValueError('Execution protocol SHA mismatch')
    plan=json.loads(planfile.read_text(encoding='utf-8'))
    if plan.get('status')!='PAIRED_FULLTRAIN_RUNTIME_EXECUTION_PROTOCOL_FROZEN':
        raise PermissionError('Local preparation cannot launch')
    expected={'task_seed':128,'orders_seed':128,'epochs':100,'updates':4000,'train_rows':1281,'dev_rows':229,
              'batch':32,'drop_last':True,'dev_batch':128,'DEV_selection':'author_batch_MSE_strict_earliest',
              'final_TEST_enabled':False,'residual_head_enabled':False}
    if any(plan.get(k)!=v for k,v in expected.items()):raise ValueError('Paired protocol mismatch')
    if args.method not in ('careflow','minimal_fixed_F'):raise ValueError('Undeclared method')
    if not str(args.root).startswith('/data/coding/paired_fulltrain_'):raise ValueError('Wrong new root')
    for rel,h in plan['source_sha256'].items():
        if sha(args.bundle/rel)!=h:raise ValueError('Source SHA mismatch: '+rel)
    for rel,h in plan['asset_sha256'].items():
        if sha(args.assets/rel)!=h:raise ValueError('Public asset SHA mismatch: '+rel)
    if sha(args.bundle/plan['orders_file'])!=plan['orders_sha256']:raise ValueError('Orders SHA mismatch')
    if not plan.get('original_CPU_and_root_specific_capture_sources_frozen'):
        raise PermissionError('Full preservation dependencies absent')
    if args.stage=='train':args.verified_original_precheck=precheck_training_gate(args,plan)
    return plan

def fresh_preflight(args,plan,out):
    raw={}
    for name,cmd in [('gpu_identity',['nvidia-smi','--query-gpu=uuid','--format=csv,noheader']),
                     ('compute',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'])]:
        result=subprocess.run(cmd,check=True,capture_output=True,text=True);raw[name]=result.stdout
        (out/(name+'.raw.txt')).write_text(result.stdout,encoding='utf-8')
    if raw['gpu_identity'].strip()!=plan['assigned_gpu_UUID'][args.method] or raw['compute'].strip():
        raise PermissionError('Assigned physical GPU UUID/empty-compute requirement failed')
    if shutil.disk_usage(args.root).free<plan['remote_free_floor_bytes']:raise PermissionError('Remote free-space gate failed')
    lease=datetime.datetime.fromisoformat(plan['conservative_lease_end_UTC'])
    if lease>datetime.datetime(2026,10,7,13,30,tzinfo=datetime.timezone.utc):raise ValueError('Unsupported lease extension')
    remaining=(lease-utc()).total_seconds()
    if remaining<plan['execution_budget_seconds']+7200:raise PermissionError('Execution plus2h saving reserve absent')
    write(out/'actual_direct_preflight.json',{'actual_utc':utc().isoformat(),'argv':sys.argv,'root':str(args.root),
          'source_bundle':str(args.bundle),'plan_sha256':args.plan_sha,'physical_raw_checks':raw,
          'remote_free_bytes':shutil.disk_usage(args.root).free,'remaining_seconds':remaining,
          'lease_limit_is_conservative_human_lease_not_platform_confirmation':True,
          'trusted_human_lease_provenance_reference':plan['human_lease_provenance_reference']})

def run(args):
    plan=validate(args)
    out=args.root/'out';out.mkdir(exist_ok=False)
    fresh_preflight(args,plan,out)
    os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
    import numpy as np
    import torch
    from paired_fulltrain_session_candidate import construct,predictions,training_loss
    from fixed_flow_components_candidate import tensor_sha
    from common_budget_selection_candidate import train_batches,author_dev_batch_mse,improves_earliest
    torch.set_num_threads(2)
    session=construct(args.method,args.bundle,args.assets,plan)
    initial_hash=tensor_sha(session.clean_initial['model'])
    def cpu_tree(x):
        if torch.is_tensor(x):return x.detach().cpu().clone()
        if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
        if isinstance(x,list):return [cpu_tree(v) for v in x]
        if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
        return x
    def rng():return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state().clone(),
                      'cuda':[x.clone() for x in torch.cuda.get_rng_state_all()]}
    def rng_hash(value):
        import pickle
        return hashlib.sha256(pickle.dumps({'python':value['python'],'numpy':value['numpy'],
               'torch':value['torch'].tolist(),'cuda':[x.tolist() for x in value['cuda']]},protocol=4)).hexdigest()
    clean_rng=rng();clean_rng_hash=rng_hash(clean_rng)
    parameter_names={id(p):name for name,p in session.model.named_parameters()}
    optimizer_index_to_name={}
    for live,saved in zip(session.optimizer.param_groups,session.optimizer.state_dict()['param_groups']):
        for param,index in zip(live['params'],saved['params']):optimizer_index_to_name[str(index)]=parameter_names[id(param)]
    parameter_info={name:{'shape':list(p.shape),'dtype':str(p.dtype),'numel':p.numel()} for name,p in session.model.named_parameters()}
    clean_checkpoint={'metadata':dict(session.construction_receipt,plan_sha256=args.plan_sha,
                            source_bundle=str(args.bundle),root=str(args.root),initial_rng_sha256=clean_rng_hash,
                            parameter_info=parameter_info,optimizer_index_to_name=optimizer_index_to_name),
                      'model':session.clean_initial['model'],'optimizer':cpu_tree(session.optimizer.state_dict()),
                      'scheduler':cpu_tree(session.scheduler.state_dict()),'rng':clean_rng}
    def save(path,value):
        torch.save(value,path)
        return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
    def budget(phase):
        torch.cuda.synchronize()
        report={'actual_utc':utc().isoformat(),'phase':phase,'allocated_peak':torch.cuda.max_memory_allocated(),
                'reserved_peak':torch.cuda.max_memory_reserved(),'remote_free_bytes':shutil.disk_usage(args.root).free}
        with (out/'cumulative_budget.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
        if max(report['allocated_peak'],report['reserved_peak'])>plan['max_gpu_peak_bytes']:
            raise RuntimeError('Cumulative GPU peak exceeded declared budget')
        if report['remote_free_bytes']<plan['remote_free_floor_bytes']:
            raise RuntimeError('Declared preservation space floor crossed')
        return report
    order=np.load(args.bundle/plan['orders_file'],allow_pickle=False)
    if order.shape!=(100,1281) or order.dtype.kind not in 'iu':raise ValueError('Shared orders shape/dtype')
    for row in order:train_batches([int(x) for x in row])
    shutil.copyfile(args.bundle/plan['orders_file'],out/'original_shared_TRAIN_orders.npy')
    if sha(out/'original_shared_TRAIN_orders.npy')!=plan['orders_sha256']:raise ValueError('Physical original orders copy mismatch')
    ids={k:tuple(v) for k,v in session.guard.ids.items()}
    identity=hashlib.sha256(json.dumps(ids,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
    if identity!=plan['official_train_dev_ID_identity_sha256']:raise PermissionError('Actual official row identity mismatch')
    write(out/'actual_official_row_identity.json',{'ids':ids,'sha256':identity,'orders_sha256':plan['orders_sha256']})
    np.save(out/'original_TRAIN_supervision.npy',session.train.tensors[3].detach().cpu().numpy().reshape(-1).astype(np.float32),allow_pickle=False)
    write(out/'actual_construction_receipt.json',session.construction_receipt)
    budget('constructed_before_optimizer')
    def row_batch(rows):return tuple(x[rows].to(session.author.DEVICE) for x in session.train.tensors)
    def predict_dev(dummy=0):
        session.model.eval();pieces=[]
        with torch.no_grad():
            for start in range(0,229,128):
                batch=[x[start:start+128].to(session.author.DEVICE) for x in session.dev_inputs.tensors]
                batch[3]=torch.full_like(batch[3],dummy)
                p=predictions(session,args.method,tuple(batch));pieces.append(p.detach().cpu().numpy())
        result=np.concatenate(pieces).astype(np.float32,copy=False)
        if result.shape!=(229,) or not np.isfinite(result).all():raise RuntimeError('Invalid DEV prediction')
        return result
    def step(rows):
        session.model.train();session.optimizer.zero_grad(set_to_none=True)
        batch=row_batch(rows);torch.cuda.synchronize();t=time.perf_counter()
        loss=training_loss(session,args.method,batch)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite TRAIN loss')
        loss.backward()
        gradients={n:float(p.grad.detach().abs().sum()) for n,p in session.model.named_parameters() if p.grad is not None}
        if any(not torch.isfinite(p.grad).all() for p in session.model.parameters() if p.grad is not None):
            raise RuntimeError('Nonfinite gradient')
        if args.method=='minimal_fixed_F' and len(gradients)!=len(list(session.model.parameters())):
            raise RuntimeError('Retained fixed parameter gradient missing')
        if not gradients or not any(n.startswith('dberta.model.') and x>0 for n,x in gradients.items()):
            raise RuntimeError('Language encoder has no real gradient')
        session.optimizer.step();session.scheduler.step();torch.cuda.synchronize()
        result={'seconds':time.perf_counter()-t,'loss':float(loss.detach()),'gradient_tensors':len(gradients),
                'gradient_zero_tensors':[n for n,x in gradients.items() if x==0],
                'gradient_missing_tensors':[n for n,p in session.model.named_parameters() if p.grad is None]}
        del batch,loss,gradients
        return result
    started=time.perf_counter()
    if args.stage=='precheck':
        initial_saved=save(out/'clean_initial_full.pt',clean_checkpoint)
        updates=[]
        for rows in [order[0,:32].tolist(),plan['precheck_mixed_TRAIN_rows']]:
            if len(rows)!=32 or len(set(rows))!=32 or any(i<0 or i>=1281 for i in rows):raise ValueError('Predeclared precheck row guard')
            updates.append(dict(step(rows),rows=rows));budget('precheck_optimizer_state_allocation')
        state=tensor_sha(session.model.state_dict());r_before=rng_hash(rng())
        t=time.perf_counter();p0=predict_dev(0);p7=predict_dev(7);dev_seconds=(time.perf_counter()-t)/2
        if not np.array_equal(p0,p7) or tensor_sha(session.model.state_dict())!=state or rng_hash(rng())!=r_before:
            raise RuntimeError('DEV dummy label/state/RNG invariance failure')
        after=save(out/'precheck_after2_full.pt',{'metadata':dict(session.construction_receipt,optimizer_steps=2,
                plan_sha256=args.plan_sha,state_sha256=state,parameter_info=parameter_info,
                optimizer_index_to_name=optimizer_index_to_name,missing_gradient_names=updates[-1]['gradient_missing_tensors'],
                rng_sha256=rng_hash(rng())),
                'model':cpu_tree(session.model.state_dict()),
                'optimizer':cpu_tree(session.optimizer.state_dict()),'scheduler':cpu_tree(session.scheduler.state_dict()),'rng':rng()})
        session.model.load_state_dict(torch.load(after['path'],map_location='cpu')['model'],strict=True)
        replay=predict_dev(0)
        if not np.array_equal(replay,p0):raise RuntimeError('Own complete checkpoint DEV replay failure')
        np.savez(out/'precheck_DEV_dummy_full_replay.npz',row_ids=np.asarray(ids['dev']),
                 prediction_dummy0=p0,prediction_dummy7=p7,prediction_disk_replay=replay,
                 model_state_sha256=np.asarray(state))
        projection=4000*max(x['seconds'] for x in updates)*1.5+100*dev_seconds+900
        remaining=(datetime.datetime.fromisoformat(plan['conservative_lease_end_UTC'])-utc()).total_seconds()
        if projection>plan['execution_budget_seconds'] or projection+7200>remaining:
            raise RuntimeError('Actual precheck timing does not satisfy execution/save reserve')
        receipt={'status':'ACTUAL_PAIRED_METHOD_PRECHECK2_COMPLETE_NOT_TRAINED100','actual_utc':utc().isoformat(),
                 'method':args.method,'root':str(args.root),'plan_sha256':args.plan_sha,'source_bundle':str(args.bundle),
                 'argv':sys.argv,'optimizer_steps':2,'clean_initial_state_sha256':initial_hash,
                 'clean_initial_rng_sha256':clean_rng_hash,'clean_initial_full':initial_saved,'precheck_after2_full':after,
                 'updates':updates,'DEV_dummy0vs7_and_same_instance_full_replay_error':0,
                 'DEV_true_labels_read':False,'official_row_ID_identity_sha256':identity,'projected_training_seconds':projection,
                 'final_budget':budget('complete_precheck_full_save_replay'),'guard_journal':session.guard.journal,
                 'D_other_CPU_joint_pending':True,'elapsed_seconds':time.perf_counter()-started}
        write(out/'actual_stage_receipt.json',receipt);print(json.dumps(receipt),flush=True);return
    parent=args.verified_original_precheck
    if parent['clean_initial_state_sha256']!=initial_hash or parent['clean_initial_rng_sha256']!=clean_rng_hash:
        raise RuntimeError('Fresh training construction differs from original clean initial state/RNG')
    if len(session.optimizer.state)!=0 or session.scheduler.last_epoch!=0:raise RuntimeError('Inherited precheck optimizer/scheduler state')
    del clean_checkpoint
    session.clean_initial=None
    history=[];best_metric=float('inf');best_epoch=0;best_state=None;best_prediction=None;updates=0
    write(out/'actual_training_start.json',{'actual_utc':utc().isoformat(),'method':args.method,'plan_sha256':args.plan_sha,
          'initial_state_sha256':initial_hash,'initial_rng_sha256':clean_rng_hash,'optimizer_steps':0,
          'clean_initial_matches_original_precheck':True,'precheck_two_steps_not_inherited':True})
    for epoch in range(100):
        batches,omitted=train_batches([int(x) for x in order[epoch]])
        times=[];losses=[]
        for number,rows in enumerate(batches,1):
            result=step(list(rows));updates+=1;times.append(result['seconds']);losses.append(result['loss'])
            with (out/'actual_TRAIN_steps.jsonl').open('a') as f:
                f.write(json.dumps(dict(result,epoch=epoch+1,step_in_epoch=number,optimizer_steps=updates,rows=rows,
                                       batch_size=32,learning_rates=[g['lr'] for g in session.optimizer.param_groups]))+'\n')
            budget('formal_optimizer_step')
        if updates!=(epoch+1)*40 or session.scheduler.last_epoch!=updates:raise RuntimeError('Actual4000 update/scheduler alignment failed')
        prediction=predict_dev(0);state=tensor_sha(session.model.state_dict())
        path=out/('DEV_epoch_%03d_prediction_only.npz'%(epoch+1))
        np.savez(path,row_ids=np.asarray(ids['dev']),prediction=prediction,model_state_sha256=np.asarray(state))
        frozen_sha=sha(path)
        with (out/'actual_DEV_prediction_frozen_before_labels.jsonl').open('a') as frozen_log:
            frozen_log.write(json.dumps({'actual_utc':utc().isoformat(),'epoch':epoch+1,'path':str(path),'sha256':frozen_sha,'state_sha256':state,'true_labels_not_yet_accessed_for_this_epoch':True})+'\n')
        y=session.guard.dev_labels_after_frozen_prediction(path,frozen_sha,state)
        if epoch==0:
            np.save(out/'original_DEV_selection_targets.npy',y,allow_pickle=False)
        elif not np.array_equal(np.load(out/'original_DEV_selection_targets.npy',allow_pickle=False),y):
            raise RuntimeError('Official DEV target/order changed across selection epochs')
        # Both methods use identical fixed DEV batches and arithmetic mean; float64 scoring is declared.
        batch_mse=[float(np.mean((prediction[start:start+128].astype(np.float64)-y[start:start+128])**2)) for start in (0,128)]
        metric=author_dev_batch_mse(batch_mse,(128,101))
        if improves_earliest(best_metric,metric):
            best_metric=metric;best_epoch=epoch+1;best_state=cpu_tree(session.model.state_dict());best_prediction=prediction.copy()
        row={'actual_utc':utc().isoformat(),'epoch':epoch+1,'optimizer_steps':updates,'dropped_TRAIN_rows':omitted,
             'TRAIN_mean_objective':float(np.mean(losses)),'DEV_author_batch_MSE_selection_only':metric,
             'best_epoch':best_epoch,'best_metric':best_metric,'DEV_prediction_file_SHA':frozen_sha,'state_SHA':state}
        history.append(row);write(out/'progress.json',row);write(out/'history.json',history);print(json.dumps(row),flush=True)
    final_state=cpu_tree(session.model.state_dict())
    final_rng=rng()
    metadata={'method':args.method,'plan_sha256':args.plan_sha,'source_bundle':str(args.bundle),'root':str(args.root),
              'epochs':100,'optimizer_steps':4000,'train_rows':1281,'DEV_rows':229,'orders_sha256':plan['orders_sha256'],
              'official_row_ID_identity_sha256':identity,'best_epoch':best_epoch,'best_metric':best_metric,
              'best_state_SHA':tensor_sha(best_state),'latest_state_SHA':tensor_sha(final_state),
              'parameter_info':parameter_info,'optimizer_index_to_name':optimizer_index_to_name,
              'missing_gradient_names':result['gradient_missing_tensors'],'rng_sha256':rng_hash(final_rng),
              'DEV_selection_precision':'float64 batch MSE from original FP32 predictions; common to both, differs from original cached float32 selection'}
    resume=save(out/'complete_resume_full.pt',{'metadata':metadata,'model':final_state,
             'optimizer':cpu_tree(session.optimizer.state_dict()),'scheduler':cpu_tree(session.scheduler.state_dict()),
             'rng':final_rng,'history':history,'best_state':best_state,'best_prediction':best_prediction})
    best=save(out/'selected_best_full.pt',{'metadata':metadata,'model':best_state,'DEV_selected_prediction':best_prediction})
    session.model.load_state_dict(torch.load(best['path'],map_location='cpu')['model'],strict=True)
    replay=predict_dev(0);dummy=predict_dev(7)
    if not np.array_equal(replay,best_prediction) or not np.array_equal(dummy,replay):
        raise RuntimeError('Selected whole checkpoint/dummy DEV replay failed')
    np.savez(out/'selected_best_DEV_replay.npz',row_ids=np.asarray(ids['dev']),prediction=replay,model_state_sha256=np.asarray(metadata['best_state_SHA']))
    receipt={'status':'ACTUAL_PAIRED_METHOD_FULLTRAIN100_COMPLETE_PRESERVATION_PENDING_NO_FINAL_TEST',
             'actual_utc':utc().isoformat(),'method':args.method,'argv':sys.argv,'root':str(args.root),
             'source_bundle':str(args.bundle),'plan_sha256':args.plan_sha,'metadata':metadata,'selected_best_full':best,
             'complete_resume_full':resume,'same_instance_whole_selected_replay_error':0,'DEV_dummy0vs7_error':0,
             'guard_journal':session.guard.journal,'final_budget':budget('complete_save_selected_replay'),
             'elapsed_seconds':time.perf_counter()-started,'original_other_CPU_and_fresh_instance_replay_pending':True,
             'final_TEST_executed':False,'new_five_metric_superiority_claimed':False}
    write(out/'actual_stage_receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for n in ('root','bundle','assets'):parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--method',choices=('careflow','minimal_fixed_F'),required=True)
    parser.add_argument('--stage',choices=('precheck','train'),required=True)
    parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--precheck-root',type=Path);parser.add_argument('--precheck-joint',type=Path);parser.add_argument('--precheck-joint-sha')
    run(parser.parse_args())
