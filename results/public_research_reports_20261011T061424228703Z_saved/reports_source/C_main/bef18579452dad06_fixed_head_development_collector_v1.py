"""Future frozen heads + same best/candidate, exact201 dummy inputs; no true labels or scores."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
import numpy as np

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def utc():return datetime.datetime.now(datetime.timezone.utc)
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def run(a):
    bundle=Path(a.bundle);plan=read(bundle/'matched_head_execution_plan.json')
    if plan['status']!='COMPLETE_MATCHED_HEAD_EXECUTION_PROTOCOL_FROZEN':raise PermissionError('FULL_EXECUTION_PROTOCOL_NOT_READY')
    e=read(a.evidence)
    fitjoint=read(a.head_fit_joint)
    if fitjoint['status']!='ACTUAL_HEADFIT232_GPU_PARENT_D_OTHER_CPU_FIT_SOURCE_STATE_CAPTURE_JOINT_PASSED' or fitjoint['plan_sha256']!=sha(bundle/'matched_head_execution_plan.json') or fitjoint['head_state_sha256']!=sha(a.head_state):raise PermissionError('ACTUAL_FIT_D_OTHER_CPU_COMPLETE_JOINT')
    head=read(a.head_state)
    if head['scope']!='FIXED_HEADFIT232_ONLY_U_W_AND_CONSTANT' or head['plan_sha256']!=sha(bundle/'matched_head_execution_plan.json') or head['headEVAL201_inputs_or_labels_used']:raise PermissionError('EXACT_FROZEN_FIT_HEAD_SCOPE')
    if datetime.datetime.fromisoformat(head['actual_frozen_utc'])>=utc():raise PermissionError('FIT_HEAD_FROZEN_BEFORE201_INPUTS')
    head_sha=sha(a.head_state)
    if e['scope']!='FIXED_HEADEVAL201_FROZEN_NINE_READOUT_INPUTS_V1':raise PermissionError('EXECUTION_SCOPE')
    if not e['human_provenance_verified'] or e['lease_source']!='DIRECT_HUMAN_NEW_P4_24H_20261006':raise PermissionError('HUMAN_LEASE_PROVENANCE')
    if not 0<=(utc()-datetime.datetime.fromisoformat(e['actual_query_utc'])).total_seconds()<=300:raise PermissionError('FIVE_MINUTE_FRESH')
    if e['gpu_uuid']!='GPU-53696803-875e-eec8-2231-29db63579891' or e['compute_processes']!=[] or not isinstance(e['python_full_argv'],list):raise PermissionError('UUID_COMPUTE_FULLARGV')
    if not e['asset_and_source_actual_SHA_verified'] or e['plan_sha256']!=sha(bundle/'matched_head_execution_plan.json'):raise PermissionError('SOURCE_ASSET_SHA')
    if e['remote_free_bytes']<4*1024**3 or e['permanent_D_free_bytes']<6*1024**3:raise PermissionError('PRESERVATION_SPACE')
    lease=datetime.datetime.fromisoformat(e['lease_end_utc'])
    if (lease-utc()).total_seconds()<1200+7200:raise PermissionError('EXECUTION_PLUS_TWO_HOUR_PRESERVATION')
    for name,h in plan['source_and_role_sha256'].items():
        if sha(bundle/name)!=h:raise ValueError('SOURCE_ROLE_SHA: '+name)
    complete=read(a.completion_joint)
    if sha(a.completion_joint)!=plan['reference_joint_sha256'] or complete['status']!='ACTUAL_REFERENCE100_GPU_D_B_COMPLETE_STATE_SOURCE_ORDER_ARRAY_CAPTURE_JOINT_PASSED':raise ValueError('ORIGINAL_100_GPU_D_B_CPU_PRESERVATION')
    if not complete['original_CPU_natural_exit']['natural_wait_verified'] or complete['original_CPU_natural_exit']['exit_code']!=0:raise ValueError('ORIGINAL_CPU_NATURAL_EXIT')
    if sha(a.checkpoint)!=plan['checkpoint_file_sha256']:raise ValueError('WHOLE_ORIGINAL_BEST_FILE')
    out=Path(a.out)
    if out.exists():raise FileExistsError('UNIQUE_NEW_OUTPUT_NO_REUSE')
    out.mkdir(parents=True)
    import torch
    import minimal_fixed_runtime_v1 as runtime
    from head_development_input_guard_v1 import allowed_head_inputs,candidate_gate
    if subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()!=e['gpu_uuid']:raise ValueError('ACTUAL_GPU_UUID_CHANGED')
    start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
    def budget(stage):
        torch.cuda.synchronize()
        b={'actual_utc':utc().isoformat(),'stage':stage,'elapsed_seconds':time.perf_counter()-start,
           'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'remote_free_bytes':shutil.disk_usage('/data').free}
        with (out/'budget_trace.jsonl').open('a') as f:f.write(json.dumps(b)+'\n')
        if b['elapsed_seconds']>1200 or max(b['peak_allocated_bytes'],b['peak_reserved_bytes'])>6*1024**3:raise RuntimeError('ACTUAL_COLLECTION_BUDGET')
        if b['remote_free_bytes']<4*1024**3 or (lease-utc()).total_seconds()<7200:raise RuntimeError('PRESERVATION_RESERVE')
        return b
    # Pinned public constructor needs a FIT dataset but no supervised update.
    # Replace only its guard class locally so even original FIT labels are never
    # indexed. Public random construction/statistics remain identical; frozen
    # best whole state is loaded strict afterwards. Old scientific source stays.
    OriginalGuard=runtime.FoldZeroGuard
    class NoLabelConstructorGuard(OriginalGuard):
        def fit_supervision(self,rows):return self.inputs_only(rows,'fit')
        def inner_labels_after_frozen_predictions(self,*args):raise PermissionError('ALL_TASK_LABELS_FORBIDDEN')
    runtime.FoldZeroGuard=NoLabelConstructorGuard
    try:session=runtime.construct_public_candidate(a.asset_base,bundle)
    finally:runtime.FoldZeroGuard=OriginalGuard
    assert all(not j['labels_read'] for j in session.guard.journal)
    assert session.construction_receipt['parameters']==185402807 and session.construction_receipt['parameter_tensors']==364
    del session.optimizer,session.scheduler,session.clean_initial,session.fit,session.inner_inputs
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    m=cp['metadata']
    if m['scope']!='FORMAL100_EARLIEST_INNER_BEST' or m['epoch']!=41 or m['fold']!=0 or m['seed']!=91819:raise ValueError('UNIQUE_BEST41_SCOPE')
    if runtime.tensor_sha(cp['model'])!=plan['checkpoint_state_sha256']:raise ValueError('FULL_STATE_SHA')
    session.model.load_state_dict(cp['model'],strict=True);del cp
    model=session.model;model.eval()
    before=runtime.tensor_sha(model.state_dict())
    assert before==plan['checkpoint_state_sha256']
    stats=lambda:runtime.tensor_sha({n:v for n,v in model.state_dict().items() if n.startswith(('dberta.v6_audio_','dberta.v6_visual_'))})
    before_stats=stats();assert before_stats==plan['FIT_statistics_sha256']
    rows=np.load(bundle/'head_development_eval_0.npy',allow_pickle=False)
    fitrows=np.load(bundle/'head_development_fit_0.npy',allow_pickle=False)
    mapping=read(bundle/'train_row_video_mapping.json');allvideos=[x['video_id'] for x in mapping]
    if set(allvideos[int(i)] for i in rows)&set(allvideos[int(i)] for i in fitrows):raise PermissionError('HEAD_FIT_EVAL_VIDEO_OVERLAP')
    records=allowed_head_inputs(session.guard._examples,rows,rows,allvideos)
    # Source-pinned helper module already imported by the public constructor.
    author,_=sys.modules['run_careflow'].load_author(type('Args',(),{'repo':Path(a.asset_base)/'assets/CaReFlow','backbone':Path(a.asset_base)/'assets/deberta-v3-base','epochs':100,'seed':91819})())
    # load_author reparses functions/DEVICE only, but may reset module globals;
    # it must not construct another model or optimizer.
    dataset=author.get_appropriate_dataset(records)
    assert len(dataset)==201 and all(x[1].shape==(1,1) for x in records)
    del records
    flow=model.dberta.own_flow;original=flow.fixed_context
    cpu_rng=torch.get_rng_state().clone();cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()]
    journal=list(session.guard.journal)+[{'role':'headEVAL201','rows':rows.tolist(),'labels_read':False,'dummy':0,'purpose':'one_candidate_inputs_only'}]
    predictions=[[] for _ in range(4)];semantic=[];dummy_checks=[]
    def forward(values,condition):
        counts=[];states=[]
        def fixed(old,feedback):
            used=feedback if condition=='F' else torch.zeros_like(feedback)
            counts.append({'directions':int(feedback.shape[1]),'old_max':float(old.detach().abs().max()),'used_feedback_max':float(used.detach().abs().max())})
            return original(old,used)
        def hook(module,inputs,output):states.append(inputs[0].detach().cpu().clone())
        h=flow.reader.register_forward_hook(hook)
        flow.fixed_context=fixed
        try:
            p=runtime.forward_batch(model,values).detach().cpu().numpy().copy()
            assert len(counts)==1 and counts[0]['directions']==6 and len(states)==2 and flow._context_updates==2
            if condition=='C':assert counts[0]['used_feedback_max']==0
            if not np.isfinite(p).all() or any(not torch.isfinite(s).all() for s in states):raise ValueError('NONFINITE_PREDICTION_OR_STATE')
            return p,states,counts[0]
        finally:
            h.remove();del flow.fixed_context
            model.dberta.last_losses={};model.dberta.last_trace={};model.dberta.last_first_prediction=None;flow._observer_prediction=None
    budget('strict_frozen_checkpoint_loaded')
    with torch.no_grad():
        for pos in range(0,201,32):
            values=[x[pos:pos+32].to('cuda') for x in dataset.tensors];values[3]=torch.zeros_like(values[3])
            runs=[]
            for slot,condition in enumerate(('F','C','C','F')):
                p,s,observed=forward(tuple(values),condition);predictions[slot].append(p);runs.append((p,s,observed))
            differences=[float(torch.max(torch.abs(runs[0][1][0]-x[1][0]))) for x in runs[1:]]
            if max(differences)!=0:raise ValueError('FIRST_EULER_CHANGED_BY_CONTEXT_ONLY_INTERVENTION')
            ferror=float(np.max(np.abs(runs[0][0].astype(np.float64)-runs[3][0])))
            cerror=float(np.max(np.abs(runs[1][0].astype(np.float64)-runs[2][0])))
            if max(ferror,cerror)>1e-6:raise ValueError('BOTH_PATH_REPEAT_REQUIRED')
            semantic.append({'rows':rows[pos:pos+32].tolist(),'sequence':['F','C','C','F'],'first_euler_errors':differences,'F_repeat_error':ferror,'C_repeat_error':cerror,'fixed_context_calls':[x[2] for x in runs]})
            if pos==0:
                for condition,slot in [('F',0),('C',1)]:
                    changed=list(values);changed[3]=torch.full_like(changed[3],7)
                    p,_,_=forward(tuple(changed),condition)
                    error=float(np.max(np.abs(p.astype(np.float64)-runs[slot][0])))
                    if error!=0:raise ValueError('DUMMY_LABEL_REPLACEMENT_MUST_BE_EXACT_ZERO')
                    dummy_checks.append({'fixed_rows':rows[:32].tolist(),'condition':condition,'dummy0_vs7_error':error,'labels_read':False})
            del values,runs
            budget('fixed_batch_'+str(pos//32+1))
    arrays=[np.concatenate(x) for x in predictions]
    video=np.asarray([allvideos[int(i)] for i in rows]);segment=np.asarray([mapping[int(i)]['segment_id'] for i in rows])
    diag=candidate_gate(arrays[0],arrays[1],rows,video,arrays[3])
    cerror=np.abs(arrays[1].astype(np.float64)-arrays[2]);ferror=np.abs(arrays[0].astype(np.float64)-arrays[3])
    delta=arrays[1].astype(np.float64)-arrays[0];rawW=4*delta**2
    pervideo={}
    for v in np.unique(video):
        x=delta[video==v];pervideo[str(v)]={'rows':len(x),'nonzero':int(np.sum(x!=0)),'positive':int(np.sum(x>0)),'negative':int(np.sum(x<0)),'mean':float(x.mean()),'RMS':float(np.sqrt(np.mean(x*x))),'abs_quantiles':np.quantile(np.abs(x),[0,.25,.5,.75,.9,.99,1]).tolist()}
    centered=arrays[0].astype(np.float64)-arrays[0].astype(np.float64).mean()
    corrected=arrays[1].astype(np.float64)-arrays[1].astype(np.float64).mean()
    normF=float(np.linalg.norm(centered));normC=float(np.linalg.norm(corrected))
    geom=None if normF==0 or normC==0 else float(np.linalg.norm(corrected/normC-centered/normF))
    diag.update(C_repeat_max_error=float(cerror.max()),F_repeat_max_error=float(ferror.max()),delta_abs_quantiles=np.quantile(np.abs(delta),[0,.25,.5,.75,.9,.99,1]).tolist(),per_video=pervideo,
       candidate_Corr_absolute_change_geometric_bound=geom,interval_discrete_conservative_Corr_absolute_change_bound=None if normF==0 else min(2.,float(2*np.linalg.norm(delta)/normF)),
       Corr_bounds_are_not_improvement_sign_or_selection_gate=True,W_raw_definition='4*delta^2 evaluated in float64; factor cancels only after global videoequal normalization')
    if runtime.tensor_sha(model.state_dict())!=before or stats()!=before_stats:raise ValueError('PERSISTENT_PARAMETER_BUFFER_STAT_CHANGED')
    if not torch.equal(cpu_rng,torch.get_rng_state()) or any(not torch.equal(x,y) for x,y in zip(cuda_rng,torch.cuda.get_rng_state_all())):raise ValueError('EVAL_RNG_CHANGED')
    np.savez_compressed(out/'original_headEVAL201_label_free.npz',row_ids=rows,video_ids=video,segment_ids=segment,pF=arrays[0],pC=arrays[1],pC_repeat=arrays[2],pF_restored=arrays[3],delta=delta,W_raw=rawW,F_repeat_errors=ferror,C_repeat_errors=cerror,
       F_acc2_class=arrays[0]>=0,C_acc2_class=arrays[1]>=0,F_acc7_class=np.round(np.clip(arrays[0],-3,3)),C_acc7_class=np.round(np.clip(arrays[1],-3,3)))
    from matched_fixed_residual_heads_v2 import predict
    readouts=predict(head['model'],arrays[0].astype(np.float64),delta,arrays[1].astype(np.float64))
    np.savez_compressed(out/'frozen_headEVAL201_predictions.npz',row_ids=rows,head_state_sha256=np.asarray(head_sha),**readouts)
    if sha(a.head_state)!=head_sha:raise ValueError('FROZEN_HEAD_CHANGED_DURING201_INFERENCE')
    write(out/'nine_prediction_freeze.json',{'actual_utc':utc().isoformat(),'plan_sha256':sha(bundle/'matched_head_execution_plan.json'),'head_state_sha256':head_sha,'head_fit_joint_sha256':sha(a.head_fit_joint),'original_label_free_array_sha256':sha(out/'original_headEVAL201_label_free.npz'),'frozen_predictions_sha256':sha(out/'frozen_headEVAL201_predictions.npz'),'nine_readouts':list(readouts),'rows':rows.tolist(),'201_task_labels_indexed':False})
    final=budget('complete_original_arrays_and_all_nine_readouts')
    receipt={'status':'ACTUAL_HEADEVAL201_FROZEN_NINE_PREDICTIONS_LABEL_FREE_COMPLETE_NOT_SCORES','actual_utc':utc().isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),'plan_sha256':sha(bundle/'matched_head_execution_plan.json'),'checkpoint_file_sha256':plan['checkpoint_file_sha256'],'checkpoint_state_before_after_sha256':before,'FIT_statistics_before_after_sha256':before_stats,
       'original_array_sha256':sha(out/'original_headEVAL201_label_free.npz'),'frozen_predictions_sha256':sha(out/'frozen_headEVAL201_predictions.npz'),'nine_prediction_freeze_sha256':sha(out/'nine_prediction_freeze.json'),'head_state_sha256':head_sha,'head_fit_joint_sha256':sha(a.head_fit_joint),'rows':201,'videos':9,'precision':'FP32 forward; float64 derived delta and weights','forward_calls':30,'batch_size':32,'batch_sizes':[32]*6+[9],
       'sequence_per_fixed_batch':['F','C','C','F'],'dummy_label_checks':dummy_checks,'semantic_batch_checks':semantic,'diagnostics':diag,'input_access_journal':journal,'rng_unchanged':True,'task_labels_read':False,'head_fit_or_optimizer_updates':False,'headEVAL201_inputs_used':True,'headEVAL201_labels_used':False,'DEV_TEST_CAL_labels_used':False,'actual_performance_or_gain_measured':False,'final_budget':final}
    write(out/'actual_candidate_collection_receipt.json',receipt)
    print('ACTUAL_CANDIDATE_COLLECTION_COMPLETE '+json.dumps(receipt),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--asset-base',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--completion-joint',required=True);p.add_argument('--evidence',required=True);p.add_argument('--head-state',required=True);p.add_argument('--head-fit-joint',required=True);p.add_argument('--out',required=True);run(p.parse_args())
