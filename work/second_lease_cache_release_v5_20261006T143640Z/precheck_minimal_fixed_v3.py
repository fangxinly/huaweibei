"""Unexecuted source candidate. No CLI launches GPU or formal training.

Future execution still requires trusted external provenance, fresh runtime
evidence, confirmed lease and space. No OUTER or INNER true labels are read.
"""
import json, os, random, sys, time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
from precheck_gates_v2 import validate_precheck_evidence, validate_saved_state_metadata

def run_precheck(asset_base, bundle, output, trusted_evidence):
    # Validate before importing Torch/CUDA or constructing a model.
    from role_guard_v1 import file_sha
    bundle, output = Path(bundle), Path(output)
    plan_path = bundle/'runtime_candidate_plan.json'
    plan_sha = file_sha(plan_path)
    permission = validate_precheck_evidence(trusted_evidence, datetime.now(timezone.utc), plan_sha)
    if output.exists():
        raise FileExistsError('NEW_UNIQUE_OUTPUT_REQUIRED_NO_OVERWRITE')
    output.mkdir(parents=True)
    started = time.perf_counter()
    import torch
    from encoder_adapter import content_mask
    from minimal_fixed_runtime_v1 import (construct_public_candidate, tensor_sha,
                                         forward_batch, fit_objective, require_real_gradients)
    torch.cuda.reset_peak_memory_stats()
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    target_uuid = permission['gpu_uuid']
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    actual_uuid = str(getattr(props, 'uuid', ''))
    if actual_uuid.startswith('GPU-'):
        actual_uuid = actual_uuid[4:]
    import subprocess
    queried_uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
    if len(queried_uuids)!=1 or queried_uuids[0].strip()!=target_uuid or torch.cuda.device_count()!=1:
        raise ValueError('CURRENT_SINGLE_DRIVER_GPU_UUID_MISMATCH')
    if actual_uuid and actual_uuid != target_uuid.removeprefix('GPU-'):
        raise ValueError('ACTUAL_TORCH_DEVICE_UUID_MISMATCH')
    uuid_method='torch_properties_and_single_driver' if actual_uuid else 'single_driver_query_Torch2_1_property_unavailable'
    def budget(stage):
        elapsed = time.perf_counter()-started
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        if elapsed > 1200 or max(allocated, reserved) > 6*1024**3:
            raise RuntimeError('ACTUAL_PRECHECK_TIME_OR_GPU_BUDGET_EXCEEDED: '+stage)
        return {'elapsed_seconds': elapsed, 'peak_allocated_bytes': allocated,
                'peak_reserved_bytes': reserved}
    def stats_sha(model):
        return tensor_sha({n:v for n,v in model.state_dict().items()
                           if n.startswith(('dberta.v6_audio_', 'dberta.v6_visual_'))})
    def save_state(path, model, scope, steps):
        # Every parameter and persistent buffer, including public encoder.
        state = {n:v.detach().cpu().clone() for n,v in model.state_dict().items()}
        meta = {'format':'MINIMAL_FIXED_FULL_MODEL_STATE_V1', 'runtime_plan_sha256':plan_sha,
                'fold':0, 'seed':91819, 'scope':scope, 'optimizer_steps':steps,
                'state_sha256':tensor_sha(state), 'fit_statistics_sha256':stats_sha(model)}
        validate_saved_state_metadata(meta,plan_sha,scope,steps)
        with Path(path).open('xb') as f:
            torch.save({'model':state, 'metadata':meta},f)
        return {'file_sha256':file_sha(path), 'bytes':Path(path).stat().st_size,
                'metadata':meta, 'tensors':len(state)}
    def row_batch(session, rows):
        local={int(v):i for i,v in enumerate(session.guard.ids['fit'])}
        index=torch.tensor([local[int(i)] for i in rows],dtype=torch.long)
        return tuple(x[index].to('cuda') for x in session.fit.tensors)
    def inner_predictions(session, dummy):
        session.model.eval(); pieces=[]
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(session.inner_inputs,batch_size=128,shuffle=False):
                values=[x.to('cuda') for x in batch]
                values[3]=torch.full_like(values[3],dummy)
                pieces.append(forward_batch(session.model,tuple(values)).cpu().numpy())
        result=np.concatenate(pieces)
        if result.shape!=(153,) or not np.isfinite(result).all():
            raise ValueError('INNER_DUMMY_INPUT_PREDICTION_NOT_FINITE')
        return result
    session=construct_public_candidate(asset_base,bundle)
    before_stats=stats_sha(session.model)
    initial=save_state(output/'clean_initial_full.pt',session.model,'CLEAN_INITIAL_NO_PRECHECK',0)
    # Restore all random streams only in a separately implemented formal runner.
    torch.save({'torch':session.clean_initial['torch_rng'], 'cuda':session.clean_initial['cuda_rng']},
               output/'clean_initial_torch_rng.pt')
    np_state=np.random.get_state()
    (output/'clean_initial_python_numpy_rng.json').write_text(json.dumps({
        'python_random':random.getstate(), 'numpy_global_random':
        [np_state[0],np_state[1].tolist(),int(np_state[2]),int(np_state[3]),float(np_state[4])]},indent=2),encoding='utf-8')
    del session.clean_initial
    snapshots={'construction_and_initial_full_save':budget('construction')}
    # Metadata-only candidate rows; actual masks are checked below.
    roles=plan['precheck_row_candidates']; gradients=[]; component_gradients=[]; times=[]
    from donor_terminal_mechanism_precheck_v1 import donor_terminal_check
    mechanism=[]
    small=row_batch(session,roles['normal'][:4])
    result,arrays=donor_terminal_check(session,small,'clean_initial')
    np.savez(output/'donor_mechanism_clean_initial.npz',**arrays);mechanism.append(result)
    del small,result,arrays
    budget('initial_donor_mechanism')
    for index,key in enumerate(('normal','mixed_singleton')):
        batch=row_batch(session,roles[key])
        valid=content_mask(batch[4]); lengths=valid.sum(1)
        if key=='mixed_singleton':
            for row in (454,505,620):
                if int(lengths[roles[key].index(row)])!=1:
                    raise ValueError('DECLARED_FIT_SINGLETON_NOT_ACTUAL_SINGLETON')
        session.model.train();session.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); t=time.perf_counter()
        from minimal_fixed_flow_v2 import objective
        prediction=forward_batch(session.model,batch)
        loss=objective(prediction,batch[3],session.model.dberta.last_losses)
        if not torch.isfinite(loss):raise ValueError('NONFINITE_PRIMARY_AND_AUXILIARY_OBJECTIVE')
        parameters=dict(session.model.named_parameters())
        names={'forward':'dberta.own_flow.forward_fields.0.net.2.weight',
               'backward':'dberta.own_flow.backward_fields.0.net.2.weight',
               'reader':'dberta.own_flow.reader.queries',
               'donor':'dberta.own_flow.donor_feedback.0.3.weight'}
        names['decoder']=next(n for n in parameters if n.startswith('dberta.predictor.'))
        probes=[parameters[n] for n in names.values()]
        primary=(prediction.view(-1)-batch[3].view(-1)).square().mean()
        terms=session.model.dberta.last_losses
        auxiliary=(.02*terms['flow_matching']+.01*terms['cycle_reconstruction']
                   +.05*terms['unimodal_sentiment']+.01*terms['variance_floor'])
        components={}
        for label,value in (('primary',primary),('auxiliary',auxiliary)):
            probe_gradients=torch.autograd.grad(value,probes,retain_graph=True,allow_unused=True)
            components[label]={}
            for group,g in zip(names,probe_gradients):
                if g is not None and not torch.isfinite(g).all():
                    raise ValueError('NONFINITE_COMPONENT_GRADIENT: '+label+'/'+group)
                components[label][group]={'none':g is None,'l1':None if g is None else float(g.detach().abs().sum())}
            del probe_gradients
        if not components['primary']['backward']['none'] or components['auxiliary']['backward']['none']:
            raise ValueError('PRIMARY_AND_CYCLE_BACKWARD_FIELD_PATH_MISMATCH')
        if components['primary']['decoder']['none']:
            raise ValueError('TASK_DECODER_NOT_TRAINED_BY_PRIMARY_OBJECTIVE')
        if not components['auxiliary']['decoder']['none']:
            raise ValueError('REMOVED_COUNTERFACTUAL_DECODER_BRANCH_REAPPEARED_IN_AUXILIARY')
        component_gradients.append(components)
        del primary,auxiliary,components,probes,parameters
        loss.backward(); report=require_real_gradients(session.model)
        session.optimizer.step();session.scheduler.step()
        torch.cuda.synchronize()
        times.append({'kind':key,'batch_rows':roles[key], 'seconds':time.perf_counter()-t,
                      'objective':float(loss.detach()),'optimizer_steps_after':index+1,'optimizer_lr_after':[g['lr'] for g in session.optimizer.param_groups],
                      'actual_content_lengths':lengths.cpu().tolist()})
        gradients.append(report);snapshots[key]=budget(key)
        del batch,loss,report,terms,names,prediction
    small=row_batch(session,roles['normal'][:4])
    result,arrays=donor_terminal_check(session,small,'after_two_steps')
    np.savez(output/'donor_mechanism_after_two_steps.npz',**arrays);mechanism.append(result)
    del small,result,arrays
    if sum(x['seconds'] for x in mechanism)>120:raise RuntimeError('COMBINED_MECHANISM_TWO_MINUTE_BUDGET')
    budget('post_step_donor_mechanism')
    # Tail has all 23 FIT rows: forward/backward only; no third step.
    batch=row_batch(session,roles['tail']);session.model.train()
    session.optimizer.zero_grad(set_to_none=True)
    state_before_tail=tensor_sha(session.model.state_dict())
    torch.cuda.synchronize();t=time.perf_counter()
    loss=fit_objective(session.model,batch);loss.backward()
    tail_gradients=require_real_gradients(session.model)
    torch.cuda.synchronize();tail_seconds=time.perf_counter()-t
    if tensor_sha(session.model.state_dict())!=state_before_tail:
        raise ValueError('TAIL_ONLY_BACKWARD_MUST_NOT_UPDATE_MODEL_STATE')
    session.optimizer.zero_grad(set_to_none=True);del batch,loss
    # Full-model scalar replay on original INNER inputs, with two dummy labels.
    state_before_eval=tensor_sha(session.model.state_dict())
    p0=inner_predictions(session,0.0);p_relabel=inner_predictions(session,7.0)
    if np.max(np.abs(p0-p_relabel))!=0:
        raise ValueError('INNER_DUMMY_LABEL_REPLACEMENT_NONZERO')
    if tensor_sha(session.model.state_dict())!=state_before_eval or stats_sha(session.model)!=before_stats:
        raise ValueError('EVAL_STATE_OR_FIT_ONLY_STATISTICS_CHANGED')
    selected=save_state(output/'after_two_steps_full.pt',session.model,'PRECHECK_TWO_STEPS_NOT_FORMAL',2)
    np.savez(output/'inner_precheck_predictions.npz',row_ids=session.guard.ids['inner'],
             prediction=p0,model_state_sha256=np.asarray(state_before_eval))
    original_guard_journal=session.guard.journal
    construction=session.construction_receipt
    del session
    torch.cuda.empty_cache()
    # Construct a fresh instance from public initialization and FIT-only stats.
    reloaded=construct_public_candidate(asset_base,bundle)
    del reloaded.clean_initial
    payload=torch.load(output/'after_two_steps_full.pt',map_location='cpu',weights_only=True)
    validate_saved_state_metadata(payload['metadata'],plan_sha,'PRECHECK_TWO_STEPS_NOT_FORMAL',2)
    if file_sha(output/'after_two_steps_full.pt')!=selected['file_sha256']:
        raise ValueError('OWN_NEW_PRECHECK_WEIGHT_FILE_SHA_CHANGED')
    if tensor_sha(payload['model'])!=selected['metadata']['state_sha256']:
        raise ValueError('COMPLETE_DISK_TENSOR_SHA_MISMATCH')
    if stats_sha(reloaded.model)!=before_stats:
        raise ValueError('FRESH_FIT_STATISTICS_NOT_EXACT')
    reloaded.model.load_state_dict(payload['model'],strict=True)
    if tensor_sha(reloaded.model.state_dict())!=selected['metadata']['state_sha256']:
        raise ValueError('STRICT_RELOADED_FULL_MODEL_STATE_NOT_EXACT')
    replay=inner_predictions(reloaded,0.0)
    difference=float(np.max(np.abs(replay-p0)))
    if difference>1e-6:raise ValueError('INNER_ORIGINAL_INPUT_FULL_DISK_REPLAY_ERROR')
    receipt={'status':'ACTUAL_MINIMAL_FIXED_GPU_PRECHECK_ONLY_COMPLETE_NOT_FORMAL100',
      'actual_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,
      'gpu_uuid':target_uuid,'gpu_uuid_check_method':uuid_method,'donor_mechanism':mechanism,'permission_evidence':permission,'runtime_plan_sha256':plan_sha,
      'construction':construction,'initial_full':initial,'after_two_steps_full':selected,
      'optimizer_steps':2,'normal_and_mixed_gradients':gradients,'component_gradient_probes':component_gradients,'tail_gradients':tail_gradients,
      'tail_optimizer_steps':0,'tail_forward_backward_seconds':tail_seconds,
      'step_times':times,'inner_rows':153,'inner_labels_read':False,
      'inner_dummy_label_replacement_max_error':0.0,'strict_full_disk_replay_max_error':difference,
      'original_guard_journal':original_guard_journal,'reloaded_guard_journal':reloaded.guard.journal,
      'snapshots':snapshots,'final_budget':budget('final'),
      'Torch_input_HVP_or_full_encoder_second_derivative_verified':False,
      'D_or_other_node_CPU_preservation_complete':False,'formal100_or_OUTER_authorized':False,
      'new_performance_scores':False}
    (output/'actual_precheck_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt

if __name__=='__main__':
    raise SystemExit('SOURCE_ONLY_NO_GPU_LAUNCH_ENTRY_POINT_CURRENT_LEASE_NOT_VERIFIED')
