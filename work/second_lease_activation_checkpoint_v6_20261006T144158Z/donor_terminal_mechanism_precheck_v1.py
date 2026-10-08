"""Predeclared one-batch label-free donor-path check; no performance selection."""
import time
import torch
from minimal_fixed_runtime_v1 import forward_batch,tensor_sha

def donor_terminal_check(session,batch,stage):
    begin=time.perf_counter();model=session.model;flow=model.dberta.own_flow
    before=tensor_sha(model.state_dict());was_training=model.training;model.eval()
    values=list(batch);values[3]=torch.zeros_like(values[3]);values=tuple(values)
    original=flow.fixed_context
    captures=[];runs=[]
    def record(old,feedback):
        context=original(old,feedback)
        captures.append({'old':old.detach().cpu(),'feedback':feedback.detach().cpu(),'context':context.detach().cpu()})
        return context
    def zero_donor(old,feedback):
        context=original(old,torch.zeros_like(feedback))
        captures.append({'old':old.detach().cpu(),'feedback':feedback.detach().cpu(),'context':context.detach().cpu()})
        return context
    states=[]
    def reader_record(module,inputs,output):
        states.append(inputs[0].detach().cpu())
    handle=flow.reader.register_forward_hook(reader_record)
    try:
        with torch.no_grad():
            for function in (record,zero_donor,record):
                flow.fixed_context=function;captures.clear();states.clear()
                prediction=forward_batch(model,values).detach().cpu()
                if len(captures)!=1 or len(states)!=2:raise ValueError('EXPECTED_ONE_CONTEXT_AND_TWO_EULER_READERS')
                runs.append({'prediction':prediction,'first_state':states[0],'terminal_state':states[1],**captures[0]})
        maximum=lambda x:float(x.abs().max())
        if maximum(runs[0]['first_state']-runs[1]['first_state'])!=0:raise ValueError('DONOR_ZEROING_CHANGED_FIRST_EULER_STATE')
        replay=max(maximum(runs[0][k]-runs[2][k]) for k in ('prediction','first_state','terminal_state','context','feedback'))
        if replay>1e-6:raise ValueError('RESTORED_FIXED_MECHANISM_REPLAY')
        if any(not torch.isfinite(value).all() for r in runs for value in r.values()):raise ValueError('NONFINITE_DONOR_MECHANISM')
        elapsed=time.perf_counter()-begin
        if elapsed>120:raise RuntimeError('EXTRA_MECHANISM_TWO_MINUTE_BUDGET')
        result={'stage':stage,'seconds':elapsed,'feedback_max':maximum(runs[0]['feedback']),
            'old_context_max':maximum(runs[0]['old']),'context_change_max':maximum(runs[0]['context']-runs[1]['context']),
            'first_euler_state_change_max':maximum(runs[0]['first_state']-runs[1]['first_state']),
            'second_euler_state_change_max':maximum(runs[0]['terminal_state']-runs[1]['terminal_state']),
            'terminal_scalar_change_max':maximum(runs[0]['prediction']-runs[1]['prediction']),
            'restored_fixed_replay_max_error':replay,'forward_calls':3,'labels_replaced_with_zero':True,
            'performance_labels_or_scores_used':False,'weights_before_sha256':before}
        arrays={str(i)+'_'+k:v.numpy() for i,r in enumerate(runs) for k,v in r.items()}
    finally:
        handle.remove()
        # Remove the instance override, restoring the class static method.
        del flow.fixed_context
        model.train(was_training)
    if tensor_sha(model.state_dict())!=before:raise ValueError('MECHANISM_MUST_NOT_CHANGE_PERSISTENT_WEIGHT_OR_BUFFER')
    return result,arrays
