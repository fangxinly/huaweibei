"""Fresh samefold selected-parent caches and the original two addon objectives.

No historical parent/cache/scales are accepted. Cache creation indexes FIT
targets only; INNER targets are exposed solely by the separate selection guard.
"""
import copy
import json
from pathlib import Path
from types import MethodType


def qualified_parent(method,fold,fold_spec,receipt_path,checkpoint):
    from group5_test_selected_contract_v1 import sha,validate_parent
    r=json.loads(Path(receipt_path).read_bytes())
    validate_parent(method,fold,r,fold_spec)
    if not r.get('CPU_original_state_qualified') or r.get('status')!='DIRECT100_COMPLETE_OUTER_UNSCORED_CPU_TRANSPORT_PENDING':
        raise PermissionError('Fresh parent original CPU/completed100 qualification absent')
    if sha(checkpoint)!=r['checkpoint']['SHA']:
        raise PermissionError('Fresh parent original checkpoint bytes differ')
    return r


def restore_parent(session,receipt,checkpoint):
    import torch
    from fixed_flow_components_candidate import tensor_sha
    saved=torch.load(checkpoint,map_location='cpu')
    if saved['metadata']['method']!=receipt['method'] or saved['metadata']['fold']!=receipt['fold']:
        raise PermissionError('Parent checkpoint identity differs')
    state=saved['selected_model']
    if state is None or tensor_sha(state)!=receipt['selected_state_SHA']:
        raise PermissionError('Selected parent state differs')
    session.model.load_state_dict(state,strict=True);session.model.eval().requires_grad_(False)
    if receipt['method']=='old_A_teacher':session.model.dberta.own_flow.set_epoch(receipt['best_epoch'])
    return session


def first_anchored(core,batch):
    # Exact first_state recipe from the saved cache_original.py; new IDs/parent.
    import torch
    from encoder_adapter import content_mask,encode_masked
    ids,visual,acoustic,dummy,mask=batch;visual=visual.squeeze(1);acoustic=acoustic.squeeze(1);valid=content_mask(mask)
    text=core.LayerNorm_l(core.proj_l(core.model(ids,attention_mask=mask)[0]))*valid[...,None]
    def norm(v,n):return ((v-getattr(core,'v6_'+n+'_mean'))/getattr(core,'v6_'+n+'_std'))*getattr(core,'v6_'+n+'_active')*valid[...,None]
    audio=core.proj_a(norm(acoustic,'audio').transpose(1,2)).permute(2,0,1);vision=core.proj_v(norm(visual,'visual').transpose(1,2)).permute(2,0,1)
    audio=core.LayerNorm_a(encode_masked(core.transa,audio,valid).transpose(0,1))*valid[...,None];vision=core.LayerNorm_v(encode_masked(core.transv,vision,valid).transpose(0,1))*valid[...,None]
    source=torch.stack([text,audio,vision],1);base=core.predictor(core.fusion((source.sum(2)/valid.sum(1)[:,None,None]).flatten(1))).view(-1)
    zero=source.new_zeros((len(source),3,100));first=(source+.5*torch.stack([core.own_flow.forward_fields[j](source[:,j],0.,zero[:,j]) for j in range(3)],1))*valid[:,None,:,None]
    slots=core.own_flow.reader(first,valid)['slots'].mean(2)
    return first,valid,base,slots


def cache_message(session,out):
    import numpy as np
    import torch
    from incremental_message import OriginalTail
    from fixed_flow_components_candidate import tensor_sha
    core=session.model.dberta;before=tensor_sha(session.model.state_dict())
    tail=OriginalTail(core).to(session.author.DEVICE);data={}
    with torch.no_grad():
        for role in ('fit','inner','outer'):
            tensors=session.inputs[role].tensors;parts={k:[] for k in ('first','mask','base','slots','p0')}
            for start in range(0,len(tensors[0]),16):
                b=tuple(t[start:start+16].to(session.author.DEVICE) for t in tensors)
                first,mask,base,slots=first_anchored(core,b);p0=tail(first,mask,base,torch.zeros_like(slots))
                for k,v in zip(parts,(first,mask,base,slots,p0)):parts[k].append(v.detach().cpu().numpy())
            for k,values in parts.items():data[role+'_'+k]=np.concatenate(values)
            data[role+'_ids']=np.asarray(session.fold['row_ids'][role])
    if before!=tensor_sha(session.model.state_dict()):raise ValueError('Message cache mutated parent state')
    np.savez(Path(out)/'first_cache.npz',**data)
    torch.save(dict(state={k:v.detach().cpu() for k,v in tail.state_dict().items()},parent_selected_state_SHA=before),Path(out)/'tail.pt')
    return tail,data


def teacher_terminal(core,state,mask,context):
    import torch
    flow=core.own_flow;flow.valid=mask.to(state.dtype)
    velocity=torch.stack([flow.forward_fields[m](state[:,m],.5,context[:,m]) for m in range(3)],1)
    final=(state+.5*velocity)*mask[:,None,:,None]
    return flow.read_prediction(final,flow.reader(final,mask.bool()),lambda x:core.predictor(core.fusion(x))).view(-1)


def cache_old_A(session,out):
    import numpy as np
    import torch
    from encoder_adapter import forward_batch
    from fixed_flow_components_candidate import tensor_sha
    core=session.model.dberta;flow=core.own_flow;original=flow.update_context;captured={}
    before=tensor_sha(session.model.state_dict())
    def hook(self,old,relation):
        if self._calls==0:captured.update(state=relation['stage_states'].detach(),pooled_state=relation['slots'].mean(2).detach(),old_context=old.detach(),mask=self.valid.bool().detach())
        return original(old,relation)
    flow.update_context=MethodType(hook,flow);data={};maxerror=0.
    try:
        for role in ('fit','inner','outer'):
            tensors=session.inputs[role].tensors;parts={}
            for start in range(0,len(tensors[0]),32):
                b=tuple(t[start:start+32].to(session.author.DEVICE) for t in tensors)
                with torch.no_grad():prediction=forward_batch(session.model,b)[0].view(-1)
                c={k:v.clone() for k,v in captured.items()};cx=(.5*c['old_context']).detach().requires_grad_(role=='fit')
                with torch.set_grad_enabled(role=='fit'):p0=teacher_terminal(core,c['state'],c['mask'],cx)
                err=float((p0.detach()-prediction).abs().max());maxerror=max(maxerror,err)
                if err>2e-5:raise ValueError('Fresh teacher first-coordinate replay differs')
                values=dict(c,reference_prediction=p0.detach())
                if role=='fit':
                    y=session.fit.tensors[3][start:start+32].to(session.author.DEVICE).view(-1)
                    jac=torch.autograd.grad(p0.sum(),cx,retain_graph=True)[0].detach()
                    gradient=torch.autograd.grad((p0-y).square().sum(),cx)[0].detach()
                    values.update(y=y,teacher_gradient=gradient,reference_jacobian=jac)
                for k,v in values.items():parts.setdefault(k,[]).append(v.cpu().numpy())
            arrays={k:np.concatenate(v) for k,v in parts.items()};arrays['row_ids']=np.asarray(session.fold['row_ids'][role]);data[role]=arrays
            np.savez(Path(out)/(role+'_cache.npz'),**arrays)
    finally:flow.update_context=original
    if before!=tensor_sha(session.model.state_dict()) or any(v.grad is not None for v in session.model.parameters()):raise ValueError('Cache mutated teacher parameters')
    if any('y' in data[role] for role in ('inner','outer')):raise PermissionError('NonFIT targets entered cache')
    rms=np.sqrt(np.mean(data['fit']['teacher_gradient'].astype(np.float64)**2,axis=(0,2))).astype(np.float32)
    if not np.isfinite(rms).all() or not (rms>=1e-8).all():raise ValueError('Invalid fresh FIT gradient scales')
    np.save(Path(out)/'FIT_gradient_rms.npy',rms,allow_pickle=False)
    return data,rms,maxerror


def make_old_A_learner(core,rms,scales):
    import torch
    from torch import nn
    from counterfactual_flow_model import WholeStateFlow
    from task_gradient_vector_candidate_v3 import TaskGradientVectorFeedback
    from finite_task_risk_feedback_v1 import FiniteTaskRiskFeedback
    from finite_task_risk_runtime_v1 import FrozenCoordinateLearner
    # Reconstruct original initialization sequence without opening any old root.
    model=FrozenCoordinateLearner.__new__(FrozenCoordinateLearner);nn.Module.__init__(model)
    model.flow=WholeStateFlow('none');model.flow.load_state_dict(core.own_flow.state_dict())
    model.fusion=nn.Sequential(nn.Linear(300,150),nn.ReLU(),nn.Linear(150,100))
    model.predictor=nn.Sequential(nn.Linear(100,150),nn.ReLU(),nn.Linear(150,1))
    model.fusion.load_state_dict(core.fusion.state_dict());model.predictor.load_state_dict(core.predictor.state_dict())
    model.requires_grad_(False);model.eval();model.donor=copy.deepcopy(model.flow.donor_feedback);model.donor.requires_grad_(True)
    model.feedback=TaskGradientVectorFeedback('fixed',relative_norm2=.1);model.feedback.set_train_fitted_gradient_rms(rms)
    model.feedback=FiniteTaskRiskFeedback('fixed')
    model.feedback.set_train_scales(scales['residual_rms'],scales['message_rms'],scales['beta'])
    return model.to(next(core.parameters()).device)


def fit_old_A_scales(core,cache,seed=128):
    import numpy as np
    import torch
    from counterfactual_flow_model import PAIRS
    n=len(cache['y']);holdout=max(1,int(round(.1*n)))
    order=np.random.RandomState(seed+173).permutation(n);fit=np.sort(order[:n-holdout]);held=np.sort(order[n-holdout:])
    rho=cache['reference_prediction']-cache['y'];residual=float(np.sqrt(np.mean(rho[fit].astype(np.float64)**2)))
    sums=np.zeros(6,dtype=np.float64);device=next(core.parameters()).device
    with torch.no_grad():
        for start in range(0,len(fit),64):
            pool=torch.as_tensor(cache['pooled_state'][fit[start:start+64]],device=device)
            for i,(m,j) in enumerate(PAIRS):
                phi=torch.cat([pool[:,m],pool[:,j],.5*(pool[:,m]-pool[:,j]),.5*(pool[:,m]+pool[:,j])],1)
                sums[i]+=core.own_flow.donor_feedback[i](phi).double().square().sum().item()
    rms=np.sqrt(sums/(len(fit)*100));jac=cache['reference_jacobian'][fit].astype(np.float64)
    sensitivity=.125**2*np.sum(np.repeat(np.sum(jac*jac,axis=2),2,axis=1)*rms[None,:]**2,axis=1);median=float(np.median(sensitivity))
    if not np.isfinite(residual) or residual<=1e-6 or not np.isfinite(rms).all() or (rms<=1e-6).any() or median<=1e-12:
        raise ValueError('Original fixedA FIT-only scale recipe not valid for this fold; no fallback floor')
    return dict(residual_rms=residual,message_rms=rms.tolist(),beta=residual**2/(2*median),
        median_reference_normalized_message_sensitivity_norm2=median,fit_rows=fit.tolist(),heldout_rows=held.tolist(),seed=seed+173,
        label_scope='Outer-fold FIT only; head holdout does not constitute a separate whole-pipeline CV')
