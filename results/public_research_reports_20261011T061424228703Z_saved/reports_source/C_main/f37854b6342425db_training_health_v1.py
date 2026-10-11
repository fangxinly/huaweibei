"""Gradient/update instrumentation and fixed-loss ablation on original flow.

Diagnostics take no validation/test labels. Gradient magnitudes are normalized
by parameter count; actual Adam updates are recorded separately. Large tensors
use fixed evenly spaced update samples, explicitly not exact full update RMS.
"""
import torch
from torch.nn import functional as F

def group_name(name):
    if name.endswith('own_flow.gain'):return 'gain'
    for part in ('forward_fields','reader','role_head','message'):
        if '.own_flow.'+part+'.' in name:return part
    if name.startswith('dberta.model.'):return 'text_encoder'
    return 'other_encoder_decoder'

def gradient_summary(model):
    groups={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        v=groups.setdefault(group_name(name),dict(numel=0,tensors=0,missing_gradients=0,nonzero_gradient_tensors=0,gradient_L1=0.,gradient_L2_squared=0.))
        v['numel']+=p.numel();v['tensors']+=1
        if p.grad is None:v['missing_gradients']+=1;continue
        assert torch.isfinite(p.grad).all(),name
        norm=float(torch.linalg.vector_norm(p.grad.detach()))
        v['nonzero_gradient_tensors']+=int(norm!=0)
        v['gradient_L1']+=float(p.grad.detach().abs().sum());v['gradient_L2_squared']+=norm*norm
    for v in groups.values():
        v['gradient_mean_absolute']=v['gradient_L1']/v['numel']
        v['gradient_RMS']=(v['gradient_L2_squared']/v['numel'])**.5
    return groups

def before_update(model):
    result={}
    for name,p in model.named_parameters():
        if not p.requires_grad:continue
        count=min(p.numel(),512)
        indices=torch.linspace(0,p.numel()-1,count,device=p.device).round().long()
        result[name]=dict(indices=indices,values=p.detach().flatten()[indices].cpu().clone(),numel=p.numel())
    return result

def after_update(model,before):
    groups={}
    for name,p in model.named_parameters():
        if name not in before:continue
        old=before[name];now=p.detach().flatten()[old['indices']].cpu()
        delta=now-old['values']
        v=groups.setdefault(group_name(name),dict(update_sample_count=0,squared_update_sum=0.,squared_parameter_sum=0.,changed_sample_values=0,fully_sampled_tensors=0,sampled_tensors=0))
        v['update_sample_count']+=len(delta);v['squared_update_sum']+=float(delta.double().square().sum())
        v['squared_parameter_sum']+=float(old['values'].double().square().sum());v['changed_sample_values']+=int((delta!=0).sum())
        v['fully_sampled_tensors']+=int(old['numel']<=512);v['sampled_tensors']+=int(old['numel']>512)
    for v in groups.values():
        v['sampled_actual_update_RMS']=(v['squared_update_sum']/v['update_sample_count'])**.5
        v['sampled_relative_update_norm']=None if v['squared_parameter_sum']==0 else (v['squared_update_sum']/v['squared_parameter_sum'])**.5
        v['includes_weight_decay']=True
    return groups

def original_task_objective(flow,p,y,*,kind,role):
    if role!='TRAIN' or not flow.training:raise ValueError('Official TRAIN update only')
    if kind not in ('huber1','mse'):raise ValueError('Predeclared huber1 or mse required')
    if p.shape!=y.shape or not torch.isfinite(y).all():raise ValueError('Finite aligned training labels required')
    task=F.huber_loss(p,y,delta=1.) if kind=='huber1' else F.mse_loss(p,y)
    return task+.01*flow.last_context.square().mean()
