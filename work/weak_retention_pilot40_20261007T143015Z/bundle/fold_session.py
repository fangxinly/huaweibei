"""Fresh pooled video-fold constructors; normalization fits FIT only."""
import hashlib
import pickle
from pathlib import Path
from types import MethodType,SimpleNamespace
import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from encoder_adapter import fit_statistics,install
from fixed_flow_components_candidate import tensor_sha,forward_fixed,forward_batch
from paired_fulltrain_session_candidate import load_author_components,verify_public_encoder,REMOVED_AUTHOR
from fold_contract import FoldGuard,sha
from anchored_flow import AnchoredFlow,objective


def construct(method, bundle, assets, protocol, split, fold_number):
    if protocol['status'] not in ('GROUP5_EXECUTION_FROZEN','GROUP5_PRECHECK_ONLY_FROZEN','PILOT40_EXECUTION_FROZEN') or not protocol['execution_enabled']:
        raise PermissionError('Preparation source cannot construct a model')
    if protocol.get('authorized_methods') and method not in protocol['authorized_methods']:
        raise PermissionError('Method outside frozen scope')
    if method not in ('anchored_flow','careflow'):raise ValueError('Unfrozen method')
    fold=split['folds'][fold_number]
    for rel,h in protocol['source_sha256'].items():
        if sha(bundle/rel)!=h:raise ValueError('Source digest mismatch: '+rel)
    for rel,h in protocol['asset_sha256'].items():
        if sha(assets/rel)!=h:raise ValueError('Asset digest mismatch: '+rel)
    with (assets/'assets/mosi.pkl').open('rb') as f:data=pickle.load(f)
    records=list(data['train'])+list(data['dev'])+list(data['test']);del data
    guard=FoldGuard(records,fold)
    author,_=load_author_components(bundle,assets);author.set_random_seed(128)
    inputs={r:author.get_appropriate_dataset(guard.inputs(r)) for r in ('fit','inner','outer')}
    updates=protocol['fold_budgets'][fold_number]['updates']
    model,discard_optimizer,discard_scheduler=author.prep_for_training(updates)
    matched=verify_public_encoder(model,assets/'assets/deberta-v3-base')
    for encoder in (model.dberta.transa,model.dberta.transv):encoder.embed_positions._float_tensor.zero_()
    stats=None
    if method=='anchored_flow':
        stats=fit_statistics(inputs['fit'])
        for name in REMOVED_AUTHOR:delattr(model.dberta,name)
        model.dberta.own_flow=AnchoredFlow();install(model.dberta,stats)
        model.dberta.forward=MethodType(forward_fixed,model.dberta)
    model.to(author.DEVICE)
    model.dberta.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
    del discard_optimizer,discard_scheduler
    for param in model.parameters():param.requires_grad_(True)
    # One scalar needs its own scale; it is frozen here, not selected by CV scores.
    gain=model.dberta.own_flow.gain if method=='anchored_flow' else None
    named=[(n,p) for n,p in model.named_parameters() if p is not gain]
    decay=[p for n,p in named if not any(k in n for k in ('bias','LayerNorm.bias','LayerNorm.weight'))]
    no_decay=[p for n,p in named if any(k in n for k in ('bias','LayerNorm.bias','LayerNorm.weight'))]
    groups=[{'params':decay,'weight_decay':.01},{'params':no_decay,'weight_decay':0.}]
    if gain is not None:groups.append({'params':[gain],'weight_decay':0.,'lr':1e-3})
    optimizer=AdamW(groups,lr=1e-5)
    optimized=[id(p) for g in optimizer.param_groups for p in g['params']]
    if len(optimized)!=len(set(optimized)) or set(optimized)!={id(p) for p in model.parameters()}:
        raise ValueError('Optimizer coverage/ownership mismatch')
    scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(updates*.1),num_training_steps=updates)
    fit=author.get_appropriate_dataset(guard.supervision('fit'))
    labels=fit.tensors[3].reshape(-1)
    if len(labels)!=fold['rows']['fit'] or not torch.isfinite(labels).all() or (labels.abs()>3).any():
        raise ValueError('Invalid original FIT targets')
    return SimpleNamespace(model=model,optimizer=optimizer,scheduler=scheduler,author=author,
                           guard=guard,fit=fit,inputs=inputs,stats=stats,
                           clean_state_sha256=tensor_sha(model.state_dict()),
                           public_encoder_matched_tensors=matched)


def loss(session, method, batch):
    if method=='anchored_flow':
        p=forward_batch(session.model,batch)
        return objective(p,batch[3],session.model.dberta.own_flow.last_base)
    p,lf,lb=session.author._forward_eval(session.model,batch)
    mse=(p.view(-1)-batch[3].view(-1)).square().mean()
    return mse+.2*lf+.1*lb,{'main_mse':mse,'forward':lf,'backward':lb}


def prediction(session, method, batch):
    if method=='anchored_flow':return forward_batch(session.model,batch)
    return session.author._forward_eval(session.model,batch)[0].view(-1)
