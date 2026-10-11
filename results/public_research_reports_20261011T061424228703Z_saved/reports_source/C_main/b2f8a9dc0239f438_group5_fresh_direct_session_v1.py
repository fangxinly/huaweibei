"""Fresh fold constructors for CaReFlow, Anchored parent and latest A/B.

Run one method in one process with its separately assembled source bundle. This
is source preparation: a separate frozen native precheck plan is required. The
old-A teacher and both second-stage adapters have separate pending pipelines.
"""
import json
import pickle
from pathlib import Path
from types import MethodType, SimpleNamespace

METHODS=('careflow','anchored_parent','factorized_aux','regression_aux','old_A_teacher')


def construct(method, bundle, assets, plan_path, expected_plan_SHA, fold_number):
    # Scientific imports occur only after the small exact execution-plan gate.
    from group5_test_selected_contract_v1 import sha
    bundle,assets,plan_path=Path(bundle),Path(assets),Path(plan_path)
    if sha(plan_path)!=expected_plan_SHA:
        raise PermissionError('Constructor plan SHA differs')
    plan=json.loads(plan_path.read_bytes())
    if plan.get('status') not in ('GROUP5_DIRECT_NATIVE_PRECHECK_FROZEN','GROUP5_DIRECT_TRAIN_FROZEN'):
        raise PermissionError('Preparation cannot instantiate a scientific model')
    if method not in METHODS or method not in plan['authorized_methods'] or fold_number not in range(5):
        raise PermissionError('Constructor method/fold outside scope')
    if plan['fold']!=fold_number or plan['old_task_weight_reuse'] or plan['task_seed']!=128:
        raise PermissionError('Fresh same-fold common-seed scope required')
    for rel,h in plan['source_SHA'].items():
        if sha(bundle/rel)!=h:raise ValueError('Exact source differs: '+rel)
    for rel,h in plan['asset_SHA'].items():
        if sha(assets/rel)!=h:raise ValueError('Exact asset differs: '+rel)
    if sha(bundle/'split.json')!=plan['split_SHA']:
        raise PermissionError('Role identity source differs')
    split=json.loads((bundle/'split.json').read_bytes())
    from fold_contract import FoldGuard,validate_folds
    validate_folds(split['canonical_row_ids'],split['folds'])
    fold=split['folds'][fold_number]
    if split['rows']!=2195 or split['videos']!=93:
        raise PermissionError('Human-authorized pooled scope differs')
    import torch
    from transformers import get_linear_schedule_with_warmup
    from paired_fulltrain_session_candidate import load_author_components,verify_public_encoder,REMOVED_AUTHOR
    from encoder_adapter import fit_statistics,install
    from fixed_flow_components_candidate import forward_fixed,tensor_sha
    with (assets/'assets/mosi.pkl').open('rb') as stream:
        data=pickle.load(stream)
    records=list(data['train'])+list(data['dev'])+list(data['test']);del data
    guard=FoldGuard(records,fold);del records
    author,argv=load_author_components(bundle,assets);author.set_random_seed(128)
    inputs={role:author.get_appropriate_dataset(guard.inputs(role)) for role in ('fit','inner','outer')}
    updates=100*((fold['rows']['fit']+31)//32)
    if plan['updates']!=updates:raise ValueError('Tail-inclusive actual schedule differs')
    model,discard_optimizer,discard_scheduler=author.prep_for_training(updates)
    for parameter in model.parameters():parameter.requires_grad_(True)
    matched=verify_public_encoder(model,assets/'assets/deberta-v3-base')
    for encoder in (model.dberta.transa,model.dberta.transv):
        encoder.embed_positions._float_tensor.zero_()
    stats=None
    if method!='careflow':
        stats=fit_statistics(inputs['fit'])
        removed=REMOVED_AUTHOR if method!='old_A_teacher' else tuple(n for n in REMOVED_AUTHOR if n!='pooler')
        for name in removed:delattr(model.dberta,name)
        if method=='old_A_teacher':
            from counterfactual_flow_model import WholeStateFlow
            flow=WholeStateFlow('none')
        elif method=='anchored_parent':
            from anchored_flow import AnchoredFlow
            flow=AnchoredFlow()
        else:
            from candidate_adapter import make_candidate
            flow=make_candidate(method)
        model.dberta.own_flow=flow
        install(model.dberta,stats)
        if method=='old_A_teacher':
            from encoder_adapter import forward_v6
            model.dberta.forward=MethodType(forward_v6,model.dberta)
        else:
            model.dberta.forward=MethodType(forward_fixed,model.dberta)
    model.to(author.DEVICE)
    model.dberta.model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={'use_reentrant':False,'preserve_rng_state':True})
    del discard_optimizer,discard_scheduler
    if method=='old_A_teacher':
        # Exact optimizer recipe from SHA-bound assets/run_control_baseline.py.
        # Importing that historical CLI would import an obsolete deployment;
        # invoke its verified optimizer function alone from a static extraction.
        from group5_teacher_optimizer_v1 import optimizer_for
        optimizer,scheduler=optimizer_for(author,model,updates)
        groups=None
    elif method in ('factorized_aux','regression_aux'):
        from candidate_adapter import optimizer_groups
        groups,_=optimizer_groups(model)
    else:
        gain=model.dberta.own_flow.gain if method=='anchored_parent' else None
        named=[(n,p) for n,p in model.named_parameters() if p is not gain]
        nd=('bias','LayerNorm.bias','LayerNorm.weight')
        groups=[dict(params=[p for n,p in named if not any(k in n for k in nd)],weight_decay=.01),
                dict(params=[p for n,p in named if any(k in n for k in nd)],weight_decay=0.)]
        if gain is not None:groups.append(dict(params=[gain],lr=.001,weight_decay=0.))
    if groups is not None:optimizer=torch.optim.AdamW(groups,lr=1e-5)
    identities=[id(p) for group in optimizer.param_groups for p in group['params']]
    if len(identities)!=len(set(identities)) or set(identities)!={id(p) for p in model.parameters() if p.requires_grad}:
        raise RuntimeError('Optimizer ownership differs')
    if method!='old_A_teacher':
        scheduler=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=int(updates*.1),num_training_steps=updates)
    fit=author.get_appropriate_dataset(guard.supervision('fit'))
    y=fit.tensors[3].reshape(-1)
    if len(y)!=fold['rows']['fit'] or not torch.isfinite(y).all() or (y.abs()>3).any():
        raise ValueError('Original fold FIT targets invalid')
    return SimpleNamespace(model=model,optimizer=optimizer,scheduler=scheduler,author=author,
                           guard=guard,fit=fit,inputs=inputs,stats=stats,method=method,
                           clean_state_SHA=tensor_sha(model.state_dict()),public_encoder_matched=matched,
                           exact_plan_SHA=expected_plan_SHA,fold=fold,author_argv=argv)


def loss(session,batch):
    if session.method=='old_A_teacher':
        from encoder_adapter import forward_batch
        from counterfactual_flow_model import CONFIG
        p=forward_batch(session.model,batch)[0].view(-1)
        a=session.model.dberta.last_losses
        parts=dict(main_mse=(p-batch[3].view(-1)).square().mean())
        pairs=(('flow_matching_weight','flow_matching'),('cycle_weight','cycle_reconstruction'),
               ('unimodal_weight','unimodal_sentiment'),('variance_weight','variance_floor'),
               ('pair_weight','pair_sentiment'),('utility_weight','utility_calibration'))
        value=parts['main_mse']
        for weight,key in pairs:
            value=value+CONFIG[weight]*a[key];parts[key]=a[key]
        return value,parts
    if session.method=='careflow':
        p,lf,lb=session.author._forward_eval(session.model,batch)
        mse=(p.view(-1)-batch[3].view(-1)).square().mean()
        return mse+.2*lf+.1*lb,dict(main_mse=mse,forward=lf,backward=lb)
    from fixed_flow_components_candidate import forward_batch
    p=forward_batch(session.model,batch)
    if session.method=='anchored_parent':
        from anchored_flow import objective
        return objective(p,batch[3],session.model.dberta.own_flow.last_base)
    from candidate_adapter import candidate_objective
    value=candidate_objective(session.model,p,batch[3])
    return value,dict(candidate_objective=value)


def prediction(session,batch):
    if session.method=='old_A_teacher':
        from encoder_adapter import forward_batch
        return forward_batch(session.model,batch)[0].view(-1)
    if session.method=='careflow':return session.author._forward_eval(session.model,batch)[0].view(-1)
    from fixed_flow_components_candidate import forward_batch
    return forward_batch(session.model,batch).view(-1)
