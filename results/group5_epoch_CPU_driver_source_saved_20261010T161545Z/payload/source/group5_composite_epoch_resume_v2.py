"""Prepared complete-epoch restoration for the two fresh composite stages.

Original checkpoint CPU audit, whole Release restoration and partial-epoch
cost accounting must be qualified separately before this helper is called.
This source is inert; it does not load files or instantiate a scientific model.
"""
import math

def history_state(method,history,per):
    epochs=20 if method=='anchored_message20' else 100 if method=='old_fixed_A' else 0
    if not epochs or not 0<len(history)<epochs:raise ValueError('Only an incomplete composite with complete epochs can resume')
    best=float('inf');which=0
    for epoch,row in enumerate(history,1):
        if row['epoch']!=epoch or row['updates']!=epoch*per:raise ValueError('Noncontiguous composite epoch/update history')
        if method=='anchored_message20':
            if any(k in row for k in ('inner_MSE','best_MSE','best_epoch')):raise PermissionError('Fixed20 message cannot select an INNER checkpoint')
        else:
            v=row['inner_MSE']
            if not isinstance(v,(int,float)) or not math.isfinite(v) or v<0:raise ValueError('Invalid INNER selection history')
            if v<best:best,which=v,epoch
            if row['best_epoch']!=which or row['best_MSE']!=best:raise ValueError('Strict-earliest composite selection differs')
    return dict(completed_epochs=len(history),updates=len(history)*per,best_MSE=best,best_epoch=which)

def restore(model,optimizers,tail,session,saved,source_SHA,origin_plan_SHA,orders,owners,parent_SHA,cache_SHA,expected):
    import numpy as np
    import random
    import torch
    from fixed_flow_components_candidate import tensor_sha
    m=saved['metadata'];fold=session.fold;method=m['method']
    for key in ('method','fold','split_SHA','clean_initial_state_SHA','initial_rng_SHA',
                'parent_checkpoint_SHA','frozen_tail_or_teacher_SHA'):
        if m[key]!=expected[key]:raise PermissionError('Original composite construction differs: '+key)
    for original in (saved['parent_model'],saved['model'],saved.get('selected_model') or {}):
        if any(t.is_floating_point() and not torch.isfinite(t).all() for t in original.values()):
            raise ValueError('Nonfinite original composite tensor')
    if method not in ('anchored_message20','old_fixed_A') or m['fold']!=fold['fold']:
        raise PermissionError('Composite resume family/fold differs')
    if m['exact_plan_SHA']!=origin_plan_SHA or m['source_SHA']!=source_SHA or m['cache_SHA']!=cache_SHA:
        raise PermissionError('Original composite plan/source/cache differs')
    if m['historical_task_weights_used'] or m['outer_labels_decoded'] or m['parent_selected_state_SHA']!=parent_SHA:
        raise PermissionError('Historical or OUTER-exposed composite/parent state')
    if m['fit_ids']!=fold['row_ids']['fit'] or m['inner_ids']!=fold['row_ids']['inner']:
        raise PermissionError('Composite resume exact role IDs differ')
    if tensor_sha(saved['parent_model'])!=parent_SHA or not np.array_equal(saved['orders'],orders) or saved['optimizer_index_to_name']!=owners:
        raise PermissionError('Composite parent/orders/optimizer ownership differs')
    if len(optimizers)!=len(saved['optimizers']) or len(optimizers)!=(1 if method=='anchored_message20' else 2):
        raise ValueError('Complete composite optimizer inventory differs')
    if (saved['statistics'] is None)!=(session.stats is None):raise PermissionError('FIT statistics scope differs')
    if session.stats is not None:
        flatten=lambda d:{modality+'.'+name:value for modality,values in d.items() for name,value in values.items()}
        if tensor_sha(flatten(saved['statistics']))!=tensor_sha(flatten(session.stats)):raise PermissionError('FIT statistic bytes differ')
    state=history_state(method,m['history'],(fold['rows']['fit']+31)//32)
    if m['updates']!=state['updates']:raise ValueError('Composite resume logical update count differs')
    combined=lambda v:{**{'parent.'+k:t for k,t in saved['parent_model'].items()},**{'addon.'+k:t for k,t in v.items()}}
    if tensor_sha(combined(saved['model']))!=m['history'][-1]['state_SHA']:raise ValueError('Composite current model differs')
    if method=='anchored_message20':
        if saved['selected_model'] is not None or saved['selected_inner_prediction'] is not None or tail is None:
            raise PermissionError('Incomplete fixed20 message has a selected/INNER state')
        if tensor_sha(saved['frozen_tail'])!=tensor_sha(tail.state_dict()) or tensor_sha(saved['frozen_tail'])!=m['frozen_tail_or_teacher_SHA']:
            raise PermissionError('Frozen message terminal tail differs')
    else:
        best=m['history'][state['best_epoch']-1]
        if tensor_sha(combined(saved['selected_model']))!=best['state_SHA']:raise ValueError('Composite selected model differs')
        pred=saved['selected_inner_prediction']
        if pred.shape!=(fold['rows']['inner'],) or not np.isfinite(pred).all():raise ValueError('Composite selected INNER prediction differs')
        frozen={n:t for n,t in saved['model'].items() if n.startswith(('flow.','fusion.','predictor.'))}
        if tensor_sha(frozen)!=m['frozen_tail_or_teacher_SHA']:raise PermissionError('Frozen fixedA terminal teacher differs')
    if any(j['labels_read'] for j in saved['guard_journal'] if j['role']=='outer'):
        raise PermissionError('Composite state accessed OUTER targets')
    if method=='anchored_message20' and any(j['labels_read'] for j in saved['guard_journal'] if j['role']=='inner'):
        raise PermissionError('Fixed20 message state accessed INNER targets')
    for opt,mapping in zip(saved['optimizers'],owners):
        ids=[i for group in opt['param_groups'] for i in group['params']]
        if len(ids)!=len(set(ids)) or set(map(str,ids))!=set(mapping) or set(opt['state'])!=set(ids):
            raise ValueError('Complete original Adam ownership/state differs')
        for index,entry in opt['state'].items():
            tensor=saved['model'][mapping[str(index)]]
            if set(entry)!={'step','exp_avg','exp_avg_sq'} or float(entry['step'])!=state['updates']:
                raise ValueError('Original complete Adam update count differs')
            for key in ('exp_avg','exp_avg_sq'):
                if entry[key].shape!=tensor.shape or not torch.isfinite(entry[key]).all():
                    raise ValueError('Original complete Adam moments differ')
    if set(saved['rng'])!={'python','numpy','torch','cuda'} or len(saved['rng']['cuda'])!=torch.cuda.device_count():
        raise PermissionError('Original complete CUDA RNG/device inventory differs')
    model.load_state_dict(saved['model'],strict=True)
    for opt,original in zip(optimizers,saved['optimizers']):opt.load_state_dict(original)
    reconstruction=list(session.guard.journal);session.guard.journal=list(saved['guard_journal'])
    r=saved['rng'];random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch'])
    torch.cuda.set_rng_state_all(r['cuda'])
    return dict(**state,history=list(m['history']),selected_model=saved['selected_model'],selected_inner_prediction=saved['selected_inner_prediction'],reconstruction_journal=reconstruction)
