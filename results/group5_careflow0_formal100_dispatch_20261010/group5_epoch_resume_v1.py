"""Restore an exact completed-epoch state without replaying consumed epochs.

Partial epochs after the last durable checkpoint remain original evidence; their
updates are extra cost, never hidden within the 4700 logical update budget.
"""
import math


def history_state(history, batches_per_epoch, total_epochs=100):
    if not history or not 0 < len(history) < total_epochs:
        raise ValueError('Only a nonfinal completed epoch can resume')
    best=float('inf');best_epoch=0
    for epoch,row in enumerate(history,1):
        mse=row['inner_MSE']
        if not isinstance(mse,(int,float)) or not math.isfinite(mse) or mse<0:
            raise ValueError('Invalid frozen INNER history')
        if row['epoch']!=epoch or row['updates']!=epoch*batches_per_epoch:
            raise ValueError('Incomplete or noncontiguous logical epoch history')
        if mse<best:best,best_epoch=mse,epoch
        if row['best_epoch']!=best_epoch or row['best_MSE']!=best:
            raise ValueError('Strict earliest INNER selection differs')
    return dict(completed_epochs=len(history),updates=len(history)*batches_per_epoch,
                best_MSE=best,best_epoch=best_epoch)


def restore(session,saved,source_SHA,origin_plan_SHA,orders,optimizer_names):
    import numpy as np
    import random
    import torch
    from fixed_flow_components_candidate import tensor_sha
    m=saved['metadata'];fold=session.fold
    if m['method']!=session.method or m['fold']!=fold['fold']:
        raise PermissionError('Resume method/fold differs')
    if m['exact_plan_SHA']!=origin_plan_SHA or m['source_SHA']!=source_SHA:
        raise PermissionError('Resume original plan/source identity differs')
    if m['historical_task_weights_used'] or m['outer_labels_decoded']:
        raise PermissionError('Unqualified historical or OUTER-exposed state')
    if m['fit_ids']!=fold['row_ids']['fit'] or m['inner_ids']!=fold['row_ids']['inner']:
        raise PermissionError('Resume training/selection rows differ')
    if not np.array_equal(saved['orders'],orders) or saved['optimizer_index_to_name']!=optimizer_names:
        raise PermissionError('Resume orders or optimizer ownership differs')
    if (saved['statistics'] is None)!=(session.stats is None):
        raise PermissionError('Resume FIT statistics scope differs')
    if session.stats is not None:
        flatten=lambda d:{modality+'.'+name:value for modality,values in d.items() for name,value in values.items()}
        if tensor_sha(flatten(saved['statistics']))!=tensor_sha(flatten(session.stats)):
            raise PermissionError('Resume FIT statistics bytes differ')
    state=history_state(m['history'],(fold['rows']['fit']+31)//32)
    if m['updates']!=state['updates'] or saved['scheduler']['last_epoch']!=state['updates']:
        raise ValueError('Resume scheduler/update budget differs')
    best_row=m['history'][state['best_epoch']-1]
    if tensor_sha(saved['model'])!=m['history'][-1]['state_SHA'] or tensor_sha(saved['selected_model'])!=best_row['state_SHA']:
        raise ValueError('Resume current/selected model identity differs')
    p=saved['selected_inner_prediction']
    if p.shape!=(fold['rows']['inner'],) or not np.isfinite(p).all():
        raise ValueError('Resume selected INNER prediction differs')
    session.model.load_state_dict(saved['model'],strict=True)
    session.optimizer.load_state_dict(saved['optimizer'])
    session.scheduler.load_state_dict(saved['scheduler'])
    reconstruction_journal=list(session.guard.journal)
    session.guard.journal=list(saved['guard_journal'])
    r=saved['rng'];random.setstate(r['python']);np.random.set_state(r['numpy'])
    torch.set_rng_state(r['torch']);torch.cuda.set_rng_state_all(r['cuda'])
    return dict(**state,history=list(m['history']),selected_model=saved['selected_model'],
                selected_inner_prediction=p,reconstruction_journal=reconstruction_journal)
