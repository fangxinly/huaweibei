"""Prepared independent audit of one stopped, nonfinal complete composite epoch.

No CLI dispatch is supplied here. A separately frozen audit driver must bind
the original files, runtime, resources, actual lease and a new audit token.
Import and metadata validation use no scientific libraries. Actual tensor
audit imports them only after original identity and available-RAM checks.
This is not a continuous-versus-restored CPU/CUDA qualification.
"""
import datetime as dt
import json
import math
from pathlib import Path
import sys
import zipfile

from group5_release_transport_v1 import digest
from group5_composite_epoch_resume_v2 import history_state


def metadata(plan, construction, saved, fold, plan_sha):
    """Check durable history and label scope; never recompute a selection score."""
    if plan.get('status') != 'GROUP5_COMPOSITE_TRAIN_FROZEN' or not plan.get('execution_enabled'):
        raise PermissionError('Only an original separately frozen formal composite is eligible')
    if plan.get('resume') is not None and plan.get('resume') is not False:
        raise PermissionError('Recovery chains require another independently qualified audit')
    method=plan['method'];m=saved['metadata']
    if method not in ('anchored_message20','old_fixed_A') or type(plan['fold']) is not int or not 0<=plan['fold']<5:
        raise PermissionError('Original method/fold outside fixed scope')
    for key in ('method','fold','source_SHA','split_SHA','clean_initial_state_SHA','initial_rng_SHA',
                'parent_checkpoint_SHA','parent_selected_state_SHA','cache_SHA','frozen_tail_or_teacher_SHA','fit_ids','inner_ids'):
        if m[key]!=construction[key]:raise PermissionError('Original construction and checkpoint differ: '+key)
    for source in (construction,m):
        if type(source['fold']) is not int or source.get('public_pretraining_fresh_start') is not True:
            raise PermissionError('Original construction has no explicit fresh public-pretraining scope')
        if source['exact_plan_SHA']!=plan_sha or source['exact_dispatch_plan_SHA']!=plan_sha:
            raise PermissionError('Only a bound fresh original plan can pass this version')
        if source['historical_task_weights_used'] is not False or source['outer_labels_decoded'] is not False:
            raise PermissionError('Historical weights or held-out label scope violated')
    if (m['method']!=method or m['fold']!=plan['fold'] or m['source_SHA']!=plan['source_SHA'] or
            m['split_SHA']!=plan['split_SHA'] or plan['old_task_weight_reuse']):
        raise PermissionError('Original formal provenance differs')
    if m['parent_checkpoint_SHA']!=plan['parent_checkpoint']['SHA'] or m['cache_SHA']!=plan['cache_reference']['files_SHA']:
        raise PermissionError('Original same-fold parent/cache differs')
    if m['fit_ids']!=fold['row_ids']['fit'] or m['inner_ids']!=fold['row_ids']['inner']:
        raise PermissionError('Original exact role identities differ')
    epochs=20 if method=='anchored_message20' else 100
    per=(fold['rows']['fit']+31)//32
    if type(plan['updates']) is not int or plan['updates']!=epochs*per:
        raise PermissionError('Original tail-inclusive full budget differs')
    history=m['history']
    for row in history:
        if type(row.get('epoch')) is not int or type(row.get('updates')) is not int:
            raise ValueError('Boolean/fractional epoch or update count is invalid')
        if method=='old_fixed_A' and (type(row.get('inner_MSE')) not in (int,float) or not math.isfinite(row['inner_MSE']) or row['inner_MSE']<0):
            raise ValueError('Original INNER history is not a finite nonnegative objective')
        if method=='old_fixed_A' and (type(row.get('best_epoch')) is not int or type(row.get('best_MSE')) not in (int,float) or not math.isfinite(row['best_MSE']) or row['best_MSE']<0):
            raise ValueError('Original strict-earliest selection summary is invalid')
    state=history_state(method,history,per)
    if type(m['updates']) is not int or m['updates']!=state['updates']:
        raise ValueError('Checkpoint is not at the durable full-epoch boundary')
    if saved.get('completed_epoch_recovery_only') is not True:
        raise PermissionError('Original atomic complete-epoch checkpoint marker absent')
    if method=='anchored_message20' and (saved['selected_model'] is not None or saved['selected_inner_prediction'] is not None):
        raise PermissionError('Nonfinal fixed20 message cannot have selected or INNER prediction state')
    if method=='old_fixed_A' and (saved['selected_model'] is None or saved['selected_inner_prediction'] is None):
        raise ValueError('Original selected addon/INNER state is incomplete')
    for entry in saved['guard_journal']:
        if type(entry['labels_read']) is not bool:raise PermissionError('Original label-access journal is not explicit')
        if entry['role'] not in ('fit','inner','outer'):
            raise PermissionError('Unknown original label role')
        if entry['labels_read'] and (entry['role']=='outer' or (method=='anchored_message20' and entry['role']=='inner')):
            raise PermissionError('Held-out target access violated composite scope')
    return state


def audit_stopped_epoch(root, plan_sha, checkpoint_sha, checkpoint_bytes, exit_sha):
    """Read the exact original state on CPU. No fit, inference or scoring."""
    root=Path(root);plan_path=root/'plan.json';exit_path=root/'wrapper_exit.json'
    if digest(plan_path)!=plan_sha or digest(exit_path)!=exit_sha:
        raise PermissionError('Bound original plan/observed exit bytes differ')
    plan=json.loads(plan_path.read_bytes());exited=json.loads(exit_path.read_bytes())
    if type(exited.get('natural_exit')) is not int or type(exited.get('child')) is not int:
        raise PermissionError('The original child has no observed natural exit identity')
    if exited['child']<=0:raise PermissionError('Invalid original child identity')
    if plan.get('status')!='GROUP5_COMPOSITE_TRAIN_FROZEN' or (plan.get('resume') is not None and plan.get('resume') is not False):
        raise PermissionError('Only a separately frozen fresh-origin formal stage can be audited')
    for rel,sha in plan['source_SHA'].items():
        path=(root/'source'/rel).resolve()
        if not path.is_relative_to((root/'source').resolve()) or digest(path)!=sha:
            raise PermissionError('Original source inventory differs')
    split=root/'source/split.json'
    if digest(split)!=plan['split_SHA']:raise PermissionError('Original split source differs')
    fold=json.loads(split.read_bytes())['folds'][plan['fold']]
    path=root/'out/complete_composite_full.pt'
    if type(checkpoint_bytes) is not int or path.stat().st_size!=checkpoint_bytes or digest(path)!=checkpoint_sha:
        raise PermissionError('Original durable checkpoint bytes differ')
    meminfo=Path('/proc/meminfo').read_text()
    available=int(next(line.split()[1] for line in meminfo.splitlines() if line.startswith('MemAvailable:')))*1024
    if available<6*1024**3:raise PermissionError('Actual CPU available RAM below6GiB')
    with zipfile.ZipFile(path) as archive:
        if len(archive.namelist())!=len(set(archive.namelist())) or archive.testzip() is not None:
            raise ValueError('Original checkpoint CRC/unique gate failed')
    import numpy as np
    import torch
    from fixed_flow_components_candidate import tensor_sha
    from group5_test_selected_contract_v1 import validate_parent
    torch.set_num_threads(2)
    saved=torch.load(path,map_location='cpu');m=saved['metadata']
    construction=json.loads((root/'out/construction.json').read_bytes())
    state=metadata(plan,construction,saved,fold,plan_sha);method=m['method'];updates=m['updates']
    if digest(plan['parent_qualification']['path'])!=plan['parent_qualification']['SHA']:
        raise PermissionError('Original parent CPU qualification bytes differ')
    parent=json.loads(Path(plan['parent_qualification']['path']).read_bytes());validate_parent(method,m['fold'],parent,fold)
    if parent.get('CPU_original_state_qualified') is not True or parent['checkpoint']['SHA']!=m['parent_checkpoint_SHA']:
        raise PermissionError('Original same-fold parent CPU identity differs')
    if tensor_sha(saved['parent_model'])!=parent['selected_state_SHA'] or parent['selected_state_SHA']!=m['parent_selected_state_SHA']:
        raise ValueError('Original complete frozen parent tensor state differs')
    for original in (saved['parent_model'],saved['model'],saved['selected_model'] or {}):
        if any(t.is_floating_point() and not torch.isfinite(t).all() for t in original.values()):
            raise ValueError('Nonfinite original model tensor')
    combined=lambda addon:{**{'parent.'+k:v for k,v in saved['parent_model'].items()},**{'addon.'+k:v for k,v in addon.items()}}
    if tensor_sha(combined(saved['model']))!=m['history'][-1]['state_SHA']:
        raise ValueError('Durable epoch/current model tensor identity differs')
    if len(saved['optimizers'])!=(1 if method=='anchored_message20' else 2) or len(saved['optimizer_index_to_name'])!=len(saved['optimizers']):
        raise ValueError('Original optimizer inventory is incomplete')
    owners=[]
    for optimizer,mapping in zip(saved['optimizers'],saved['optimizer_index_to_name']):
        ids=[i for group in optimizer['param_groups'] for i in group['params']]
        if len(ids)!=len(set(ids)) or set(map(str,ids))!=set(mapping) or set(optimizer['state'])!=set(ids):
            raise ValueError('Original complete Adam ownership/state differs')
        owners.extend(mapping.values())
        for index,entry in optimizer['state'].items():
            tensor=saved['model'][mapping[str(index)]]
            if set(entry)!={'step','exp_avg','exp_avg_sq'} or float(entry['step'])!=updates:
                raise ValueError('Original complete Adam steps differ')
            for key in ('exp_avg','exp_avg_sq'):
                if entry[key].shape!=tensor.shape or not torch.isfinite(entry[key]).all():
                    raise ValueError('Original complete Adam moments differ')
    expected={n for n in saved['model'] if (not n.endswith('.bias') or not n.startswith('heads.') or '.3.' not in n)} if method=='anchored_message20' else {n for n in saved['model'] if n.startswith(('donor.','feedback.gradient_heads.'))}
    if len(owners)!=len(set(owners)) or set(owners)!=expected:raise ValueError('Original complete trainable ownership differs')
    rng=saved['rng']
    if set(rng)!={'python','numpy','torch','cuda'} or not isinstance(rng['cuda'],list) or not rng['cuda']:
        raise ValueError('Original Python/NumPy/Torch/CUDA RNG inventory absent')
    if rng['torch'].dtype!=torch.uint8 or rng['torch'].ndim!=1 or not rng['torch'].numel():raise ValueError('Original Torch RNG state invalid')
    if any(v.dtype!=torch.uint8 or v.ndim!=1 or not v.numel() for v in rng['cuda']):raise ValueError('Original CUDA RNG state invalid')
    if not np.array_equal(saved['orders'],np.load(root/'out/FIT_orders.npy',allow_pickle=False)):
        raise ValueError('Original complete FIT orders differ')
    if saved['orders'].shape!=((20 if method=='anchored_message20' else 100),fold['rows']['fit']):raise ValueError('Original FIT order shape differs')
    for order in saved['orders']:
        if sorted(order.tolist())!=list(range(fold['rows']['fit'])):raise ValueError('Original FIT order is not an exact permutation')
    for name,sha in m['cache_SHA'].items():
        if digest(root/'out'/name)!=sha:raise PermissionError('Original cache bytes differ')
    statistics=saved['statistics']
    if not isinstance(statistics,dict) or not statistics:raise ValueError('Composite FIT statistics absent')
    flat={modality+'.'+name:value for modality,values in statistics.items() for name,value in values.items()}
    if any(t.is_floating_point() and not torch.isfinite(t).all() for t in flat.values()):raise ValueError('Nonfinite original FIT statistics')
    statistics_sha=tensor_sha(flat)
    if method=='anchored_message20':
        if tensor_sha(saved['frozen_tail'])!=m['frozen_tail_or_teacher_SHA']:raise ValueError('Original frozen message tail differs')
    else:
        frozen={name:value for name,value in saved['model'].items() if name.startswith(('flow.','fusion.','predictor.'))}
        if tensor_sha(frozen)!=m['frozen_tail_or_teacher_SHA']:raise ValueError('Original frozen terminal teacher differs')
        best_row=m['history'][state['best_epoch']-1]
        if tensor_sha(combined(saved['selected_model']))!=best_row['state_SHA']:raise ValueError('Original selected model tensor identity differs')
        for epoch,row in enumerate(m['history'],1):
            frozen=root/'out'/f'INNER_epoch{epoch:03d}.npz'
            freeze=json.loads((root/'out'/f'INNER_epoch{epoch:03d}_freeze.json').read_bytes())
            if digest(frozen)!=row['prediction_SHA'] or freeze['SHA']!=row['prediction_SHA'] or freeze['state_SHA']!=row['state_SHA']:
                raise PermissionError('Original frozen INNER evidence differs')
            with np.load(frozen,allow_pickle=False) as values:
                if values['row_ids'].tolist()!=fold['row_ids']['inner'] or values['model_state_sha256'].item()!=row['state_SHA'] or values['prediction'].shape!=(fold['rows']['inner'],) or not np.isfinite(values['prediction']).all():
                    raise ValueError('Original INNER array identity/shape/finiteness differs')
                if epoch==state['best_epoch'] and not np.array_equal(values['prediction'],saved['selected_inner_prediction']):
                    raise ValueError('Original selected INNER array differs')
    return dict(status='INDEPENDENT_CPU_ORIGINAL_COMPOSITE_EPOCH_STATE_AUDIT_PASS',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                method=method,fold=m['fold'],origin_plan_SHA=plan_sha,source_SHA=plan['source_SHA'],split_SHA=plan['split_SHA'],
                original_child=exited['child'],observed_natural_exit=exited['natural_exit'],original_exit_SHA=exit_sha,
                checkpoint_SHA=checkpoint_sha,bytes=checkpoint_bytes,completed_epochs=state['completed_epochs'],updates=updates,
                parent_checkpoint_SHA=m['parent_checkpoint_SHA'],cache_SHA=m['cache_SHA'],statistics_SHA=statistics_sha,
                CPU_available_RAM_bytes=available,Adam_owned_tensors=len(owners),outer_labels_decoded=False,
                inner_labels_decoded=any(j['labels_read'] for j in saved['guard_journal'] if j['role']=='inner'),
                new_fit_inference_score_or_task_target_decode=False,CUDA_recovery_qualified=False,
                audit_is_not_Release_restore_or_recovery_dispatch_qualification=True,argv=[sys.executable]+sys.argv)
