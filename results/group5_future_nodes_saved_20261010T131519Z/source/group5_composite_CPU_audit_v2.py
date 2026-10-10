"""Independent CPU audit of original composite state; no fitting or scoring."""
import argparse,datetime as dt,json,os,sys,zipfile
from pathlib import Path
from group5_release_transport_v1 import digest,write


def audit(root,plan_sha):
    if digest(root/'plan.json')!=plan_sha:raise PermissionError('Original frozen plan differs')
    plan=json.loads((root/'plan.json').read_bytes());r=json.loads((root/'out/actual_stage_receipt.json').read_bytes())
    exit_code=json.loads((root/'wrapper_exit.json').read_bytes())['natural_exit']
    if type(exit_code) is not int or exit_code!=0:raise PermissionError('Original composite child did not exit0')
    from group5_composite_resume_contract_v2 import qualify
    recovery=qualify(plan,r['method'],r['fold'],'train' if plan['status']=='GROUP5_COMPOSITE_TRAIN_FROZEN' else 'precheck')
    origin_plan_sha=recovery['origin_plan_SHA'] if recovery else plan_sha
    if r['exact_plan_SHA']!=origin_plan_sha or r['exact_dispatch_plan_SHA']!=plan_sha:
        raise PermissionError('Original composite origin/dispatch plan differs')
    path=root/'out'/Path(r['checkpoint']['path']).name
    if digest(path)!=r['checkpoint']['SHA'] or path.stat().st_size!=r['checkpoint']['bytes']:raise PermissionError('Original full composite bytes differ')
    available=int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:')))*1024
    if available<6*1024**3:raise PermissionError('Actual CPU available RAM below6GiB')
    with zipfile.ZipFile(path) as z:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip() is not None:raise ValueError('Original full-state CRC/unique differs')
    import numpy as np
    import torch
    from fixed_flow_components_candidate import tensor_sha
    from group5_test_selected_contract_v1 import validate_parent
    torch.set_num_threads(2);saved=torch.load(path,map_location='cpu');m=saved['metadata'];updates=r['updates']
    if m['method'] not in ('anchored_message20','old_fixed_A') or m['method']!=r['method'] or m['fold']!=r['fold']:
        raise PermissionError('Original composite family/fold differs')
    if m['exact_plan_SHA']!=origin_plan_sha or m['exact_dispatch_plan_SHA']!=plan_sha or m['source_SHA']!=plan['source_SHA'] or m['split_SHA']!=plan['split_SHA'] or m['updates']!=updates:
        raise PermissionError('Original full provenance differs')
    if m['historical_task_weights_used'] or m['outer_labels_decoded']:raise PermissionError('Original scope violated')
    if recovery:
        if m.get('recovery',{}).get('restored_updates')!=recovery['restored_updates']:
            raise PermissionError('Loaded original recovery ancestry differs')
        for key,name in (('FIT_steps','original_FIT_steps.jsonl'),('cost_accounting','original_cost_accounting.json')):
            if digest(root/'out/recovery_origin'/name)!=plan['resume'][key]['SHA']:
                raise PermissionError('Original recovery replay-cost evidence differs')
    split=root/'source/split.json'
    if digest(split)!=plan['split_SHA']:raise PermissionError('Original split source differs')
    fold=json.loads(split.read_bytes())['folds'][m['fold']]
    if m['fit_ids']!=fold['row_ids']['fit'] or m['inner_ids']!=fold['row_ids']['inner']:raise ValueError('Exact role IDs differ')
    if digest(plan['parent_qualification']['path'])!=plan['parent_qualification']['SHA']:raise PermissionError('Parent qualification differs')
    parent=json.loads(Path(plan['parent_qualification']['path']).read_bytes());validate_parent(m['method'],m['fold'],parent,fold)
    if not parent.get('CPU_original_state_qualified') or parent['checkpoint']['SHA']!=r['parent_checkpoint_SHA']:
        raise PermissionError('Fresh trained parent original CPU identity differs')
    if tensor_sha(saved['parent_model'])!=parent['selected_state_SHA'] or parent['selected_state_SHA']!=r['parent_selected_state_SHA']:
        raise ValueError('Complete frozen parent state differs')
    for state in (saved['parent_model'],saved['model'],saved.get('selected_model') or {}):
        if any(t.is_floating_point() and not torch.isfinite(t).all() for t in state.values()):raise ValueError('Nonfinite full composite state')
    def combined(state):return {**{'parent.'+k:v for k,v in saved['parent_model'].items()},**{'addon.'+k:v for k,v in state.items()}}
    owners=[]
    if len(saved['optimizers'])!=(1 if m['method']=='anchored_message20' else 2):raise ValueError('Optimizer inventory differs')
    for opt,mapping in zip(saved['optimizers'],saved['optimizer_index_to_name']):
        ids=sum([g['params'] for g in opt['param_groups']],[])
        if len(ids)!=len(set(ids)) or set(map(str,ids))!=set(mapping):raise ValueError('Optimizer ownership differs')
        names=list(mapping.values());owners.extend(names)
        # In fixed-A, a head may receive a zero auxiliary update on an all-held
        # minibatch; this still creates a finite zero gradient/Adam state.
        if set(opt['state'])!=set(ids):raise ValueError('Missing addon Adam state')
        for i,entry in opt['state'].items():
            t=saved['model'][mapping[str(i)]]
            if set(entry)!={'step','exp_avg','exp_avg_sq'} or float(entry['step'])!=updates:raise ValueError('Original Adam steps differ')
            for key in ('exp_avg','exp_avg_sq'):
                if entry[key].shape!=t.shape or not torch.isfinite(entry[key]).all():raise ValueError('Invalid original Adam moments')
    if len(owners)!=len(set(owners)):raise ValueError('Two optimizers share parameters')
    expected={n for n in saved['model'] if (not n.endswith('.bias') or not n.startswith('heads.') or '.3.' not in n)} if m['method']=='anchored_message20' else {n for n in saved['model'] if n.startswith(('donor.','feedback.gradient_heads.'))}
    if set(owners)!=expected:raise ValueError('Original trainable parameter inventory differs')
    if set(saved['rng'])!={'python','numpy','torch','cuda'} or not saved['rng']['cuda']:raise ValueError('Complete Python/NumPy/Torch/CUDA RNG absent')
    if not np.array_equal(saved['orders'],np.load(root/'out/FIT_orders.npy',allow_pickle=False)):raise ValueError('Complete original orders differ')
    if m['method']=='anchored_message20':
        if tensor_sha(saved['frozen_tail'])!=r['frozen_tail_or_teacher_SHA']:raise ValueError('Frozen original tail differs')
    else:
        if tensor_sha({n:t for n,t in saved['model'].items() if n.startswith(('flow.','fusion.','predictor.'))})!=r['frozen_tail_or_teacher_SHA']:
            raise ValueError('Frozen original terminal teacher differs')
    for name,h in r['cache_SHA'].items():
        if digest(root/'out'/name)!=h:raise PermissionError('Original fresh cache bytes differ')
    if any(j['labels_read'] for j in saved['guard_journal'] if j['role']=='outer'):raise PermissionError('Training accessed outer targets')
    if m['method']=='anchored_message20' and any(j['labels_read'] for j in saved['guard_journal'] if j['role']=='inner'):
        raise PermissionError('Fixed20 message accessed INNER targets')
    if r['status']=='COMPOSITE_NATIVE_PRECHECK3_EXIT_PENDING_CPU_TRANSPORT':
        if updates!=3 or tensor_sha(combined(saved['model']))!=r['after_state_SHA']:raise ValueError('Original3-step state differs')
        with np.load(root/'out/INNER_dummy_predictions.npz',allow_pickle=False) as z:
            if z['row_ids'].tolist()!=fold['row_ids']['inner'] or not np.array_equal(z['dummy0'],z['dummy7']):raise ValueError('Original dummy arrays differ')
        if any(j['labels_read'] for j in saved['guard_journal'] if j['role']!='fit'):raise PermissionError('Precheck exposed nonFIT targets')
    elif r['status']=='COMPOSITE_COMPLETE_OUTER_UNSCORED_CPU_TRANSPORT_PENDING':
        epochs=20 if m['method']=='anchored_message20' else 100;per=(fold['rows']['fit']+31)//32;h=m['history']
        if len(h)!=epochs or updates!=epochs*per:raise ValueError('Original complete epoch/update count differs')
        best=float('inf');which=0
        for epoch,row in enumerate(h,1):
            if row['epoch']!=epoch or row['updates']!=epoch*per:raise ValueError('Original tail-inclusive history differs')
            if m['method']=='old_fixed_A':
                if not np.isfinite(row['inner_MSE']):raise ValueError('Nonfinite INNER selection objective')
                if row['inner_MSE']<best:best,which=row['inner_MSE'],epoch
                if row['best_epoch']!=which or row['best_MSE']!=best:raise ValueError('Strict earliest whole-INNER selection differs')
                frozen=root/'out'/f'INNER_epoch{epoch:03d}.npz'
                if digest(frozen)!=row['prediction_SHA']:raise PermissionError('Original frozen INNER prediction differs')
                with np.load(frozen,allow_pickle=False) as z:
                    if z['row_ids'].tolist()!=fold['row_ids']['inner'] or z['model_state_sha256'].item()!=row['state_SHA']:raise ValueError('Original INNER state/IDs differ')
        if tensor_sha(combined(saved['model']))!=h[-1]['state_SHA'] or tensor_sha(combined(saved['selected_model']))!=r['selected_state_SHA']:
            raise ValueError('Complete current/selected composite state differs')
        if m['method']=='old_fixed_A':
            if r['best_epoch']!=which or r['best_inner_MSE']!=best:raise ValueError('Original selected receipt differs')
            with np.load(root/'out'/f'INNER_epoch{which:03d}.npz',allow_pickle=False) as z:
                if not np.array_equal(z['prediction'],saved['selected_inner_prediction']):raise ValueError('Original selected INNER array differs')
        elif r['best_epoch']!=20 or saved['selected_inner_prediction'] is not None:raise ValueError('Fixed20 message used checkpoint selection')
        outer=root/'out/OUTER_prediction_only.npz'
        if digest(outer)!=r['outer_prediction_SHA']:raise PermissionError('Frozen original OUTER bytes differ')
        with np.load(outer,allow_pickle=False) as z:
            if z['row_ids'].tolist()!=fold['row_ids']['outer'] or z['model_state_sha256'].item()!=r['selected_state_SHA'] or z['prediction'].shape!=(fold['rows']['outer'],) or not np.isfinite(z['prediction']).all():
                raise ValueError('Original OUTER identity/shape/finiteness differs')
    else:raise PermissionError('Unknown composite status')
    return dict(status='INDEPENDENT_CPU_ORIGINAL_COMPOSITE_STATE_AUDIT_PASS',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                method=m['method'],fold=m['fold'],updates=updates,checkpoint_SHA=digest(path),bytes=path.stat().st_size,source_SHA=plan['source_SHA'],
                exact_plan_SHA=plan_sha,original_native_exit=0,CPU_available_RAM_bytes=available,model_tensors=len(saved['model']),
                parent_model_tensors=len(saved['parent_model']),Adam_owned_tensors=len(owners),new_fit_predict_or_score=False,argv=[sys.executable]+sys.argv)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();write(a.out,audit(a.root,a.plan_sha));print('INDEPENDENT_COMPOSITE_CPU_AUDIT_PASS')
