"""Separate CPU process audits exact original direct-stage state; no new fit/score."""
import argparse,datetime as dt,json,os,sys,zipfile
from pathlib import Path
from group5_release_transport_v1 import digest,write


def audit(root,plan_sha):
    plan=json.loads((root/'plan.json').read_bytes())
    if digest(root/'plan.json')!=plan_sha:raise PermissionError('Exact original plan differs')
    exitrec=json.loads((root/'wrapper_exit.json').read_bytes())
    if exitrec['natural_exit']!=0:raise PermissionError('Original native stage did not exit0')
    r=json.loads((root/'out/actual_stage_receipt.json').read_bytes());checkpoint=root/'out'/Path(r['checkpoint']['path']).name
    if digest(checkpoint)!=r['checkpoint']['SHA'] or checkpoint.stat().st_size!=r['checkpoint']['bytes']:
        raise ValueError('Original checkpoint bytes differ')
    meminfo=Path('/proc/meminfo').read_text();available=int(next(l.split()[1] for l in meminfo.splitlines() if l.startswith('MemAvailable:')))*1024
    if available<6*1024**3:raise PermissionError('CPU available RAM below6GiB')
    with zipfile.ZipFile(checkpoint) as z:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip() is not None:raise ValueError('Checkpoint CRC/unique differs')
    import numpy as np
    import torch
    from fixed_flow_components_candidate import tensor_sha
    torch.set_num_threads(2);saved=torch.load(checkpoint,map_location='cpu')
    updates=r['updates'];meta=saved['metadata']
    if meta['updates']!=updates or saved['scheduler']['last_epoch']!=updates:raise ValueError('Update/scheduler count differs')
    if meta['exact_plan_SHA']!=plan_sha or meta['source_SHA']!=plan['source_SHA'] or meta['split_SHA']!=plan['split_SHA']:
        raise PermissionError('Scientific provenance differs')
    if meta['historical_task_weights_used'] or meta['outer_labels_decoded']:raise PermissionError('Scope violated')
    if r['status']=='NATIVE_PRECHECK3_EXIT_PENDING_CPU_TRANSPORT' and updates!=3:raise ValueError('Precheck update count differs')
    for n,t in saved['model'].items():
        if t.is_floating_point() and not torch.isfinite(t).all():raise ValueError('Nonfinite model: '+n)
    opt=saved['optimizer'];mapping=saved['optimizer_index_to_name'];indices=sum([g['params'] for g in opt['param_groups']],[])
    if len(indices)!=len(set(indices)) or set(map(str,indices))!=set(mapping):raise ValueError('Optimizer owner inventory differs')
    allowed={'dberta.pooler.dense.weight','dberta.pooler.dense.bias'} if meta['method'] in ('careflow','old_A_teacher') else set()
    absent={mapping[str(i)] for i in indices if i not in opt['state']}
    if absent!=allowed:raise ValueError('Unexpected absent optimizer state')
    for i,entry in opt['state'].items():
        t=saved['model'][mapping[str(i)]]
        if set(entry)!={'step','exp_avg','exp_avg_sq'} or int(entry['step'])!=updates:raise ValueError('Adam ownership/steps differ')
        for key in ('exp_avg','exp_avg_sq'):
            if entry[key].shape!=t.shape or not torch.isfinite(entry[key]).all():raise ValueError('Adam moments incomplete')
    fold=json.loads((root/'source'/meta['method']/'split.json').read_bytes())['folds'][meta['fold']]
    if meta['fit_ids']!=fold['row_ids']['fit'] or meta['inner_ids']!=fold['row_ids']['inner']:
        raise ValueError('Exact row identity differs')
    if not np.array_equal(saved['orders'],np.load(root/'out/FIT_orders.npy',allow_pickle=False)):
        raise ValueError('Original full orders differ')
    if set(saved['rng'])!={'python','numpy','torch','cuda'} or not saved['rng']['cuda']:
        raise ValueError('Complete RNG absent')
    if r['status']=='NATIVE_PRECHECK3_EXIT_PENDING_CPU_TRANSPORT':
        if r['after_state_SHA']!=tensor_sha(saved['model']):raise ValueError('Precheck saved model state differs')
        with np.load(root/'out/INNER_dummy_predictions.npz',allow_pickle=False) as z:
            if not np.array_equal(z['dummy0'],z['dummy7']) or z['row_ids'].tolist()!=fold['row_ids']['inner']:
                raise ValueError('Original dummy-label array equality differs')
        if any(j['labels_read'] for j in saved['guard_journal'] if j['role']!='fit'):
            raise PermissionError('Precheck accessed nonFIT labels')
    return dict(status='INDEPENDENT_CPU_ORIGINAL_DIRECT_STATE_AUDIT_PASS',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        checkpoint_SHA=digest(checkpoint),bytes=checkpoint.stat().st_size,method=meta['method'],fold=meta['fold'],updates=updates,
        model_tensors=len(saved['model']),Adam_owned_tensors=len(opt['state']),absent_unused_tensors=sorted(absent),
        source_SHA=plan['source_SHA'],exact_plan_SHA=plan_sha,original_native_exit=0,CPU_available_RAM_bytes=available,
        new_solve_fit_score_or_outer_label_decode=False,argv=[sys.executable]+sys.argv)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();write(a.out,audit(a.root,a.plan_sha));print('INDEPENDENT_CPU_AUDIT_PASS')
