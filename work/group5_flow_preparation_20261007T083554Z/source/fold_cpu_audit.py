"""Independent native Torch CPU audit of one complete new fold state.

No model forward and no OUTER labels. Audits original bytes, state ownership,
Adam/scheduler/RNG/orders, frozen INNER selection and prediction identity.
"""
import argparse
import json
import os
import sys
import zipfile
from pathlib import Path
import numpy as np
from fold_contract import sha,batches,inner_mse
from fold_runtime import utc,write


def audit(args):
    plan=json.loads(args.protocol.read_text(encoding='utf-8'))
    if sha(args.protocol)!=args.protocol_sha or plan['status']!='GROUP5_EXECUTION_FROZEN':
        raise PermissionError('Exact execution protocol required')
    for rel,h in plan['source_sha256'].items():
        if sha(args.bundle/rel)!=h:raise ValueError('Source mismatch: '+rel)
    receipt=json.loads((args.original/'out/actual_stage_receipt.json').read_text())
    pre=receipt['status']=='GROUP5_PRECHECK3_COMPLETE_CPU_STORAGE_PENDING'
    if not pre and receipt['status']!='GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING':
        raise ValueError('No complete original stage')
    if receipt['protocol_sha256']!=args.protocol_sha:raise ValueError('Receipt protocol mismatch')
    split=json.loads((args.bundle/'split.json').read_text())
    fold=split['folds'][receipt['fold']]
    ref=receipt['complete_checkpoint'] if pre else receipt['complete_resume_and_selected']
    path=args.original/'out'/Path(ref['path']).name
    if sha(path)!=ref['sha256'] or path.stat().st_size!=ref['bytes']:
        raise ValueError('Whole original checkpoint bytes differ')
    with zipfile.ZipFile(path) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or z.testzip() is not None:raise ValueError('Checkpoint CRC/unique failed')
    import torch
    from fixed_flow_components_candidate import tensor_sha
    torch.set_num_threads(2)
    state=torch.load(path,map_location='cpu')
    meta=state['metadata'];updates=3 if pre else plan['fold_budgets'][receipt['fold']]['updates']
    if meta['optimizer_steps']!=updates or state['scheduler']['last_epoch']!=updates:
        raise ValueError('Adam/scheduler update count differs')
    if meta['source_sha256']!=plan['source_sha256'] or meta['split_sha256']!=plan['split_sha256']:
        raise ValueError('State source/split binding differs')
    for name,t in state['model'].items():
        if t.is_floating_point() and not torch.isfinite(t).all():raise ValueError('Nonfinite state: '+name)
    opt=state['optimizer'];mapping=state['optimizer_index_to_name']
    indices=sum([g['params'] for g in opt['param_groups']],[])
    if len(indices)!=len(set(indices)) or set(map(str,indices))!=set(mapping):
        raise ValueError('Optimizer ownership differs')
    allowed={'dberta.pooler.dense.weight','dberta.pooler.dense.bias'} if receipt['method']=='careflow' else set()
    absent={mapping[str(i)] for i in indices if i not in opt['state']}
    if absent!=allowed:raise ValueError('Unexpected absent Adam state: '+str(absent))
    for index,entry in opt['state'].items():
        param=state['model'][mapping[str(index)]]
        if set(entry)!={'step','exp_avg','exp_avg_sq'} or int(entry['step'])!=updates:
            raise ValueError('Adam state incomplete or step differs')
        for key in ['exp_avg','exp_avg_sq']:
            if entry[key].shape!=param.shape or not torch.isfinite(entry[key]).all():
                raise ValueError('Adam moment invalid')
    if set(state['rng'])!={'python','numpy','torch','cuda'} or not state['rng']['cuda']:
        raise ValueError('Complete RNG absent')
    orders=np.load(args.bundle/f"orders_fold{receipt['fold']}.npy",allow_pickle=False)
    if not np.array_equal(state['orders'],orders):raise ValueError('Shared orders differ')
    for order in orders:batches(order.tolist())
    if state['fit_ids']!=fold['row_ids']['fit'] or state['inner_ids']!=fold['row_ids']['inner']:
        raise ValueError('Physical roles differ')
    if receipt['method']=='anchored_flow':
        for role,fields in state['statistics'].items():
            for field,value in fields.items():
                if not torch.equal(state['model'][f'dberta.v6_{role}_{field}'],value):
                    raise ValueError('FIT normalization state differs')
    errors=[]
    if pre:
        with np.load(args.original/'out/precheck_dummy_predictions.npz',allow_pickle=False) as z:
            if not np.array_equal(z['dummy0'],z['dummy7']):raise ValueError('Dummy predictions differ')
            if str(z['model_state_sha256'].item())!=tensor_sha(state['model']):raise ValueError('Precheck state differs')
    else:
        labels=np.load(args.original/'out/original_INNER_selection_labels.npy',allow_pickle=False)
        best=float('inf');best_epoch=0
        history=meta['history']
        if len(history)!=100:raise ValueError('Incomplete history')
        frozen=[json.loads(s) for s in (args.original/'out/prediction_freeze.jsonl').read_text().splitlines()]
        if len(frozen)!=100:raise ValueError('Incomplete prediction freezing history')
        for epoch,h in enumerate(history,1):
            p=args.original/f'out/INNER_epoch{epoch:03d}.npz'
            if sha(p)!=h['inner_prediction_sha256'] or sha(p)!=frozen[epoch-1]['sha256']:
                raise ValueError('Original INNER bytes differ')
            with np.load(p,allow_pickle=False) as z:
                if z['row_ids'].tolist()!=fold['row_ids']['inner'] or str(z['model_state_sha256'].item())!=h['state_sha256']:
                    raise ValueError('INNER identity differs')
                score=inner_mse(z['prediction'],labels)
            errors.append(abs(score-h['INNER_selection_MSE']))
            if score<best:best,best_epoch=score,epoch
            if best_epoch!=h['best_epoch'] or best!=h['best_mse']:raise ValueError('Strict earliest selection differs')
        if best_epoch!=receipt['best_epoch'] or tensor_sha(state['selected_model'])!=receipt['selected_state_sha256']:
            raise ValueError('Unique selected model differs')
        with np.load(args.original/f'out/INNER_epoch{best_epoch:03d}.npz',allow_pickle=False) as z:
            if not np.array_equal(z['prediction'],state['selected_inner_prediction']):raise ValueError('Selected prediction differs')
        outer=args.original/'out/OUTER_prediction_only.npz'
        if sha(outer)!=receipt['outer_prediction_sha256']:raise ValueError('OUTER original bytes differ')
        with np.load(outer,allow_pickle=False) as z, np.load(args.original/'out/OUTER_dummy_invariance.npz',allow_pickle=False) as d:
            if set(z.files)!={'row_ids','prediction','model_state_sha256'} or z['row_ids'].tolist()!=fold['row_ids']['outer']:
                raise ValueError('OUTER role differs')
            if str(z['model_state_sha256'].item())!=receipt['selected_state_sha256']:
                raise ValueError('OUTER selected model differs')
            if not np.isfinite(z['prediction']).all() or not np.array_equal(z['prediction'],d['dummy0']) or not np.array_equal(d['dummy0'],d['dummy7']):
                raise ValueError('OUTER dummy invariance differs')
    result={'status':'GROUP5_ORIGINAL_CPU_STATE_AUDIT_COMPLETE','actual_utc':utc().isoformat(),
            'fullargv':[sys.executable]+sys.argv,'pid':os.getpid(),'method':receipt['method'],'fold':receipt['fold'],
            'protocol_sha256':args.protocol_sha,'original_receipt_sha256':sha(args.original/'out/actual_stage_receipt.json'),
            'whole_checkpoint_sha256':sha(path),'checkpoint_bytes':path.stat().st_size,'zip_crc_unique_members':len(names),
            'optimizer_states':len(opt['state']),'optimizer_steps':updates,'selection_max_error':max(errors,default=0.),
            'outer_labels_read':False,'CPU_model_forward':False}
    write(args.output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['protocol','bundle','original','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--protocol-sha',required=True);audit(p.parse_args())
