"""One pooled OOF score after all ten original predictions and states are saved.

No fitting or selection. Former official TEST is part of the pooled CV dataset.
"""
import argparse
import json
import pickle
import os
import sys
from pathlib import Path
import numpy as np
from fold_contract import sha,validate_folds
from fold_runtime import write,utc
from sentiment_metrics_careflow_v1 import metrics

FIVE=('Acc7','Acc2','F1','MAE','Corr')


def load_all_saved_predictions(plan,split,manifest,base):
    if manifest['status']!='GROUP5_ALL_TEN_D_ORIGINAL_CPU_CAPTURE_COMPLETE':
        raise PermissionError('All ten complete preservation records required')
    entries=manifest['entries']
    expected={(m,i) for m in plan['methods'] for i in range(5)}
    if len(entries)!=10 or {(e['method'],e['fold']) for e in entries}!=expected:
        raise ValueError('Missing or duplicate method/fold')
    predictions={};position={s:i for i,s in enumerate(split['canonical_row_ids'])}
    for e in entries:
        for kind in ['D_joint','CPU_receipt','original_receipt','prediction','complete_checkpoint']:
            ref=e[kind];p=base/ref['path']
            if sha(p)!=ref['sha256']:raise ValueError('Complete original bytes changed: '+kind)
        joint=json.loads((base/e['D_joint']['path']).read_text())
        cpu=json.loads((base/e['CPU_receipt']['path']).read_text())
        r=json.loads((base/e['original_receipt']['path']).read_text())
        if joint['status']!='GROUP5_D_ORIGINAL_CPU_CAPTURE_COMPLETE' or joint['method']!=e['method'] or joint['fold']!=e['fold']:
            raise PermissionError('Physical preservation linkage missing')
        if joint['CPU_receipt_sha256']!=e['CPU_receipt']['sha256'] or joint['original_receipt_sha256']!=e['original_receipt']['sha256']:
            raise ValueError('Preservation parents differ')
        if joint['whole_checkpoint_sha256']!=e['complete_checkpoint']['sha256'] or joint['prediction_sha256']!=e['prediction']['sha256']:
            raise ValueError('Preservation original state/prediction differs')
        if cpu['status']!='GROUP5_ORIGINAL_CPU_STATE_AUDIT_COMPLETE' or cpu['protocol_sha256']!=manifest['protocol_sha256']:
            raise ValueError('CPU audit differs')
        if r['status']!='GROUP5_FOLD100_COMPLETE_OUTER_UNSCORED_STORAGE_CPU_PENDING' or r['protocol_sha256']!=manifest['protocol_sha256']:
            raise ValueError('Original training not complete')
        with np.load(base/e['prediction']['path'],allow_pickle=False) as z:
            ids=z['row_ids'].tolist();p=z['prediction'].astype(np.float64)
            if ids!=split['folds'][e['fold']]['row_ids']['outer'] or p.shape!=(len(ids),) or not np.isfinite(p).all():
                raise ValueError('OUTER prediction inventory differs')
            if str(z['model_state_sha256'].item())!=r['selected_state_sha256']:
                raise ValueError('OOF selected model differs')
            predictions[e['method'],e['fold']]=(ids,p)
    pooled={}
    for m in plan['methods']:
        p=np.full(len(position),np.nan)
        for i in range(5):
            ids,v=predictions[m,i]
            for s,x in zip(ids,v):
                k=position[s]
                if np.isfinite(p[k]):raise ValueError('Duplicate outer prediction')
                p[k]=x
        if not np.isfinite(p).all():raise ValueError('Incomplete OOF coverage')
        pooled[m]=p
    return predictions,pooled


def score(args):
    plan=json.loads(args.protocol.read_text(encoding='utf-8'))
    if sha(args.protocol)!=args.protocol_sha or plan['status']!='GROUP5_EXECUTION_FROZEN':
        raise PermissionError('Exact frozen protocol required')
    if str(args.token)!=plan['global_score_token']:
        raise PermissionError('Exact global once-score token required')
    if args.root.exists():raise ValueError('Fresh score root required')
    split=json.loads((args.bundle/'split.json').read_text(encoding='utf-8'))
    if sha(args.bundle/'split.json')!=plan['split_sha256']:raise ValueError('Split differs')
    validate_folds(split['canonical_row_ids'],split['folds'])
    for rel,h in plan['source_sha256'].items():
        if sha(args.bundle/rel)!=h:raise ValueError('Scoring source differs')
    manifest=json.loads(args.manifest.read_text())
    if sha(args.manifest)!=args.manifest_sha or manifest['protocol_sha256']!=args.protocol_sha:
        raise ValueError('All-ten preservation manifest differs')
    predictions,pooled=load_all_saved_predictions(plan,split,manifest,args.manifest.parent)
    # The exclusive token is acquired before any original scalar is read.
    args.token.parent.mkdir(parents=True,exist_ok=True)
    with args.token.open('x',encoding='utf-8') as f:
        json.dump({'actual_utc':utc().isoformat(),'protocol_sha256':args.protocol_sha,
                   'manifest_sha256':args.manifest_sha,'pid':os.getpid()},f)
    args.root.mkdir()
    datafile=args.assets/'assets/mosi.pkl'
    if sha(datafile)!=plan['asset_sha256']['assets/mosi.pkl']:raise ValueError('Original target asset differs')
    with datafile.open('rb') as f:data=pickle.load(f)
    records=list(data['train'])+list(data['dev'])+list(data['test']);del data
    by_id={}
    for r in records:
        sid=r[2].decode() if isinstance(r[2],bytes) else r[2]
        if sid in by_id:raise ValueError('Duplicate original row')
        by_id[sid]=r
    ids=split['canonical_row_ids']
    if set(ids)!=set(by_id):raise ValueError('Original inventory differs')
    y=np.asarray([np.asarray(by_id[s][1]).reshape(-1)[0] for s in ids],dtype=np.float32).astype(np.float64)
    if not np.isfinite(y).all() or (np.abs(y)>3).any():raise ValueError('Original label scale differs')
    pos={s:i for i,s in enumerate(ids)};results={}
    for m in plan['methods']:
        folds=[]
        for i in range(5):
            sid,p=predictions[m,i];folds.append(metrics(p,y[[pos[s] for s in sid]]))
        results[m]={'pooled_OOF':metrics(pooled[m],y),'folds':folds,
                    'fold_mean_std':{k:{'mean':float(np.mean([f[k] for f in folds])),
                                        'sample_std':float(np.std([f[k] for f in folds],ddof=1))}
                                      for k in FIVE if all(f[k] is not None for f in folds)}}
    np.savez(args.root/'original_labels_and_pooled_OOF.npz',row_ids=np.asarray(ids),labels=y,**pooled)
    a=results['anchored_flow']['pooled_OOF'];c=results['careflow']['pooled_OOF']
    wins={k:(a[k]<c[k] if k=='MAE' else a[k]>c[k]) if a[k] is not None and c[k] is not None else False for k in FIVE}
    write(args.root/'actual_score.json',{'status':'GROUP5_ONCE_ALL_TEN_POOLED_OOF_FIVE_COMPLETE',
          'actual_utc':utc().isoformat(),'fullargv':[sys.executable]+sys.argv,'pid':os.getpid(),
          'protocol_sha256':args.protocol_sha,'manifest_sha256':args.manifest_sha,'results':results,
          'all_five_strictly_better':all(wins.values()),'five_wins':wins,
          'official_TEST_no_longer_holdout':True,'stable_five_seed_claimed':False,
          'same_checkpoint_all_five':True,'whole_process_video_isolation':True})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ['protocol','bundle','manifest','assets','root','token']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--protocol-sha',required=True);p.add_argument('--manifest-sha',required=True)
    score(p.parse_args())
