"""Read saved, fixed TRAIN/DEV evidence. No forward, TEST access or model choice."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def describe(y, p):
    e = p - y
    nz = y != 0
    return {
        'rows': len(y), 'label_mean': float(y.mean()),
        'prediction_mean': float(p.mean()), 'bias': float(e.mean()),
        'label_std': float(y.std()), 'prediction_std': float(p.std()),
        'MAE': float(np.abs(e).mean()), 'MSE': float((e*e).mean()),
        'nonzero_sign_errors': int(((p[nz] >= 0) != (y[nz] >= 0)).sum()),
        'over_abs3_predictions': int((np.abs(p) > 3).sum()),
    }


def run(root, out):
    if out.exists():
        raise FileExistsError(out)
    plan_path = root/'paired_fixed_selected_DEV_five_20261007T034922Z/plan.json'
    plan = read(plan_path)
    result = {'status': 'SAVED_FIXED_TRAIN_DEV_DIAGNOSIS_ONLY',
              'actual_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'inputs': {'DEV_plan': {'path': str(plan_path), 'sha256': sha(plan_path)}},
              'methods': {}, 'new_forward': False, 'TEST_read': False,
              'checkpoint_or_readout_chosen': False}
    common_y = None
    common_ids = None
    predictions = {}
    for method in ('minimal_fixed_F', 'careflow'):
        spec = plan['parents'][method]
        source = root/method/'a/run/out'
        for name, digest in spec['arrays_sha256'].items():
            if sha(source/name) != digest:
                raise ValueError('Original DEV array SHA mismatch: '+name)
        with np.load(source/'selected_best_DEV_replay.npz', allow_pickle=False) as z:
            ids = z['row_ids'].tolist()
            p = z['prediction'].astype(np.float64)
            if str(z['model_state_sha256'].item()) != spec['best_state_SHA']:
                raise ValueError('Selected state mismatch')
        y = np.load(source/'original_DEV_selection_targets.npy', allow_pickle=False)
        if common_y is None:
            common_y, common_ids = y.copy(), ids
        elif not np.array_equal(y, common_y) or ids != common_ids:
            raise ValueError('DEV order/targets differ')
        if len(ids) != 229 or p.shape != (229,) or not np.isfinite(p).all():
            raise ValueError('Original DEV schema mismatch')
        history_path = source/'history.json'
        history = read(history_path)
        if len(history) != 100 or history[-1]['best_epoch'] != spec['best_epoch']:
            raise ValueError('Fixed history mismatch')
        steps_path = source/'actual_TRAIN_steps.jsonl'
        steps = [json.loads(s) for s in steps_path.read_text().splitlines()]
        if len(steps) != 4000 or steps[-1]['optimizer_steps'] != 4000:
            raise ValueError('Original update count differs')
        # Inspect predeclared checkpoints, not a new DEV-based selection.
        trace = [{k: h[k] for k in ('epoch', 'optimizer_steps', 'TRAIN_mean_objective',
                                   'DEV_author_batch_MSE_selection_only')}
                 for h in history if h['epoch'] in (1,10,20,30,40,50,60,70,80,90,100)]
        groups = {}
        for label, mask in [('negative', y < 0), ('zero', y == 0), ('positive', y > 0),
                            ('abs_le1', np.abs(y) <= 1), ('abs_gt1', np.abs(y) > 1)]:
            groups[label] = describe(y[mask], p[mask])
        videos = sorted({s.rsplit('[',1)[0] for s in ids})
        video_rows = {}
        for video in videos:
            mask = np.array([s.rsplit('[',1)[0] == video for s in ids])
            video_rows[video] = describe(y[mask], p[mask])
        result['inputs'][method] = {
            'selected_state_sha256': spec['best_state_SHA'],
            'history_sha256': sha(history_path), 'steps_sha256': sha(steps_path),
            'selected_prediction_sha256': spec['arrays_sha256']['selected_best_DEV_replay.npz'],
            'labels_sha256': spec['arrays_sha256']['original_DEV_selection_targets.npy']}
        result['methods'][method] = {
            'fixed_best_epoch': spec['best_epoch'], 'all_rows': describe(y,p),
            'label_groups': groups, 'videos': video_rows, 'curve': trace,
            'last10_mean_TRAIN_objective': float(np.mean([h['TRAIN_mean_objective'] for h in history[-10:]])),
            'last10_mean_DEV_selection_MSE': float(np.mean([h['DEV_author_batch_MSE_selection_only'] for h in history[-10:]])),
            'seconds_per_step': {'median': float(np.median([s['seconds'] for s in steps])),
                                 'p95': float(np.quantile([s['seconds'] for s in steps],.95))},
            'selection_not_changed': True}
        predictions[method] = p
    f,c = (predictions[k] for k in ('minimal_fixed_F','careflow'))
    df = np.abs(f-common_y)-np.abs(c-common_y)
    video_delta = []
    for video in sorted({s.rsplit('[',1)[0] for s in common_ids}):
        mask = np.array([s.rsplit('[',1)[0] == video for s in common_ids])
        video_delta.append({'video':video,'rows':int(mask.sum()),'F_minus_C_MAE':float(df[mask].mean())})
    result['paired_DEV'] = {'F_lower_absolute_error_rows':int((df<0).sum()),
                            'F_higher_absolute_error_rows':int((df>0).sum()),
                            'ties':int((df==0).sum()), 'video_MAE_differences':video_delta}
    result['limits'] = [
        'TRAIN objective includes different auxiliary terms; its gap is not a pure matched TRAIN MSE gap.',
        'Saved prediction diagnostics do not identify the causal source of degradation.',
        'No statistical confirmation, TEST-based method selection, or calibrated replacement is performed.',
        'Source-code omission of direct source prediction supervision is a mechanism hypothesis only.']
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':result['status'],'actual_utc':result['actual_utc'],
                      'methods':{k:{'all_rows':v['all_rows'],'last10_TRAIN_objective':v['last10_mean_TRAIN_objective'],
                                    'last10_DEV_MSE':v['last10_mean_DEV_selection_MSE'],
                                    'seconds_per_step':v['seconds_per_step']} for k,v in result['methods'].items()},
                      'output_sha256':sha(out)},ensure_ascii=False))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    run(args.root,args.out)
