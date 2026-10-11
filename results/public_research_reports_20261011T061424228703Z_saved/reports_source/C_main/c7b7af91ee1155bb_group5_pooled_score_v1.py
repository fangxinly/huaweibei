"""One sealed scoring pass after all 25 original outer predictions are preserved.

No model imports, fitting, selection, prediction adjustment or failed-row removal.
Qualification verifies the original receipts and restoration, not progress notes.
"""
import argparse
import datetime as dt
import importlib.metadata
import json
import math
import pickle
import shutil
import sys
import zipfile
from pathlib import Path
from group5_release_transport_v1 import digest, write

METHODS = ('anchored_message20', 'old_fixed_A', 'factorized_aux', 'regression_aux', 'careflow')
METRICS = ('Acc7', 'Acc2', 'F1', 'MAE', 'Corr')


def exact_json(spec):
    p = Path(spec['path'])
    if digest(p) != spec['SHA']:
        raise PermissionError('Frozen evidence bytes differ: '+str(p))
    return json.loads(p.read_bytes())


def preflight(plan, entries, split):
    """Standard-library only: require every original stage/CPU/restore chain."""
    from fold_contract import validate_folds
    validate_folds(split['canonical_row_ids'], split['folds'])
    if split['rows'] != 2195 or split['videos'] != 93 or len(split['canonical_row_ids']) != 2195:
        raise PermissionError('Pooled human-authorized identity differs')
    keys = [(e['method'], e['fold']) for e in entries]
    wanted = {(m, f) for m in METHODS for f in range(5)}
    if len(keys) != 25 or len(set(keys)) != 25 or set(keys) != wanted:
        raise PermissionError('All 25 distinct preserved method/fold outputs are required')
    qualified = []
    for e in entries:
        r, cpu, restored, original_plan = [exact_json(e[k]) for k in ('receipt', 'CPU_audit', 'restoration', 'stage_plan')]
        method, fold = e['method'], e['fold']
        if (r['method'], r['fold']) != (method, fold) or (cpu['method'], cpu['fold']) != (method, fold):
            raise PermissionError('Wrong-family or cross-fold output')
        if r.get('historical_task_weights_used', True) or not r.get('public_pretraining_fresh_start') or r.get('outer_labels_decoded', True):
            raise PermissionError('Fresh per-fold training/held-out target scope violated')
        status = ('COMPOSITE_COMPLETE_OUTER_UNSCORED_CPU_TRANSPORT_PENDING' if method in METHODS[:2]
                  else 'DIRECT100_COMPLETE_OUTER_UNSCORED_CPU_TRANSPORT_PENDING')
        audit_status = ('INDEPENDENT_CPU_ORIGINAL_COMPOSITE_STATE_AUDIT_PASS' if method in METHODS[:2]
                        else 'INDEPENDENT_CPU_ORIGINAL_DIRECT_STATE_AUDIT_PASS')
        if r['status'] != status or cpu['status'] != audit_status or cpu['original_native_exit'] != 0:
            raise PermissionError('Incomplete or unqualified original stage')
        if cpu['checkpoint_SHA'] != r['checkpoint']['SHA'] or cpu['exact_plan_SHA'] != e['stage_plan']['SHA']:
            raise PermissionError('CPU audit is not bound to this original checkpoint/plan')
        if r.get('exact_dispatch_plan_SHA', r['exact_plan_SHA']) != e['stage_plan']['SHA']:
            raise PermissionError('Stage dispatch provenance differs')
        if original_plan['split_SHA'] != plan['split_SHA'] or cpu['source_SHA'] != original_plan['source_SHA']:
            raise PermissionError('Source/split evidence differs')
        spec = split['folds'][fold]
        if r['fit_ids'] != spec['row_ids']['fit'] or r['inner_ids'] != spec['row_ids']['inner']:
            raise PermissionError('Training or selection IDs differ')
        updates = (20 if method == 'anchored_message20' else 100) * math.ceil(spec['rows']['fit']/32)
        if r['updates'] != updates or cpu['updates'] != updates:
            raise PermissionError('Fixed tail-inclusive update budget differs')
        if (not restored.get('all_member_SHA_CRC_unique_exact_set_passed') or
                restored['whole_SHA'] != e['original_archive_SHA'] or
                restored.get('downloaded_bytes', restored.get('bytes')) != e['original_archive_bytes']):
            raise PermissionError('Whole original GitHub download restoration is absent')
        members = exact_json(e['original_member_manifest'])
        matches = [v for v in members if v['name'] == e['prediction_member']]
        if len(matches) != 1 or matches[0]['sha256'] != r['outer_prediction_SHA']:
            raise PermissionError('Prediction is not a member of the preserved original')
        p = Path(e['prediction_path'])
        if digest(p) != r['outer_prediction_SHA'] or p.stat().st_size != matches[0]['bytes']:
            raise PermissionError('Original unscored prediction bytes differ')
        if any(j.get('labels_read') for j in r['guard_journal'] if j['role'] == 'outer'):
            raise PermissionError('An outer target was read during training')
        if method in METHODS[:2]:
            parent = exact_json(e['parent_qualification'])
            from group5_test_selected_contract_v1 import validate_parent
            validate_parent(method, fold, parent, spec)
            if not parent.get('CPU_original_state_qualified') or r['parent_checkpoint_SHA'] != parent['checkpoint']['SHA']:
                raise PermissionError('Fresh teacher/parent CPU/checkpoint identity differs')
        qualified.append((e, r))
    return qualified


def frozen_predictions(qualified, split):
    import numpy as np
    predictions = {}
    for e, r in qualified:
        with np.load(e['prediction_path'], allow_pickle=False) as z:
            if set(z.files) != {'row_ids', 'prediction', 'model_state_sha256'}:
                raise ValueError('Exact original prediction schema differs')
            ids = z['row_ids'].tolist(); p = z['prediction'].astype(np.float64)
            expected = split['folds'][e['fold']]['row_ids']['outer']
            if ids != expected or z['model_state_sha256'].item() != r['selected_state_SHA']:
                raise PermissionError('Original prediction row/selected-state identity differs')
            if p.shape != (len(expected),) or not np.isfinite(p).all():
                raise ValueError('Nonfinite/missing outer predictions; no row removal')
            predictions[(e['method'], e['fold'])] = (ids, p.copy())
    return predictions


def aggregate(predictions, targets, split):
    import numpy as np
    from sentiment_metrics_careflow_v1 import metrics, SEMANTICS
    canonical = split['canonical_row_ids']; position = {s: i for i, s in enumerate(canonical)}
    result = {}
    for method in METHODS:
        oof = np.full(len(canonical), np.nan, dtype=np.float64); folds = []
        for fold in range(5):
            ids, p = predictions[(method, fold)]
            ix = np.asarray([position[s] for s in ids], dtype=np.int64)
            if np.isfinite(oof[ix]).any():
                raise ValueError('Duplicate outer prediction assignment')
            oof[ix] = p; folds.append(dict(fold=fold, metrics=metrics(p, targets[ix])))
        if not np.isfinite(oof).all():
            raise ValueError('Incomplete pooled OOF coverage')
        summary = {}
        for name in METRICS:
            values = [f['metrics'][name] for f in folds]
            # Undefined correlations are kept explicit, never silently dropped.
            summary[name] = dict(values=values, mean=None, sample_SD=None, defined_folds=sum(v is not None for v in values))
            if all(v is not None for v in values):
                summary[name].update(mean=float(np.mean(values)), sample_SD=float(np.std(values, ddof=1)))
        result[method] = dict(folds=folds, unweighted_fold_summary=summary,
                              concatenated_OOF=metrics(oof, targets), all_rows_once=True)
    return dict(methods=result, metric_semantics=SEMANTICS,
                interpretation='Historical TEST-selected exploratory repartition CV, not a new blind test. Fold SD is descriptive; overlapping training folds are not independent repetitions.',
                candidate_selection_exposure='Historical anchored-message TEST includes FIT/INNER overlap; historical TEST guided the fixed candidate roster.',
                upstream_exposure='Original pickle generation, upstream processing/units and pretrained exposure are not completely identified.',
                prohibited_claims=['MI/PID', 'causal mechanism', 'independent five-run replication', 'new blind-test evidence'])


def run(a):
    if digest(a.plan) != a.plan_sha:
        raise PermissionError('Exact separately frozen score plan differs')
    plan = json.loads(a.plan.read_bytes())
    if plan.get('status') != 'GROUP5_ALL25_ONCE_SCORE_FROZEN' or not plan.get('execution_enabled'):
        raise PermissionError('Scoring preparation is not execution authorization')
    if a.out.exists():
        raise FileExistsError('A fresh score output directory is required')
    for name, spec in plan['code_SHA'].items():
        if digest(a.source/name) != spec:
            raise PermissionError('Frozen scoring source differs: '+name)
    for name, version in plan['runtime_versions'].items():
        if importlib.metadata.version(name) != version:
            raise PermissionError('Frozen score runtime differs')
    split = exact_json(plan['split']); entries = exact_json(plan['all25_manifest'])
    if plan['split']['SHA'] != plan['split_SHA'] or digest(plan['original_pickle']['path']) != plan['original_pickle']['SHA']:
        raise PermissionError('Exact pooled source identity differs')
    qualified = preflight(plan, entries, split)
    predictions = frozen_predictions(qualified, split)
    if sys.executable != plan['python']:
        raise PermissionError('Frozen scoring interpreter differs')
    if Path('/proc/meminfo').exists():
        available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))*1024
    elif sys.platform == 'win32':
        import ctypes
        class MemoryStatus(ctypes.Structure):
            _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [(n,ctypes.c_ulonglong) for n in ('totalPhys','availPhys','totalPageFile','availPageFile','totalVirtual','availVirtual','availExtendedVirtual')]
        status=MemoryStatus();status.length=ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Cannot observe actual available CPU RAM')
        available=status.availPhys
    else:
        raise PermissionError('Actual CPU available-RAM observation is required')
    if available < 6*1024**3 or shutil.disk_usage(a.out.parent).free < plan['required_output_space_bytes']:
        raise PermissionError('Actual CPU RAM/output space gate failed')
    # Every preserved original prediction is validated before the first target index.
    once = Path(plan['new_once_token']); once.parent.mkdir(parents=True, exist_ok=True)
    with once.open('x', encoding='utf8') as f:
        json.dump(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(), plan_SHA=a.plan_sha,
                       all25_preflight_passed=True, action='one combined five-method scoring pass'), f)
    a.out.mkdir()
    import numpy as np
    with Path(plan['original_pickle']['path']).open('rb') as f:
        data = pickle.load(f)
    records = list(data['train'])+list(data['dev'])+list(data['test']); del data
    by_id = {}
    for row in records:
        s = row[2].decode() if isinstance(row[2], bytes) else row[2]
        if s in by_id:
            raise ValueError('Duplicate physical source ID')
        by_id[s] = row
    if set(by_id) != set(split['canonical_row_ids']):
        raise ValueError('Original pooled physical row inventory differs')
    targets = []
    for s in split['canonical_row_ids']:
        value = np.asarray(by_id[s][1])
        if value.size != 1:
            raise ValueError('Original target must be exactly scalar')
        targets.append(value.reshape(-1)[0])
    targets = np.asarray(targets, dtype=np.float64)
    if targets.shape != (2195,) or not np.isfinite(targets).all() or (np.abs(targets)>3).any():
        raise ValueError('Invalid pooled targets; no clipping/removal')
    report = aggregate(predictions, targets, split)
    report.update(status='GROUP5_ALL25_ONCE_ORIGINAL_PREDICTION_SCORE_COMPLETE',
                  actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(), exact_plan_SHA=a.plan_sha,
                  original_pickle_SHA=plan['original_pickle']['SHA'], split_SHA=plan['split_SHA'],
                  arrays=25, target_rows_decoded=2195, models_loaded=0, new_fit_or_prediction=0,
                  target_scope='Pooled original TRAIN/DEV/TEST targets decoded only after all25 preservation preflight. Trusted pickle materialization may contain all original bytes.',
                  argv=[sys.executable]+sys.argv)
    write(a.out/'all25_original_scores.json', report)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for n in ('plan', 'source', 'out'): p.add_argument('--'+n, type=Path, required=True)
    p.add_argument('--plan-sha', required=True); args=p.parse_args()
    try:
        run(args)
    except BaseException as exc:
        if args.out.exists(): write(args.out/'failed_natural_exit.json', dict(exit=1, actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(), error_type=type(exc).__name__, message=str(exc)))
        raise
    else:
        write(args.out/'natural_exit.json', dict(exit=0, actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat()))
