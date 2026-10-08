"""Independent NumPy audit of original TRAIN oracle arrays and receipt."""
from pathlib import Path
import argparse, datetime, hashlib, json
import numpy as np

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def rank(values):
    order = np.argsort(values, kind='stable'); out = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]; start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]: end += 1
        out[order[start:end]] = .5 * (start + end - 1); start = end
    return out

def correlation(x, y):
    return None if np.std(x) == 0 or np.std(y) == 0 else float(np.corrcoef(x, y)[0, 1])

def main():
    p = argparse.ArgumentParser(); p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--directory', type=Path, required=True); p.add_argument('--report', type=Path, required=True)
    a = p.parse_args(); plan = json.loads(a.plan.read_text(encoding='utf-8'))
    receipt = json.loads((a.directory/'receipt.json').read_text(encoding='utf-8'))
    assert receipt['status'] == 'TRAIN_ORACLE_DIAGNOSTIC_COMPLETE_NOT_TRAINING'
    assert receipt['plan_sha256'] == sha(a.plan)
    assert receipt['source_sha256'] == plan['diagnostic_source_sha256']
    assert receipt['diagnostics_sha256'] == sha(a.directory/'diagnostics.npz')
    assert receipt['pinned_files'] == plan['pinned_files']
    assert not any(receipt[k] for k in ['dev_requested','test_requested','optimizer_created','fitted'])
    assert receipt['parameter_sha_before'] == receipt['parameter_sha_after'] and receipt['parameter_gradients_none']
    assert receipt['initial_compute'] == '' and receipt['gpu'].split(',')[0] == plan['expected_uuid']
    assert receipt['cached_learned_replay_error'] <= 2e-6 and receipt['label_replacement_error'] == 0
    assert receipt['p0_replay_error'] < 2e-5 and receipt['peak_allocated_bytes'] <= plan['maximum_peak_allocated_bytes']
    assert receipt['elapsed_seconds'] <= plan['maximum_seconds'] + 120
    z = np.load(a.directory/'diagnostics.npz', allow_pickle=False)
    assert np.array_equal(z['row'], np.arange(1281)) and len(np.unique(z['video'])) == 52
    assert np.array_equal(z['mismatched_label_row'], np.roll(np.arange(1281), 1))
    for key in z.files:
        if np.issubdtype(z[key].dtype, np.number): assert np.isfinite(z[key]).all(), key
    y = z['y'].astype(np.float64); p0 = z['p0'].astype(np.float64); raw = z['raw_prediction'].astype(np.float64)
    beta = np.float64(plan['controller']['beta']); rms2 = np.float64(plan['controller']['scales']['residual_rms'])**2
    # Runtime buffers are float32; objective reconstruction tolerates their rounding.
    beta32 = np.float64(np.float32(beta)); rms32 = np.float64(np.float32(np.sqrt(rms2)))**2
    results = {}; groups = np.unique(z['video'])
    def metrics(pred):
        error = pred - y; gain = error**2 - (raw-y)**2
        return dict(mae=float(np.abs(error).mean()), mse=float((error**2).mean()),
            video_equal_mse=float(np.mean([(error[z['video']==g]**2).mean() for g in groups])),
            risk_change_vs_raw=float(gain.mean()), improvement_fraction=float((gain < 0).mean()))
    results['raw_fixed'] = metrics(raw)
    for mode in ['learned','oracle','zero','mismatched']:
        pred = z[mode+'_prediction_path'].astype(np.float64); prox = z[mode+'_prox_path'].astype(np.float64)
        rho = z[mode+'_residual'].astype(np.float64)
        assert pred.shape == prox.shape == (1281,4)
        assert np.max(np.abs(pred[:,0]-raw)) <= 2e-6
        relative = z[mode+'_relative_path']
        assert relative.shape == (1281,4) and np.max(relative) <= .25001
        assert np.all(prox >= 0) and np.all(prox[:,0] == 0)
        assert z[mode+'_message_distance_path'].shape == (1281,4,6)
        assert z[mode+'_message_cosine_path'].shape == (1281,4,6)
        delta = pred-p0[:,None]
        reconstructed = prox+beta32*(2*rho[:,None]*delta+delta**2)/rms32
        objective = z[mode+'_objective_path'].astype(np.float64)
        assert np.max(np.abs(objective-reconstructed)/(1+np.abs(objective)+np.abs(reconstructed))) < 5e-5
        results[mode+'_last'] = metrics(pred[:,-1])
        proxy = 2*rho*(pred[:,-1]-raw)+(pred[:,-1]-p0)**2-(raw-p0)**2
        observed = (pred[:,-1]-y)**2-(raw-y)**2
        results[mode+'_last'].update(proxy_change_mean=float(proxy.mean()),
            proxy_observed_pearson=correlation(proxy,observed),
            proxy_observed_spearman=correlation(rank(proxy),rank(observed)),
            boundary_fraction=float((relative[:,-1] >= .25-1e-5).mean()))
    assert np.max(np.abs(z['learned_prediction_path'][:,-1]-z['original_prediction'])) <= 2e-6
    assert np.max(np.abs(z['oracle_residual']-(z['p0']-z['y']))) == 0
    assert np.all(z['zero_residual'] == 0)
    idx = z['oracle_best_true_step']; assert idx.shape == (1281,) and np.all((idx>=0)&(idx<=3))
    pred = z['oracle_prediction_path'].astype(np.float64); prox = z['oracle_prox_path'].astype(np.float64)
    objectives = prox+beta32*(pred-y[:,None])**2/rms32
    selected = objectives[np.arange(1281),idx]
    assert np.max(selected-objectives.min(1)) <= 1e-4
    best = z['oracle_best_true_prediction'].astype(np.float64)
    assert np.max(np.abs(best-pred[np.arange(1281),idx])) <= 2e-6
    assert np.max((best-y)**2-(raw-y)**2) <= 2e-5
    results['oracle_best_including_F'] = metrics(best)
    truth = p0-y; estimate = z['estimated_residual'].astype(np.float64)
    residual = dict(mae=float(np.abs(estimate-truth).mean()),mse=float(((estimate-truth)**2).mean()),
                    observed_residual_sign_agreement=float((np.sign(estimate)==np.sign(truth)).mean()),
                    target='observed TRAIN residual, not true conditional residual')
    report = dict(status='ORIGINAL_TRAIN_ORACLE_ARRAYS_INDEPENDENTLY_AUDITED',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),receipt=receipt,
        results=results,residual=residual,scope=plan['scope'],
        interpretation='Fixed-weight label-known finite-step TRAIN feasibility; no global optimum or generalization claim.')
    assert not a.report.exists(); a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(results,ensure_ascii=True))

if __name__ == '__main__': main()
