"""Post-hoc descriptive audit of already frozen DEV arrays; no fit or selection."""
from pathlib import Path
import argparse, datetime, hashlib, io, json, shutil, zipfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('D:/CodexBackups/selective_flow_20261003_1105')
SHOTS = BASE / 'finite_c2_completed_snapshots_20261005T1627Z'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def ranks(x):
    order = np.argsort(x, kind='stable')
    values = x[order]
    edges = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1, len(x)]
    out = np.empty(len(x), dtype=np.float64)
    for lo, hi in zip(edges[:-1], edges[1:]):
        out[order[lo:hi]] = (lo + hi - 1) / 2
    return out

def correlation(x, y):
    xc, yc = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(xc) * np.linalg.norm(yc)
    return None if denom == 0 else float(np.dot(xc, yc) / denom)

def strata(score, gain, predicted_gain):
    order = np.argsort(-score, kind='stable')
    bins, coverage = [], []
    for i, ids in enumerate(np.array_split(order, 10)):
        bins.append(dict(bin_highest_first=i+1, count=len(ids),
                         score_mean=float(score[ids].mean()),
                         predicted_gain_mean=float(predicted_gain[ids].mean()),
                         actual_gain_mean=float(gain[ids].mean()),
                         actual_improved_fraction=float(np.mean(gain[ids] > 0))))
    for pct in range(10, 101, 10):
        ids = order[:int(np.ceil(len(order)*pct/100))]
        coverage.append(dict(requested_percent=pct, selected_count=len(ids),
                             actual_coverage=len(ids)/len(order),
                             selected_mean_gain=float(gain[ids].mean()),
                             all_rows_gain_with_unselected_fallback=float(gain[ids].sum()/len(order)),
                             selected_improved_fraction=float(np.mean(gain[ids] > 0))))
    return dict(bins=bins, fixed_coverage=coverage,
                scope='All prespecified bins/coverage reported. No best threshold, tuning, or confidence guarantee.')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stamp', required=True)
    args = parser.parse_args()
    out = ROOT / 'work' / ('archived_dev_residual_signal_' + args.stamp)
    out.mkdir(exist_ok=False)
    scope = ('Post-hoc descriptive reuse of the original DEV229 frozen diagnostics. '
             'All checkpoints were selected on DEV. Not an isolated validation, '
             'new confirmation, seed replication, whole-pipeline crossfit, GPU inference, '
             'other-node CPU, or confidence/risk guarantee. No TEST or new labels decoded. '
             'No model fit, shrink coefficient, cutoff, candidate, or checkpoint selected. '
             'Rows are pooled, not assumed independent. Video bootstrap is not performed '
             'without an independently verified DEV row-to-video mapping.')
    plan = dict(schema_version=1, frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_sha256=sha(Path(__file__).read_bytes()), scope=scope,
                arms=['a','b','c'], signed_residual='rho_true=p0-y; correction delta=-rho_hat',
                direct_coefficient=1.0, variance_ddof=0,
                diagnostics=['Pearson and tie-aware Spearman', 'sign and majority/independence baselines',
                             'mean/variance/second moment and observed direct correction gain',
                             'all 10 fixed equal-count bins', 'all fixed 10%-100% coverage points'],
                direct_order_score='rho_hat squared',
                controller_order_score='negative original proxy control-vs-raw loss change',
                tie_rule='Original row order, stable sort', bootstrap=False,
                input_snapshot_directory=str(SHOTS))
    write_json(out/'frozen_plan.json', plan)
    plan_sha = sha((out/'frozen_plan.json').read_bytes())
    rows, provenance, derivative = {}, {}, {}
    expected_y = None
    for arm in plan['arms']:
        with zipfile.ZipFile(SHOTS/arm/'snapshot.zip') as z:
            pre = 'finite_c2/diagnostics_v4/' if arm == 'c' else 'finite_diagnostics_v3/'
            raw = z.read(pre+'diagnostics.npz')
            receipt_raw = z.read(pre+'diagnostics_receipt.json')
            receipt = json.loads(receipt_raw)
            assert sha(raw) == receipt['diagnostics_sha256']
            assert receipt['tensor_sha_before'] == receipt['tensor_sha_after']
            assert receipt['rows'] == 229 and not receipt['test_requested']
            with np.load(io.BytesIO(raw), allow_pickle=False) as d:
                y, p0, h, fixed, control = [np.asarray(d[k], dtype=np.float64).copy() for k in
                    ['valid_y','reference_prediction','estimated_residual','raw_fixed_prediction','prediction_default']]
            for a in [y,p0,h,fixed,control]:
                assert a.shape == (229,) and np.isfinite(a).all()
            if expected_y is None:
                expected_y = y.copy()
            else:
                assert np.array_equal(y, expected_y)
            (out/('original_'+arm+'_diagnostics_receipt.json')).write_bytes(receipt_raw)
            r = p0-y
            corrected = p0-h
            loss0, loss_direct = r*r, (corrected-y)**2
            gain = loss0-loss_direct
            identity = 2*h*r-h*h
            err = float(np.max(np.abs(gain-identity)))
            assert err < 1e-12
            pos, pred_pos = r>0, h>0
            p, q = float(pos.mean()), float(pred_pos.mean())
            recalls = [float(np.mean(pred_pos[pos])), float(np.mean(~pred_pos[~pos]))]
            delta = control-fixed
            proxy = 2*h*(control-fixed)+(control-p0)**2-(fixed-p0)**2
            controller_gain = (fixed-y)**2-(control-y)**2
            proxy_gain = -proxy
            old_gain_err = float(np.max(np.abs((-proxy-controller_gain)+2*(h-r)*delta)))
            assert old_gain_err < 1e-12
            independent_pearson = float(np.corrcoef(h,r)[0,1]) if h.std()>0 and r.std()>0 else None
            pearson = correlation(h,r)
            if pearson is not None:
                assert abs(pearson-independent_pearson) < 1e-12
            independent_direct = float(2*np.dot(h,r)/len(y)-np.dot(h,h)/len(y))
            assert abs(float(gain.mean())-independent_direct) < 1e-12
            row = dict(n=229, pearson_h_observed_residual=pearson,
                       spearman_h_observed_residual=correlation(ranks(h),ranks(r)),
                       sign_accuracy=float(np.mean(pred_pos==pos)),
                       balanced_sign_accuracy=float(np.mean(recalls)),
                       true_positive_fraction=p, predicted_positive_fraction=q,
                       majority_constant_sign_accuracy=max(p,1-p),
                       independence_sign_accuracy_matching_marginals=p*q+(1-p)*(1-q),
                       residual_mean=float(r.mean()), residual_variance=float(r.var()),
                       estimate_mean=float(h.mean()), estimate_variance=float(h.var()),
                       estimate_second_moment=float(np.mean(h*h)),
                       residual_estimate_cross_moment=float(np.mean(h*r)),
                       reference_mse=float(loss0.mean()), direct_correction_mse=float(loss_direct.mean()),
                       direct_correction_mean_gain=float(gain.mean()),
                       direct_relative_mse_improvement=float(gain.mean()/loss0.mean()),
                       direct_improved_fraction=float(np.mean(gain>0)),
                       direct_identity_max_error=err,
                       original_control_vs_own_raw_mean_gain=float(controller_gain.mean()),
                       original_control_vs_own_raw_improved_fraction=float(np.mean(controller_gain>0)),
                       original_controller_identity_max_error=old_gain_err,
                       original_controller_predicted_actual_gain_pearson=correlation(proxy_gain,controller_gain),
                       direct_correction_fixed_strata=strata(h*h,gain,h*h),
                       original_controller_fixed_strata=strata(proxy_gain,controller_gain,proxy_gain))
            rows[arm] = row
            provenance[arm] = dict(snapshot=str(SHOTS/arm/'snapshot.zip'), member=pre+'diagnostics.npz',
                                   diagnostics_sha256=sha(raw), receipt_sha256=sha(receipt_raw),
                                   checkpoint_tensor_sha256=receipt['tensor_sha_before'])
            for name, a in [('p0',p0),('rho_hat',h),('observed_residual',r),('y',y),
                            ('direct_prediction',corrected),('direct_gain',gain),
                            ('controller_gain',controller_gain),('controller_proxy_gain',proxy_gain)]:
                derivative[arm+'_'+name] = a
    np.savez(out/'derived_arrays.npz', **derivative)
    report = dict(status='ACTUAL_LOCAL_POSTHOC_ARCHIVED_DEV_RESIDUAL_AUDIT_COMPLETE',
                  actual_completion_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  frozen_plan_sha256=plan_sha, source_sha256=plan['source_sha256'],
                  derived_arrays_sha256=sha((out/'derived_arrays.npz').read_bytes()),
                  input_provenance=provenance, rows=rows, scope=scope,
                  empirical_h_squared_is_true_ceiling=False,
                  independent_local_checks=['Centered dot product vs numpy.corrcoef',
                                            'Direct saved prediction MSE vs dot product gain identity',
                                            'Original proxy/actual identity'],
                  actual_GPU=False, other_node_CPU=False, new_model_fit=False)
    write_json(out/'actual_result.json', report)
    shutil.copyfile(__file__, out/Path(__file__).name)
    permanent = BASE / ('archived_dev_residual_signal_actual_' + args.stamp)
    assert shutil.disk_usage(BASE).free >= 1024**3
    permanent.mkdir(exist_ok=False)
    hashes = {}
    for f in sorted(out.iterdir()):
        shutil.copyfile(f, permanent/f.name)
        hashes[f.name] = dict(sha256=sha(f.read_bytes()), bytes=f.stat().st_size)
        assert sha((permanent/f.name).read_bytes()) == hashes[f.name]['sha256']
    with zipfile.ZipFile(permanent/'records.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for name in hashes:
            z.write(permanent/name, name)
    with zipfile.ZipFile(permanent/'records.zip') as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(hashes)
        for name, meta in hashes.items():
            assert sha(z.read(name))==meta['sha256']
    preservation = dict(status='NEW_LOCAL_DESCRIPTIVE_ORIGINALS_D_SHA_CRC_UNIQUE_PASSED',
                        actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                        members=hashes, archive_sha256=sha((permanent/'records.zip').read_bytes()),
                        directory=str(permanent), scope=scope)
    write_json(permanent/'preservation_receipt.json', preservation)
    print(json.dumps(dict(directory=str(out), permanent=str(permanent),
                          plan_sha256=plan_sha,
                          rows={k:{a:v for a,v in x.items() if 'strata' not in a} for k,x in rows.items()}),
                     ensure_ascii=False))

if __name__=='__main__':
    main()
