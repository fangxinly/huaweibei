"""Synthetic dependency/leakage checks only; no study outcome labels loaded."""
from pathlib import Path
import datetime, hashlib, importlib.util, json
import numpy as np

work = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('new_calibration_math', work/'amplitude_calibrator_draft_v1.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
rng = np.random.RandomState(91819)
rows = np.arange(97)
cal = rows[:31]
p = rng.normal(size=97)
t = rng.normal(size=97)
mu = p+t
y_cal = p[cal]+.13*t[cal]+rng.normal(scale=.01, size=len(cal))
fold = np.zeros(97, dtype=np.int64)
fit = module.fit_given_calibration_labels(p, mu, rows, cal, y_cal, cal, fold)
lam = fit['amplitude']
assert 0 < lam < 1
formula = lambda a: np.mean((p[cal]+a*t[cal]-y_cal)**2)
for delta in (.001, .01, .1):
    assert formula(lam) <= formula(max(0, lam-delta))+1e-14
    assert formula(lam) <= formula(min(1, lam+delta))+1e-14
assert abs(formula(lam)-formula(0)-fit['calibration_relative_risk']) < 1e-14
changed_p, changed_mu = p.copy(), mu.copy()
changed_p[31:] = np.nan
changed_mu[31:] = 10**6
invariant = module.fit_given_calibration_labels(changed_p, changed_mu, rows, cal, y_cal, cal, fold)
assert fit == invariant  # Non-CAL predictions cannot influence fitted amplitude.
zero = module.fit_given_calibration_labels(p, p, rows, cal, y_cal, cal, fold)
assert zero['amplitude'] == 0 and zero['calibration_relative_risk'] == 0
wrong = module.fit_given_calibration_labels(p, mu, rows, cal, p[cal]-t[cal], cal, fold)
assert wrong['amplitude'] == 0
saturated = module.fit_given_calibration_labels(p, mu, rows, cal, p[cal]+2*t[cal], cal, fold)
assert saturated['amplitude'] == 1
rejected = 0
for bad_rows, bad_labels, bad_fold in [
    (np.r_[cal[:-1], 50], y_cal, fold),
    (cal[::-1], y_cal, fold),
    (cal, y_cal, np.r_[np.ones(1, dtype=np.int64), fold[1:]])]:
    try:
        module.fit_given_calibration_labels(p, mu, rows, bad_rows, bad_labels, cal, bad_fold)
    except PermissionError:
        rejected += 1
    else:
        raise AssertionError('Role/fold mismatch not rejected')
assert rejected == 3
source_sha = hashlib.sha256((work/'amplitude_calibrator_draft_v1.py').read_bytes()).hexdigest()
result = {'status': 'SYNTHETIC_CAL_ONLY_BOUNDED_LS_AMPLITUDE_CHECKED_NOT_REAL_CALIBRATION',
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'source_sha256': source_sha,
    'synthetic_rows': 97, 'synthetic_cal_rows': 31, 'synthetic_amplitude': lam,
    'risk_stationary_and_boundary_checks': True, 'non_cal_prediction_mutation_invariant': True,
    'zero_signal_and_wrong_direction_fallback': True, 'full_step_boundary': True,
    'evaluation_role_or_teacher_mixing_requests_rejected': rejected,
    'real_labels_loaded': False, 'real_calibrator_fitted': False,
    'real_gpu_solver_integrated': False, 'real_output_cap_fitted': False, 'real_epsilon_fitted': False,
    'true_risk_guarantee': False,
    'scope': 'This draft is a target-amplitude primitive, not a deployed controller or an empirical result. Real execution must wait for the existing collection, label-role, source, GPU and preservation gates.'}
path = work.parent/'outputs/纠错幅度CAL限定最小二乘草稿合成核验.json'
assert not path.exists()
path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(result['status'])
