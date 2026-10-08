"""CAL-only amplitude mathematics. No main, no real labels loaded or fit here."""
import hashlib
import numpy as np


def fit_given_calibration_labels(p_f, teacher_mu, all_row_ids, cal_row_ids, cal_labels,
                                 authorized_cal_ids, teacher_fold):
    """Return a bounded LS amplitude, without accepting evaluation labels."""
    rows = np.asarray(all_row_ids, dtype=np.int64)
    cal = np.asarray(cal_row_ids, dtype=np.int64)
    allowed = np.asarray(authorized_cal_ids, dtype=np.int64)
    p_f = np.asarray(p_f, dtype=np.float64)
    teacher_mu = np.asarray(teacher_mu, dtype=np.float64)
    labels = np.asarray(cal_labels, dtype=np.float64)
    teacher_fold = np.asarray(teacher_fold, dtype=np.int64)
    if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
        raise ValueError('Unique global row IDs required')
    if p_f.shape != rows.shape or teacher_mu.shape != rows.shape or teacher_fold.shape != rows.shape:
        raise ValueError('Prediction arrays must align with global row IDs')
    if cal.ndim != 1 or labels.shape != cal.shape or not len(cal):
        raise ValueError('Only aligned nonempty CAL outcome labels accepted')
    if len(np.unique(cal)) != len(cal) or not np.array_equal(cal, allowed):
        raise PermissionError('Exactly the frozen authorized CAL rows required')
    if not np.isfinite(labels).all():
        raise ValueError('Nonfinite CAL outcome')
    lookup = {int(row): index for index, row in enumerate(rows)}
    if not set(cal.tolist()) <= set(lookup):
        raise PermissionError('Unknown CAL row')
    positions = np.asarray([lookup[int(row)] for row in cal], dtype=np.int64)
    p = p_f[positions]
    mu = teacher_mu[positions]
    if not np.isfinite(p).all() or not np.isfinite(mu).all():
        raise ValueError('Nonfinite CAL prediction')
    folds = np.unique(teacher_fold[positions])
    if len(folds) != 1 or int(folds[0]) not in (0, 1, 2):
        raise PermissionError('CAL targets must come from the same T_k')
    direction = mu-p
    residual = p-labels
    linear = float(np.mean(residual*direction))
    quadratic = float(np.mean(direction*direction))
    amplitude = 0.0 if quadratic == 0 else float(np.clip(-linear/quadratic, 0, 1))
    q = 2*amplitude*linear+amplitude**2*quadratic
    if not np.isfinite(amplitude) or not np.isfinite(q):
        raise ValueError('Nonfinite CAL calibration')
    digest = hashlib.sha256()
    for value in (cal, p, mu, labels):
        digest.update(str((value.shape, str(value.dtype))).encode())
        digest.update(value.tobytes())
    return {'fold': int(folds[0]), 'amplitude': amplitude,
            'calibration_rows': len(cal), 'linear_mean': linear,
            'direction_square_mean': quadratic, 'calibration_relative_risk': q,
            'calibration_inputs_sha256': digest.hexdigest(),
            'evaluation_labels_received': False, 'evaluation_metrics_computed': False,
            'conditional_mean_risk_guarantee': False, 'whole_pipeline_crossfit': False,
            'output_cap_fitted': False, 'epsilon_fitted': False}


def prediction_target(p_f, teacher_mu, amplitude):
    """A solver target only. This function never returns modified messages."""
    p_f = np.asarray(p_f, dtype=np.float64)
    teacher_mu = np.asarray(teacher_mu, dtype=np.float64)
    if p_f.shape != teacher_mu.shape or not np.isfinite(p_f).all() or not np.isfinite(teacher_mu).all():
        raise ValueError('Finite aligned prediction arrays required')
    if not np.isfinite(amplitude) or not 0 <= amplitude <= 1:
        raise ValueError('Amplitude must lie in [0,1]')
    return p_f+amplitude*(teacher_mu-p_f)
