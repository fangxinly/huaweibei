"""CaReFlow metric semantics for frozen prediction arrays; never fit or select."""
import numpy as np

SEMANTICS = {
    'Acc7': 'All rows: clip predictions and labels to [-3,3], NumPy round (half to even), exact class match.',
    'Acc2': 'Exclude labels exactly equal to 0; classify prediction >=0 and label >=0.',
    'F1': 'Same nonzero rows and binary thresholds; class-support-weighted F1, not macro or positive-class-only F1.',
    'MAE': 'All rows, original unclipped regression predictions.',
    'Corr': 'All rows, Pearson correlation; undefined for a constant vector or fewer than 2 rows.',
    'Has0_Acc2_and_F1': 'Additional all-row binary metrics; neutral label belongs to the nonnegative class.',
    'units': 'Acc7, Acc2 and F1 stored as fractions; display as percentages. Corr fraction, MAE original sentiment scale.',
    'precision': 'Independent CPU float64 aggregation; no rounding, clipping or calibration of regression predictions.'
}

def _binary_metrics(prediction, labels):
    if len(labels) == 0:
        return {'Acc2': None, 'F1': None, 'count': 0, 'confusion_true_rows_pred_cols': [[0,0],[0,0]]}
    target, predicted = labels >= 0, prediction >= 0
    matrix = np.bincount(2*target.astype(np.int64)+predicted.astype(np.int64), minlength=4).reshape(2,2)
    supports = matrix.sum(axis=1)
    denominators = supports + matrix.sum(axis=0)
    scores = np.divide(2*np.diag(matrix), denominators, out=np.zeros(2,dtype=np.float64), where=denominators!=0)
    return {'Acc2': float(np.trace(matrix)/len(labels)), 'F1': float(np.dot(scores,supports)/len(labels)),
            'count': len(labels), 'confusion_true_rows_pred_cols': matrix.tolist()}

def metrics(predictions, labels):
    p, y = np.asarray(predictions,dtype=np.float64), np.asarray(labels,dtype=np.float64)
    if p.ndim != 1 or y.ndim != 1 or p.shape != y.shape or len(y) == 0:
        raise ValueError('Require nonempty equal-shape 1D prediction and label arrays')
    if not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Nonfinite prediction or label; do not drop failing rows')
    nonzero = y != 0
    binary, has_zero = _binary_metrics(p[nonzero],y[nonzero]), _binary_metrics(p,y)
    pc, yc = p-p.mean(), y-y.mean()
    norm = np.linalg.norm(pc)*np.linalg.norm(yc)
    corr = float(np.dot(pc,yc)/norm) if len(y)>1 and norm>0 else None
    return {
        'Acc7': float(np.mean(np.round(np.clip(p,-3,3))==np.round(np.clip(y,-3,3)))),
        'Acc2': binary['Acc2'], 'F1': binary['F1'], 'MAE': float(np.mean(np.abs(p-y))), 'Corr': corr,
        'MSE': float(np.mean((p-y)**2)), 'Has0_Acc2': has_zero['Acc2'], 'Has0_F1': has_zero['F1'],
        'samples': len(y), 'nonzero_samples': int(nonzero.sum()), 'zero_label_samples': int((~nonzero).sum()),
        'exact_zero_predictions': int((p==0).sum()), 'correlation_defined': corr is not None,
        'nonzero_confusion_matrix': binary['confusion_true_rows_pred_cols'],
        'haszero_confusion_matrix': has_zero['confusion_true_rows_pred_cols']
    }
