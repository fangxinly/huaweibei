"""Unqualified inference primitive: caller target may contain labels; no provenance certificate. Terminal is reevaluated on actual messages."""
import numpy as np


def backtrack_message(original, proposed, terminal, target, output_cap, epsilon=None,
                      fractions=(1.0, .5, .25, .125, .0625, .03125, .015625, .0078125)):
    original = np.asarray(original)
    proposed = np.asarray(proposed)
    target = np.asarray(target, dtype=np.float64)
    if original.shape != proposed.shape or original.ndim < 2:
        raise ValueError('Per-row message arrays must have identical shapes')
    n = len(original)
    if target.shape != (n,) or not np.isfinite(target).all():
        raise ValueError('Invalid target')
    cap = np.broadcast_to(np.asarray(output_cap, dtype=np.float64), (n,))
    if not np.isfinite(cap).all() or np.any(cap < 0):
        raise ValueError('Invalid output cap')
    if not fractions or any(not 0 < value <= 1 for value in fractions):
        raise ValueError('Fractions must lie in (0,1]')
    if any(fractions[i] <= fractions[i+1] for i in range(len(fractions)-1)):
        raise ValueError('Fractions must be strictly decreasing')
    eps = np.zeros(n) if epsilon is None else np.broadcast_to(np.asarray(epsilon, dtype=np.float64), (n,))
    if not np.isfinite(eps).all() or np.any(eps < 0):
        raise ValueError('Invalid empirical epsilon')
    raw = np.asarray(terminal(original), dtype=np.float64)
    if raw.shape != (n,) or not np.isfinite(raw).all():
        raise ValueError('Original terminal output must be finite')
    selected = original.copy()
    selected_prediction = raw.copy()
    selected_fraction = np.zeros(n)
    score = np.zeros(n)  # Relative target squared risk, optionally empirically penalized.
    accepted = np.zeros(n, dtype=bool)
    traces = []
    axes = (n,) + (1,)*(original.ndim-1)
    for fraction in fractions:
        messages = original + fraction*(proposed-original)
        pred = np.asarray(terminal(messages), dtype=np.float64)
        if pred.shape != (n,):
            raise ValueError('Terminal must return one output per row')
        delta = pred-raw
        qhat = 2*(raw-target)*delta+delta**2
        penalty = qhat+2*eps*np.abs(delta)
        legal = np.isfinite(pred) & (np.abs(delta) <= cap)
        choose = legal & (~accepted) & (penalty < 0)
        selected = np.where(choose.reshape(axes), messages, selected)
        selected_prediction = np.where(choose, pred, selected_prediction)
        selected_fraction = np.where(choose, fraction, selected_fraction)
        score = np.where(choose, penalty, score)
        accepted |= choose
        traces.append({'fraction': float(fraction), 'prediction': pred.copy(),
                       'output_cap_passed': legal.copy(), 'relative_proxy': penalty.copy()})
    # An implementation with cross-row coupling cannot combine separately evaluated
    # rows silently. Reevaluate the final assembled message and reject incoherence.
    final = np.asarray(terminal(selected), dtype=np.float64)
    if final.shape != (n,) or not np.isfinite(final).all():
        raise ValueError('Invalid final terminal output')
    if not np.allclose(final, selected_prediction, rtol=0, atol=1e-6):
        raise ValueError('Terminal has cross-row coupling or stochastic behavior')
    if np.any(np.abs(final-raw) > cap+1e-6):
        raise ValueError('Final assembled messages violate output cap')
    return {'messages': selected, 'prediction': final, 'raw_prediction': raw,
            'fraction': selected_fraction, 'accepted': accepted, 'relative_proxy': score,
            'traces': traces, 'true_labels_consumed': None,
            'target_provenance': 'UNVERIFIED_CALLER_ARRAY',
            'target_provenance_verified': False,
            'true_conditional_risk_guarantee': False}
