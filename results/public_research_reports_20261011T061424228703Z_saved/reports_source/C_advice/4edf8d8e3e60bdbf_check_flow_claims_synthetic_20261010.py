"""Synthetic mathematical witnesses only; never reads MOSI or runs a model.

Run with --output PATH to save the numerical audit. These are formula checks,
not a statistical estimator qualification or evidence about a real dataset.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def run() -> dict:
    out = {}

    # An invertible two-dimensional ODE rotation has scalar readouts that
    # cannot be a function of the original scalar readout alone.
    z = np.array([[0.0, -1.0], [0.0, 1.0]])
    phi = z @ np.array([[0.0, 1.0], [-1.0, 0.0]])
    assert z[0, 0] == z[1, 0] and phi[0, 0] != phi[1, 0]
    out['scalar_output_is_not_scalar_state'] = {'base': z[:, 0].tolist(), 'flow_readout': phi[:, 0].tolist()}

    # Exact conditional affine flow: n=m+s*epsilon. Conditional surprise
    # includes both the base negative log density and the forward logdet.
    eps = np.array([0.0, 2.0])
    scale = 2.0
    logdet = math.log(scale)
    nll = 0.5 * eps**2 + 0.5 * math.log(2 * math.pi) + logdet
    assert nll[1] - nll[0] == 2.0
    out['surprise_is_not_logdet_only'] = {'same_logdet': logdet, 'nll': nll.tolist()}

    # Exact expectations by enumerating independent Rademacher variables.
    # m_T=t, m_all=t+q*a. Their squared-risk gap equals q**2.
    t, a, n = np.array([(t, a, n) for t in [-1., 1.] for a in [-1., 1.] for n in [-1., 1.]]).T
    q = 0.5
    y = t + q * a + n
    m_t, m_all = t, t + q * a
    risk = lambda p: float(np.mean((y - p)**2))
    g = float(np.mean((m_all - m_t)**2))
    assert math.isclose(risk(m_t) - risk(m_all), g)
    # A bad text predictor can be corrected from text alone with G=0.
    y_null = t + n
    bad_text = 2 * t
    text_only_gain = float(np.mean((y_null - bad_text)**2) - np.mean((y_null - t)**2))
    assert text_only_gain == 1.0
    # An arbitrary fitted prediction norm is not a lower bound on G.
    residual = y_null - t
    h = 10 * a
    delta = float(np.mean(residual**2 - (residual - h)**2))
    assert delta == -100.0 and float(np.mean(h**2)) == 100.0
    out['risk_gap_and_false_probe'] = {'oracle_G': g, 'oracle_risk_gap': risk(m_t) - risk(m_all), 'text_only_correction_when_G_zero': text_only_gain, 'false_probe_prediction_norm': float(np.mean(h**2)), 'false_probe_risk_improvement': delta}

    # f is an exact scalar rescaling of b: negative gain can improve risk
    # despite adding no information beyond b.
    y = t + n
    b, f = 2 * t, 3.25 * t
    eb, ef = b - y, f - y
    aa, bb, cc = (float(np.mean(v)) for v in [eb**2, ef**2, eb*ef])
    gain = (aa - cc) / (aa + bb - 2 * cc)
    assert math.isclose(gain, -0.8)
    calibrated = b + gain * (f - b)
    assert np.allclose(calibrated, t)
    out['negative_gain_without_new_information'] = {'g_star': gain, 'error_correlation': float(np.corrcoef(eb, ef)[0, 1]), 'f_is_b_times': 3.25 / 2, 'MSE_before': float(np.mean(eb**2)), 'MSE_after': float(np.mean((calibrated - y)**2))}

    # Two-class, common-covariance Gaussian discriminant decomposition.
    d = np.array([1.0, 0.4])
    r = 0.8
    covariance = np.array([[1.0, r], [r, 1.0]])
    joint = float(d @ np.linalg.solve(covariance, d))
    additional = (d[1] - r*d[0])**2 / (1-r*r)
    assert math.isclose(joint - d[0]**2, additional)
    out['gaussian_discriminant_identity'] = {'joint_separation_squared': joint, 'extra_separation_squared': additional, 'not_a_general_MAE_or_mutual_information_formula': True}

    mse = 0.70
    baseline_mae = math.sqrt(2 * mse / math.pi)
    out['illustrative_gaussian_MAE_and_power'] = {
        'assumed_baseline_MSE': mse,
        'baseline_MAE_under_gaussian_scale_family': baseline_mae,
        'MAE_improvement_by_residual_fraction': {str(q): baseline_mae*(1-math.sqrt(1-q)) for q in [.02, .05, .10, .20]},
        'approx_MDE_by_independent_equal_weight_groups': {str(k): (1.959963984540054+0.8416212335729143)*0.109/math.sqrt(k) for k in [31, 93]},
        'these_are_assumption_based_not_MOSI_estimates': True,
    }

    # d/n alone leaves the signal-to-noise ratio unspecified.
    variance = 100 / 1281
    out['dimension_does_not_force_zero_shrinkage'] = {
        'assumed_noise_variance_sigma2_one': variance,
        's_star_by_signal_squared': {str(theta2): theta2/(theta2+variance) for theta2 in [.001, .1, 10.]},
    }
    out['status'] = 'ALL_SYNTHETIC_FORMULA_AND_COUNTEREXAMPLE_CHECKS_PASSED'
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run()
    content = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content + '\n', encoding='utf-8')
    print(content)
