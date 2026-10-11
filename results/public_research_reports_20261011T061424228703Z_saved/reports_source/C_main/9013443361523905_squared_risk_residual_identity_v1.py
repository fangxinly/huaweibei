"""Synthetic algebra only. No official data, checkpoints, GPU or fitted task model."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work/squared_risk_residual_math_20261006T1259Z'
OUT.mkdir(exist_ok=True)
started = time.perf_counter()
seed = 19723
rng = np.random.default_rng(seed)
n = 10000
p = rng.normal(size=n)
delta = rng.normal(size=n)
delta[::97] = 0.0
r = rng.normal(size=n)
h = rng.normal(size=n)
y = p + r
q = delta * delta - 2.0 * delta * r
qhat = delta * delta - 2.0 * delta * h
loss_difference = (p + delta - y)**2 - (p - y)**2
q_squared_error = (qhat - q)**2
weighted_residual_error = 4.0 * delta**2 * (h - r)**2
identity_error = float(np.max(np.abs(loss_difference - q)))
weight_error = float(np.max(np.abs(q_squared_error - weighted_residual_error)))
assert identity_error < 1e-12 and weight_error < 1e-12
assert np.all(q[delta == 0] == 0) and np.all(qhat[delta == 0] == 0)

# Exact four equally likely atoms; delta is generated from observable X, not Y.
x = np.array([-1., -1., 1., 1.])
noise = np.array([-.4, .4, -.4, .4])
small_y = .75*x + noise
small_delta = x.copy()
base = np.zeros(4)
true_residual_mean_given_full_info = .75*x
exact_q = small_delta**2 - 2.0*small_delta*small_y
true_conditional_q = np.array([exact_q[x == v].mean() for v in x])
structured_full = small_delta**2 - 2.0*small_delta*true_residual_mean_given_full_info
assert np.allclose(true_conditional_q, structured_full, atol=1e-15, rtol=0)
coarse_structured = small_delta**2  # E[r|p_F=0] is 0, delta is omitted from h.
cov_delta_r = float(np.mean(small_delta*small_y) - small_delta.mean()*small_y.mean())
coarse_q_by_covariance = float(np.mean(small_delta**2) - 2.0*(small_delta.mean()*small_y.mean()+cov_delta_r))
assert abs(coarse_q_by_covariance - exact_q.mean()) < 1e-15
assert np.all(coarse_structured > 0) and np.all(structured_full < 0)
toy_mse = {'F': float(np.mean((base-small_y)**2)),
           'candidate': float(np.mean((small_delta-small_y)**2)),
           'conditional_mean_direct': float(np.mean((true_residual_mean_given_full_info-small_y)**2))}
assert np.allclose(list(toy_mse.values()), [.7225, .2225, .16], atol=1e-15, rtol=0)

# One shared residual estimate conditioned on the complete pool information.
pool = np.column_stack([np.zeros(n), delta, -.3*delta, .5*delta])
pool_qhat = pool**2 - 2*pool*h[:, None]
nearest = (pool-h[:, None])**2
idx_q = np.argmin(pool_qhat, axis=1)
idx_nearest = np.argmin(nearest, axis=1)
assert np.array_equal(idx_q, idx_nearest)
assert np.all(np.min(pool_qhat, axis=1) <= 0)

# Same constant hypothesis class; different losses can change fitted coefficients.
wd = np.array([.1, 1.])
wr = np.array([1., -1.])
h_unweighted = float(wr.mean())
h_weighted = float(np.sum(wd**2*wr)/np.sum(wd**2))
weighted_toy = {}
for name, value in [('unweighted', h_unweighted), ('delta_squared_weighted', h_weighted)]:
    weighted_toy[name] = {'h': value,
        'residual_mse': float(np.mean((value-wr)**2)),
        'structured_Q_mse': float(np.mean(4*wd**2*(value-wr)**2))}
assert weighted_toy['delta_squared_weighted']['structured_Q_mse'] < weighted_toy['unweighted']['structured_Q_mse']
assert weighted_toy['delta_squared_weighted']['residual_mse'] > weighted_toy['unweighted']['residual_mse']

np.savez(OUT/'synthetic_witnesses.npz', p=p, delta=delta, r=r, h=h,
         pool=pool, index_Q=idx_q, index_nearest=idx_nearest,
         toy_x=x, toy_noise=noise, toy_y=small_y, toy_delta=small_delta,
         toy_Q=exact_q, toy_full_Qhat=structured_full, toy_coarse_Qhat=coarse_structured)
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
receipt = {'status':'SYNTHETIC_NUMPY_IDENTITIES_AND_FINITE_COUNTEREXAMPLES_PASSED',
    'actual_utc':datetime.now(timezone.utc).isoformat(), 'elapsed_seconds':time.perf_counter()-started,
    'seed':seed,'random_rows':n,'exact_distribution_atoms':4,
    'loss_difference_identity_max_error':identity_error,
    'structured_Q_MSE_weighted_residual_identity_max_error':weight_error,
    'zero_delta_Q_exact':True, 'shared_full_pool_nearest_rule_equal':True,
    'omitted_delta_toy':{'covariance':cov_delta_r,'true_expected_Q':float(exact_q.mean()),
        'coarse_Qhat':1.0,'toy_MSE_not_task_scores':toy_mse},
    'weighted_constant_toy_not_task_scores':weighted_toy,
    'source_sha256':sha(Path(__file__)), 'arrays_sha256':sha(OUT/'synthetic_witnesses.npz'),
    'official_data_or_labels_read':False,'Torch_or_GPU':False,'new_task_scores':False,
    'other_node_CPU_or_remote_capture':False,'whole_research_or_lease_complete':False}
(OUT/'synthetic_math_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
