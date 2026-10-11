"""Local mathematical review; no Torch, remote execution, or dataset access."""
import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parents[1]
now = datetime.datetime.now(datetime.timezone.utc)
stamp = now.strftime('%Y%m%dT%H%M%SZ')
out = root / 'outputs' / f'外部AI建议数学核对_{stamp}.json'

# A perfect conditional-mean teacher can improve expected risk while helping
# only half the individual observed labels. This is not observed research data.
y = np.array([-1., 1.], dtype=np.float64)
prob = np.array([.5, .5], dtype=np.float64)
mu = float(prob @ y)
p0 = .1
delta = -.1
rhohat = p0 - mu
proxy = 2 * rhohat * delta + delta ** 2
observed = (p0 + delta - y) ** 2 - (p0 - y) ** 2
assert np.isclose(proxy, prob @ observed, atol=1e-15, rtol=0)
assert np.isclose(prob @ (observed < 0), .5, atol=1e-15, rtol=0)

# Same nonlinear terminal, reference, proximal penalty, scale, and beta.
# Compare both objectives and their analytically computed gradient/Hessian.
z = np.array([-.7, -.2, .0, .4, .9], dtype=np.float64)
f, target, beta, scale = .15, -.4, .6, .7
p = .2 + z + .3 * z ** 2
dp = 1 + .6 * z
d2p = np.full_like(z, .6)
d = p - p0
rho = p0 - target
prox = .5 * (z - f) ** 2
direct = prox + beta * (p - target) ** 2 / scale ** 2
expanded = prox + beta * (2 * rho * d + d ** 2) / scale ** 2
constant = beta * (p0 - target) ** 2 / scale ** 2
gd = z - f + 2 * beta * (p - target) * dp / scale ** 2
ge = z - f + 2 * beta * (rho + d) * dp / scale ** 2
hd = 1 + 2 * beta * (dp ** 2 + (p - target) * d2p) / scale ** 2
he = 1 + 2 * beta * (dp ** 2 + (rho + d) * d2p) / scale ** 2
errors = {
    'objective_constant_error': float(np.max(np.abs(direct - expanded - constant))),
    'analytic_gradient_error': float(np.max(np.abs(gd - ge))),
    'analytic_hessian_error': float(np.max(np.abs(hd - he))),
}
assert max(errors.values()) < 1e-12

record = {
    'utc': now.isoformat(),
    'scope': 'local_float64_mathematical_examples_not_GPU_or_research_dataset',
    'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'perfect_teacher_counterexample': {
        'label_values': y.tolist(), 'probabilities': prob.tolist(),
        'conditional_mean': mu, 'reference': p0, 'delta': delta,
        'conditional_teacher_error': 0., 'proxy_risk_change': proxy,
        'observed_loss_changes': observed.tolist(),
        'expected_loss_change': float(prob @ observed),
        'proxy_improvement_fraction': 1.,
        'observed_improvement_fraction': float(prob @ (observed < 0)),
    },
    'oracle_objectives_equivalence': errors,
    'oracle_experiment_executed': False,
    'teacher_experiment_executed': False,
    'formal_protocol_changed': False,
}
out.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({'evidence': str(out), 'errors': errors,
                  'perfect_teacher_observed_improvement_fraction': .5}, ensure_ascii=False))
