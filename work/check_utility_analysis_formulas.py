"""Synthetic formula checks only; does not evaluate any actual model."""
import ast,hashlib,json
from pathlib import Path
import numpy as np
import analyze_utility_completed_v1 as m
root=Path(__file__).resolve().parent
for p in [root/'analyze_utility_completed_v1.py',root/'save_utility_heartbeat_v1.py']:ast.parse(p.read_text(encoding='utf-8'))
assert np.array_equal(m.rank(np.array([3.,1.,1.,2.])),np.array([3.,.5,.5,2.]))
q=np.array([-2.,-.1,0.,.1,2.]);u=np.tanh(q/.5)
r=m.calibration(u,q);assert r['smooth_l1']==0 and r['MSE_to_target']==0 and r['sign_accuracy_nonzero_q']==1 and r['balanced_sign_recall']==1
r=m.calibration(np.zeros_like(q),q);assert r['sign_accuracy_nonzero_q']==0 and r['balanced_sign_recall']==0 and r['spearman_u_q'] is None
r=m.calibration(-u,q);assert r['sign_accuracy_nonzero_q']==0 and np.isclose(r['spearman_u_q'],-1)
y=np.linspace(-2,2,229);on=y+.2;off=y-.3;delta=on-off;e=off-y
assert np.max(np.abs(((off-y)**2-(on-y)**2)-(-2*e*delta-delta*delta)))<1e-10
print(json.dumps({'status':'AST_AND_SYNTHETIC_FORMULAS_VERIFIED','model_results_computed':False,'analyzer_sha256':hashlib.sha256((root/'analyze_utility_completed_v1.py').read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256((root.parent/'outputs/逐样本效用冻结诊断计划.md').read_bytes()).hexdigest()}))
