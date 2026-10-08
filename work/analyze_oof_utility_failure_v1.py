"""Post-freeze TRAIN failure analysis; no coefficient selection for an experiment."""
from pathlib import Path
import json,datetime,hashlib
import numpy as np

r=Path('D:/CodexBackups/selective_flow_20261003_1105/oof_utility_completed_20261006T033054Z')
z=np.load(r/'original_C/execute/predictions_frozen.npz');y=np.load(r/'original_C/execute/metric_labels.npz')['y'].astype('float64')
old=np.load('D:/CodexBackups/selective_flow_20261003_1105/train_oracle_completed_20261006T0158Z/c/out/diagnostics.npz')
raw=z['prediction_path'][:,0].astype('float64');mu=z['mu'].astype('float64');t=mu-raw;error=raw-y
def details(ids):
    e=error[ids];v=t[ids];c=z['prediction_path'][ids,-1].astype('float64')-raw[ids]
    dot=float(np.mean(e*v));second=float(np.mean(v**2))
    return dict(rows=int(np.sum(ids)),teacher_RMSE=float(np.sqrt(np.mean((mu[ids]-y[ids])**2))),
        original_F_RMSE=float(np.sqrt(np.mean(e**2))),teacher_disagreement_RMS=float(np.sqrt(second)),
        empirical_expected_error_disagreement_product=dot,disagreement_squared_mean=second,
        infinitesimal_teacher_pull_risk_derivative=2*dot,
        unconstrained_output_lambda_stationary_point=-dot/second if second else None,
        teacher_direction_agreement_fraction=float(np.mean(v*(-e)>0)),
        last_output_shift_RMS=float(np.sqrt(np.mean(c**2))),
        last_risk_linear_term_mean=float(np.mean(2*e*c)),last_risk_quadratic_term_mean=float(np.mean(c**2)),
        last_trust_boundary_fraction=float(np.mean(z['relative_path'][ids,-1]>=.25-1e-5)),
        teacher_proxy_last_risk_change_mean=float(np.mean(2*(raw[ids]-mu[ids])*c+c**2)))
all_rows=np.ones(1281,dtype=bool)
out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all=details(all_rows),
    folds={str(k):details(z['fold']==k) for k in range(3)},
    new_rho_RMS=float(np.sqrt(np.mean((z['p0'].astype('float64')-mu)**2))),
    old_learned_rho_RMS=float(np.sqrt(np.mean(old['estimated_residual'].astype('float64')**2))),
    frozen_residual_scale=.22068938092649817,
    input_prediction_SHA256=hashlib.sha256((r/'original_C/execute/predictions_frozen.npz').read_bytes()).hexdigest(),
    use='Post-freeze diagnostic only: empirical stationary lambda is not selected deployment hyperparameter, independent calibration or conditional-mean truth.',
    next='Separate target reliability from message solve. Freeze video-disjoint calibration/evaluation roles and compare original-F fallback, teacher disagreement shrinkage and matching zero-direction/old-residual controls. Teacher target for each student fold must come from same T_k. Current reference leakage remains explicit.')
Path('outputs/OOF教师新路径失败机制CPU分析.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
