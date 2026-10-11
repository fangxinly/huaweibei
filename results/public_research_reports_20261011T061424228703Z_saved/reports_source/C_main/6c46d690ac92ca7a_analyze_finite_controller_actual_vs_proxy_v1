from pathlib import Path
import json,datetime,zipfile,io
import numpy as np
root=Path(__file__).parent.parent;base=Path('D:/CodexBackups/selective_flow_20261003_1105');shots=base/'finite_c2_completed_snapshots_20261005T1627Z';rows={}
for n in ['a','b','c']:
 with zipfile.ZipFile(shots/n/'snapshot.zip') as z:
  pre='finite_c2/diagnostics_v4/' if n=='c' else 'finite_diagnostics_v3/';d=np.load(io.BytesIO(z.read(pre+'diagnostics.npz')))
  run='finite_c2/run/' if n=='c' else 'finite_run/';saved=np.load(io.BytesIO(z.read(run+'predictions.npz')))
  y=d['valid_y'].astype('float64');p0=d['reference_prediction'].astype('float64');rho=d['estimated_residual'].astype('float64');raw=d['raw_fixed_prediction'].astype('float64');final=d['prediction_default'].astype('float64')
  change=lambda pred:2*rho*(pred-p0)+(pred-p0)**2
  proxy=change(final)-change(raw);actual=(final-y)**2-(raw-y)**2
  row=dict(proxy_control_vs_raw_mean=float(proxy.mean()),actual_control_vs_raw_mean=float(actual.mean()),proxy_improved_fraction=float(np.mean(proxy<0)),actual_improved_fraction=float(np.mean(actual<0)),proxy_actual_improvement_sign_agreement=float(np.mean((proxy<0)==(actual<0))),estimated_residual_sign_accuracy=float(np.mean((rho>0)==((p0-y)>0))),residual_true_positive_fraction=float(np.mean(p0-y>0)),residual_mae=float(np.mean(np.abs(rho-(p0-y)))),proxy_actual_difference_identity_error=float(np.max(np.abs((proxy-actual)-2*(rho-(p0-y))*(final-raw)))),final_context_to_raw_norm=float(np.linalg.norm(d['transmitted_message']-d['message'])),raw_fixed_mse=float(np.mean((raw-y)**2)),default_mse=float(np.mean((final-y)**2)))
  assert row['proxy_actual_difference_identity_error']<1e-12
  if n=='c':
   trust=saved['trust_relative_change'];assert trust.shape==(229,) and np.isfinite(trust).all() and float(trust.max())<.25001
   row['trust_relative_change']={'min':float(trust.min()),'mean':float(trust.mean()),'max':float(trust.max()),'boundary_fraction_at_0_25':float(np.mean(np.isclose(trust,.25,atol=1e-6)))}
   assert saved['inner_objectives'].shape==(229,3) and np.isfinite(saved['inner_objectives']).all()
   row['inner_objectives_mean']=saved['inner_objectives'].mean(0).tolist()
  rows[n]=row
out=dict(status='FROZEN_DEV_CONTROLLERS_PROXY_VERSUS_ACTUAL_RISK_IDENTITY_AND_C2_TRUST_ANALYSIS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,scope='DEVselected single seed diagnostic only. Signed residual is an estimator, not true conditional rho. Final squared-loss difference decomposes exactly with observed label residual; no TEST, no tuning, no semantic truth.')
(root/'outputs/有限任务风险控制代理与实际风险及信赖域分析.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(rows))
