"""New CPU analysis of preserved TRAIN arrays; no GPU rerun or deployment tuning."""
from pathlib import Path
import datetime,hashlib,json,shutil
import numpy as np
w=Path(__file__).resolve().parent;o=w.parent/'outputs'
source=Path('D:/CodexBackups/selective_flow_20261003_1105/train_oracle_completed_20261006T0158Z/c/out/diagnostics.npz')
digest=hashlib.sha256(source.read_bytes()).hexdigest();assert digest=='92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658'
with np.load(source,allow_pickle=False) as z:
 y=z['y'].astype(np.float64);p0=z['p0'].astype(np.float64);raw=z['raw_prediction'].astype(np.float64);control=z['learned_prediction_path'][:,-1].astype(np.float64);rho=z['learned_residual'].astype(np.float64);video=z['video'].copy()
assert len(y)==1281 and len(set(video))==52
d=control-raw;rhat_raw=rho+(raw-p0)
proxy=2*rhat_raw*d+d*d;observed=(control-y)**2-(raw-y)**2
identity_error=float(np.max(np.abs(observed-proxy-2*((p0-y)-rho)*d)));assert identity_error<1e-12
rows=[]
for eps in [0.,.025,.05,.1,.2,.4]:
 upper=proxy+2*eps*np.abs(d);accept=upper<0;prediction=np.where(accept,control,raw)
 rows.append({'illustrative_error_budget':eps,'accepted_rows':int(accept.sum()),'accepted_fraction':float(accept.mean()),'accepted_observed_harm_fraction':float(np.mean(observed[accept]>0)) if accept.any() else None,'mean_observed_risk_change':float(np.mean((prediction-y)**2-(raw-y)**2)),'TRAIN_mse':float(np.mean((prediction-y)**2))})
# An explicitly label-known algebra witness, never a deployable uncertainty estimator.
label_known_error=np.abs(rho-(p0-y));upper=proxy+2*label_known_error*np.abs(d)
assert np.max(observed-upper)<1e-12
accept=upper<0;assert np.all(observed[accept]<1e-12)
result={'status':'PRESERVED_TRAIN_ARRAYS_RAW_RELATIVE_ERROR_IDENTITY_AND_CONSERVATIVE_ACCEPTANCE_DIAGNOSTIC','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':digest,'rows':1281,'videos':52,'raw_relative_exact_identity_max_error':identity_error,'raw_TRAIN_mse':float(np.mean((raw-y)**2)),'unrestricted_learned_TRAIN_mse':float(np.mean((control-y)**2)),'illustrative_budgets_not_parameter_selection':rows,'label_known_upper_witness':{'accepted_rows':int(accept.sum()),'accepted_observed_harm_rows':int(np.sum(observed[accept]>1e-12)),'mean_risk_change':float(np.mean(np.where(accept,observed,0.)))},'scope':'Same already fullTRAINfit/DEVselected frozen C2 and preserved TRAIN Oracle arrays. Fixed illustrative error budgets are not fitted/calibrated deployment bounds and select no experiment setting. The label-known witness uses y and is never a deployment rule. No GPU rerun, parameter update, new weight, DEV/TEST access, conditional coverage, or generalization conclusion.'}
out=o/'TRAIN_Oracle原消息相对保守接受诊断.json';assert not out.exists();out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'identity_error':identity_error,'illustrative_rows':rows,'label_known_witness':result['label_known_upper_witness']},ensure_ascii=False))
