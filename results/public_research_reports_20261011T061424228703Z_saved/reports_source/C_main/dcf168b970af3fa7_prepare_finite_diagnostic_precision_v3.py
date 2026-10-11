from pathlib import Path
import json,hashlib,datetime,base64
w=Path(__file__).parent
s=(w/'diagnose_finite_task_risk_v2.py').read_text(encoding='utf-8').replace('finite_diagnostics_plan_v2.json','finite_diagnostics_plan_v3.json')
s=s.replace('identity_error=0.;context_error=0.','identity_error=0.;identity_float32_error=0.;context_error=0.')
old="trueformula=2*rho[:,None]*(withoutdelta-withdelta[:,None])+withoutdelta.square()-withdelta[:,None].square();identity_error=max(identity_error,float((trueformula-utilitytrue).abs().max()))"
new="""trueformula=2*rho[:,None]*(withoutdelta-withdelta[:,None])+withoutdelta.square()-withdelta[:,None].square();identity_float32_error=max(identity_float32_error,float((trueformula-utilitytrue).abs().max()))
  # Verify the algebra from the same frozen predictions in float64. Model
  # predictions and float32 utility arrays stay unchanged, as do interventions.
  r64=b['reference_prediction'].double();y64=b['y'].double();w64=raw_with.double();o64=raw_without.double()
  dw64=w64-r64;do64=o64-r64[:,None];rho64=r64-y64
  algebra64=2*rho64[:,None]*(do64-dw64[:,None])+do64.square()-dw64[:,None].square()
  loss64=(o64-y64[:,None]).square()-(w64-y64)[:,None].square()
  identity_error=max(identity_error,float((algebra64-loss64).abs().max()))"""
assert old in s;s=s.replace(old,new)
s=s.replace('identity_error=identity_error);','identity_error=identity_error,identity_float32_error=identity_float32_error);')
s=s.replace('identity_error<2e-6','identity_error<1e-12')
s=s.replace('finite_squared_identity_error=identity_error,','finite_squared_identity_error=identity_error,finite_squared_identity_float32_error=identity_float32_error,identity_verification_dtype="float64 from unchanged float32 predictions",')
p=w/'diagnose_finite_task_risk_v3.py';assert not p.exists();p.write_text(s,encoding='utf-8')
plan=json.loads((w/'finite_diagnostics_plan_v2.json').read_text(encoding='utf-8'))
plan.update(source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),freeze_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),amendment='V3 verifies algebra in float64 from the SAME model predictions with tighter 1e-12 tolerance; float32 predictions, utility arrays, conditions and parameters unchanged; reports original float32 cancellation separately. V1/V2 rejected attempts retained.',prior_attempt_errors={'a':2.2649765014648438e-6,'b':2.734363079071045e-6,'replay_label_context':0})
q=w/'finite_diagnostics_plan_v3.json';q.write_text(json.dumps(plan,indent=2),encoding='utf-8')
(w/'finite_diag_precision_payload.json').write_text(json.dumps({x.name:base64.b64encode(x.read_bytes()).decode() for x in [p,q]}),encoding='utf-8')
print(json.dumps({'source_sha256':plan['source_sha256'],'plan_sha256':hashlib.sha256(q.read_bytes()).hexdigest()}))
