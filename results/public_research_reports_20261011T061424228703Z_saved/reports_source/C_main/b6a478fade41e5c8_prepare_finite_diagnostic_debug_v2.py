from pathlib import Path
import json,hashlib,datetime
s=Path('work/diagnose_finite_task_risk_v1.py').read_text(encoding='utf-8').replace('finite_diagnostics_plan_v1.json','finite_diagnostics_plan_v2.json')
line='assert replay==label_error==context_error==0 and p0error<2e-5 and identity_error<2e-6'
s=s.replace(line,"validation=dict(replay=replay,label_error=label_error,context_error=context_error,p0error=p0error,identity_error=identity_error);(a.out/'attempt_validation.json').write_text(json.dumps(validation,indent=2));print('FINITE_DIAGNOSTIC_VALIDATION',validation,flush=True)\n"+line)
p=Path('work/diagnose_finite_task_risk_v2.py');p.write_text(s,encoding='utf-8');plan=json.loads(Path('work/finite_diagnostics_plan_v1.json').read_text());plan.update(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),amendment='v1 late validation failed without logging components; v2 logs exact components before the same assertions. No scientific condition/parameters/selection/control changes, original source/output kept.')
Path('work/finite_diagnostics_plan_v2.json').write_text(json.dumps(plan,indent=2))
