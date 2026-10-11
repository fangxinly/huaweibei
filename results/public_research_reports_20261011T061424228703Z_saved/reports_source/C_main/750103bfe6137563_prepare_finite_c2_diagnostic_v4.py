from pathlib import Path
import hashlib,json,datetime
w=Path(__file__).parent;s=(w/'diagnose_finite_task_risk_v3.py').read_text(encoding='utf-8').replace('finite_task_risk_runtime_v1','finite_task_risk_runtime_v2').replace('finite_diagnostics_plan_v3.json','finite_diagnostics_plan_v4.json')
p=w/'diagnose_finite_task_risk_v4.py';assert not p.exists();p.write_text(s,encoding='utf-8')
plan=json.loads((w/'finite_diagnostics_plan_v3.json').read_text(encoding='utf-8'));plan.update(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),freeze_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),amendment='C2 runtime analytic-single-token only; same20 primary +7raw calibration; same precision v3. Run only after natural100completion and selectedepoch restoration; no additional training or TEST.',budget='Only completedC2:20primary plus7raw calibration; max estimate30min; no training.',c2_formal_plan_sha256=hashlib.sha256((w/'finite_formal_plan_v2.json').read_bytes()).hexdigest())
(w/'finite_diagnostics_plan_v4.json').write_text(json.dumps(plan,indent=2),encoding='utf-8');print(plan['source_sha256'])
