import ast,hashlib,json,shutil,zipfile,importlib.util,copy
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path(__file__).resolve().parents[1];parent=ROOT/'work/second_lease_fixed_precheck_v3_20261006T1345Z'
OUT=ROOT/'work/second_lease_fixed_execution_v1_20261006T1349Z';OUT.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in parent.iterdir():
 if p.is_file():shutil.copyfile(p,OUT/p.name)
for name in ('second_lease_precheck_wrapper_v1.py','audit_second_lease_full_precheck_cpu_v1.py','capture_new_fixed_precheck_v20.py','capture_soft_vector_v19.py'):
 shutil.copyfile(ROOT/'work'/name,OUT/name)
assert sha(OUT/'capture_soft_vector_v19.py')=='6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
for p in OUT.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
spec=importlib.util.spec_from_file_location('new_gate',OUT/'precheck_gates_v2.py');gate=importlib.util.module_from_spec(spec);spec.loader.exec_module(gate)
now=datetime.now(timezone.utc);runtime_sha=sha(OUT/'runtime_candidate_plan.json')
fixture={'scope':'NEW_MINIMAL_FIXED_FOLD0_GPU_PRECHECK_ONLY','trusted_human_or_provider_provenance_verified':True,'lease_source':'DIRECT_HUMAN_NEW_P4_24H_20261006','human_new_lease_assertion_verified':True,'lease_end_utc':(now+timedelta(hours=23)).isoformat(),'queried_actual_utc':now.isoformat(),'node':'a','gpu_uuid':gate.UUIDS['a'],'compute_processes':[],'python_full_argv':[],'host_identity_and_credentials_verified':True,'runtime_plan_sha256':runtime_sha,'assets_source_complete_space_verified':True,'remote_free_bytes':8*1024**3,'permanent_D_free_bytes':8*1024**3,'platform_lease_confirmed':False}
gate.validate_precheck_evidence(fixture,now,runtime_sha)
bad=[('gpu_uuid','GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7'),('lease_source','OLD_HEARTBEAT_ESTIMATE'),('human_new_lease_assertion_verified',False),('trusted_human_or_provider_provenance_verified',False),('lease_end_utc',(now+timedelta(hours=1)).isoformat()),('queried_actual_utc',(now-timedelta(minutes=6)).isoformat()),('compute_processes',[1]),('assets_source_complete_space_verified',False),('permanent_D_free_bytes',1)]
for key,value in bad:
 e=copy.deepcopy(fixture);e[key]=value
 try:gate.validate_precheck_evidence(e,now,runtime_sha)
 except PermissionError:continue
 raise AssertionError('NEGATIVE_GATE_NOT_REJECTED '+key)
receipt={'scope':'AST_AND_SYNTHETIC_GATE_FORMAT_ONLY_NOT_REAL_PERMISSION','actual_utc':now.isoformat(),'negative_fixtures_rejected':len(bad),'positive_synthetic_fixture_passed':True,'actual_Torch_or_GPU':False,'real_permission_provenance_not_created_by_JSON':True}
(OUT/'local_static_gate_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
plan={'status':'NEW_SECOND_LEASE_PRECHECK_EXECUTION_SOURCE_FROZEN_NOT_GPU_PASS','frozen_actual_utc':datetime.now(timezone.utc).isoformat(),'runtime_plan_sha256':runtime_sha,'parent_precheck_plan_sha256':sha(OUT/'second_lease_precheck_plan.json'),'source_sha256':{p.name:sha(p) for p in OUT.glob('*.py')},'role_and_order_sha256':{p.name:sha(p) for p in OUT.glob('*.npy')},'gpu_execution_before_fresh_gate':False,'scope':'Only public fold0 two optimizer steps plus tail no update, two-state label-free four-row donor intervention, own complete checkpoint INNER dummy replay. No performance metrics, formal100 or OUTER.','capture':'capture20 genuinely new root/argv coverage, capture19 pinned unchanged historical parent only; no legacy dummy files','CPU_auditor':'complete parameters/buffers and arrays in separate CPU process, no CPU model forward','source_provenance':'Human directly supplied new endpoints and 24h in this chat; exact platform expiry unverified, no credential values in files'}
(OUT/'deployment_execution_plan_v1.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
pkg=OUT.parent/(OUT.name+'.zip');files=[p for p in OUT.iterdir() if p.is_file()]
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for p in files:z.write(p,p.name)
with zipfile.ZipFile(pkg) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(files)
 for p in files:assert hashlib.sha256(z.read(p.name)).hexdigest()==sha(p)
print(json.dumps({'directory':str(OUT),'package':str(pkg),'package_sha256':sha(pkg),'execution_plan_sha256':sha(OUT/'deployment_execution_plan_v1.json'),'members':len(files),'actual_GPU':False,'negative_gates':len(bad)}))
