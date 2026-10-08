import ast,copy,hashlib,importlib.util,json
from pathlib import Path
from datetime import datetime,timezone,timedelta
ROOT=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
PRE=ROOT/'work/minimal_fixed_precheck_local_20261006T1240Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((PRE/'precheck_candidate_plan_v2.json').read_text(encoding='utf-8'))
for name,h in plan['source_sha256'].items():
 assert sha(PRE/name)==h
 compile((PRE/name).read_text(encoding='utf-8'),str(PRE/name),'exec')
tree=ast.parse((PRE/'precheck_minimal_fixed_v2.py').read_text(encoding='utf-8'))
run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_precheck')
gate_index=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Call) and isinstance(n.value.func,ast.Name) and n.value.func.id=='validate_precheck_evidence')
torch_index=next(i for i,n in enumerate(run.body) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names))
assert gate_index<torch_index
loops=[n for n in ast.walk(run) if isinstance(n,ast.For)]
step_loop=next(n for n in loops if any(isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='step' for c in ast.walk(n)))
assert isinstance(step_loop.iter,ast.Call) and step_loop.iter.func.id=='enumerate'
assert tuple(v.value for v in step_loop.iter.args[0].elts)==('normal','mixed_singleton')
step_calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='step']
assert len(step_calls)==2 and all(n in list(ast.walk(step_loop)) for n in step_calls)
strict_calls=[n for n in ast.walk(run) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='load_state_dict']
assert len(strict_calls)==1 and any(k.arg=='strict' and k.value.value is True for k in strict_calls[0].keywords)
assert not any(isinstance(n,ast.Attribute) and n.attr=='inner_labels_after_frozen_predictions' for n in ast.walk(run))
assignments={n.targets[0].id:n.value for n in ast.walk(run) if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)}
assert isinstance(assignments['auxiliary'],ast.BinOp) and not isinstance(assignments['auxiliary'].op,ast.Sub)
spec=importlib.util.spec_from_file_location('precheck_gates_only',PRE/'precheck_gates_v1.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
now=datetime.now(timezone.utc);runtime_sha=plan['runtime_plan_sha256'];rejected=[]
fixture={'scope':'NEW_MINIMAL_FIXED_FOLD0_GPU_PRECHECK_ONLY','trusted_human_or_provider_provenance_verified':True,
 'platform_lease_confirmed':True,'lease_end_utc':(now+timedelta(hours=4)).isoformat(),'queried_actual_utc':now.isoformat(),
 'node':'c','gpu_uuid':g.UUIDS['c'],'compute_processes':[],'python_full_argv':[],
 'host_identity_and_credentials_verified':True,'runtime_plan_sha256':runtime_sha,
 'assets_source_complete_space_verified':True,'remote_free_bytes':5*1024**3,'permanent_D_free_bytes':7*1024**3}
assert g.validate_precheck_evidence(fixture,now,runtime_sha)['formal100_or_outer_authorized'] is False
for name,key,value in [('estimate_not_confirmed','platform_lease_confirmed',False),('missing_provenance','trusted_human_or_provider_provenance_verified',False),('wrong_UUID','gpu_uuid','GPU-wrong'),('GPU_busy','compute_processes',[{'pid':1}]),('missing_argv','python_full_argv',None),('unverified_host','host_identity_and_credentials_verified',False),('source_plan_mismatch','runtime_plan_sha256','0'*64),('assets_missing','assets_source_complete_space_verified',False),('remote_space_low','remote_free_bytes',0),('D_space_low','permanent_D_free_bytes',0),('expired_lease','lease_end_utc',(now-timedelta(minutes=1)).isoformat()),('insufficient_reserve','lease_end_utc',(now+timedelta(hours=2)).isoformat()),('stale_query','queried_actual_utc',(now-timedelta(minutes=6)).isoformat())]:
 e=copy.deepcopy(fixture);e[key]=value
 try:g.validate_precheck_evidence(e,now,runtime_sha)
 except PermissionError:rejected.append(name)
 else:raise AssertionError('NEGATIVE_GATE_NOT_REJECTED: '+name)
meta={'format':'MINIMAL_FIXED_FULL_MODEL_STATE_V1','runtime_plan_sha256':runtime_sha,'fold':0,'seed':91819,'scope':'CLEAN_INITIAL_NO_PRECHECK','optimizer_steps':0,'state_sha256':'0'*64,'fit_statistics_sha256':'1'*64}
assert g.validate_saved_state_metadata(meta,runtime_sha,'CLEAN_INITIAL_NO_PRECHECK',0)
try:g.validate_saved_state_metadata(meta,runtime_sha,'PRECHECK_TWO_STEPS_NOT_FORMAL',2)
except ValueError:rejected.append('clean_initial_as_two_step_state')
else:raise AssertionError('STATE_SUBSTITUTION_NOT_REJECTED')
receipt={'status':'LOCAL_AST_JSON_FIXTURE_PRECHECK_GATES_AND_CHECKPOINT_SCOPE_ONLY_PASSED',
 'actual_utc':now.isoformat(),'candidate_plan_sha256':sha(PRE/'precheck_candidate_plan_v2.json'),
 'negative_gates_rejected':rejected,'positive_gate_is_synthetic_fixture_only':True,
 'actual_execution_gate_passed':False,'actual_Torch_or_GPU_imported':False,
 'actual_raw_input_or_true_labels_read':False,'actual_gradients_checkpoint_replay_or_budget_verified':False,
 'new_scores':False,'formal100_or_outer_authorized':False}
(PRE/'local_static_gate_receipt_v2.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt))
