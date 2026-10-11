import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
PRE=ROOT/'work/minimal_fixed_precheck_local_20261006T1240Z'
RUNTIME=ROOT/'work/minimal_fixed_runtime_v1_local_20261006T1228Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.loads((RUNTIME/'runtime_candidate_plan.json').read_text(encoding='utf-8'))
for name in {**plan['source_sha256'],**plan['role_file_sha256']}:
 shutil.copyfile(RUNTIME/name,PRE/name);assert sha(RUNTIME/name)==sha(PRE/name)
shutil.copyfile(RUNTIME/'runtime_candidate_plan.json',PRE/'runtime_candidate_plan.json')
preplan={'status':'LOCAL_UNEXECUTED_GPU_PRECHECK_SOURCE_CANDIDATE_NOT_AUTHORIZATION',
 'created_actual_utc':datetime.now(timezone.utc).isoformat(),'runtime_plan_sha256':sha(PRE/'runtime_candidate_plan.json'),
 'source_sha256':{p.name:sha(p) for p in sorted(PRE.glob('*.py'))},
 'scope':'New full-state fixed reference constructor/two FIT steps/tail no step/INNER dummy-label disk replay only; no formal100 or OUTER.',
 'actual_GPU_execution_allowed':False,'actual_execution_gate_passed':False,'new_scores':False,
 'actual_Torch_or_CUDA_imported':False,'actual_gradient_or_replay_verified':False,
 'gate_proposal':'Trusted human/provider provenance outside JSON, confirmed lease >20min precheck+2h save reserve, fresh <=5min UUID/emptycompute/fullargv/source/assets/completion/space; remote >=4GiB and D>=6GiB. Current estimate already passed and no confirmed future lease.',
 'budget_proposal':{'torch_allocator_peak_max_bytes':6*1024**3,'precheck_max_seconds':1200,'saving_reserve_seconds':7200},
 'objective':'MSE+.02FM+.01cycle+.05unimodal+.01variance; no new counterfactual decoder branch.',
 'optimizer_updates':2,'tail_optimizer_updates':0,'original_INNER_rows':153,
 'full_state_initial':'clean_initial_full.pt before any updates, Torch/CUDA plus Python/NumPy RNG saved separately; not inherited by formal runner.',
 'full_state_replay':'after_two_steps_full.pt every parameter/buffer; own file SHA+tensor SHA+source/fold/seed/scope/steps+exact FIT buffers then strict fresh-instance load and original INNER <=1e-6.',
 'gradient_probes':'Actual primary and separate weighted auxiliary graph probes; backward field None for primary/nonNone for auxiliary; task decoder nonNone for primary/None for auxiliary. Final aggregate all retained finite nonNone gradients, zeros reported honestly.',
 'limitations':['No actual asset/runtime/GPU/gradient/mask/time/budget/reload evidence.','No model input HVP or full encoder second derivative implemented; control solver needs separate actual audit.','No process-wrapper natural-exit source, independent CPU checkpoint/array auditor, capture coverage, actual D/other-node preservation, formal100 or OUTER completion authorization yet.','JSON field validation is not provenance verification or tool/human permission.']}
(PRE/'precheck_candidate_plan.json').write_text(json.dumps(preplan,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':preplan['status'],'plan_sha256':sha(PRE/'precheck_candidate_plan.json'),'source_count':len(preplan['source_sha256']),'new_GPU':False,'new_scores':False}))
