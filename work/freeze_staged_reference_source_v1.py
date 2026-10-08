"""Freeze protocol, source, matching boundaries and real prior evidence on D."""
import ast, datetime, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs';WORK=ROOT/'work'
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105/second_lease_activation_checkpoint_actual_20261006T144615Z')
PARENT=WORK/'second_lease_activation_checkpoint_v6_20261006T144158Z'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
assert shutil.disk_usage('D:/').free>12*1024**3
bundle=WORK/('minimal_fixed_staged_reference_v1_'+stamp);bundle.mkdir(exist_ok=False)
runtime=read(PARENT/'runtime_candidate_plan.json')
files=set(runtime['source_sha256'])|set(runtime['role_file_sha256'])|{'runtime_candidate_plan.json','donor_terminal_mechanism_precheck_v1.py'}
for n in files:shutil.copy2(PARENT/n,bundle/n)
added=('minimal_fixed_staged_training_v3.py','resume_next_update_check_v2.py','staged_reference_wrapper_v1.py','launch_staged_reference_v1.py','audit_staged_reference_CPU_v1.py')
for n in added:shutil.copy2(WORK/n,bundle/n)
shutil.copy2(BASE/'complete_GPU_D_B_CPU_joint_audit.json',bundle/'original_complete_GPU_D_B_CPU_joint_audit.json')
roles=WORK/'future_fold0_head_role_reservation_20261006T140048Z'
for p in roles.iterdir():shutil.copy2(p,bundle/p.name)
raw=read(BASE/'a/original_small_files/run/out/actual_precheck_receipt.json')
orders=np.load(bundle/'fit_orders_seed91819_100.npy',allow_pickle=False);fit=np.load(bundle/'fit_0.npy',allow_pickle=False)
assert orders.shape==(100,695) and all(np.array_equal(np.sort(r),np.sort(fit)) for r in orders)
max32=max(x['seconds'] for x in raw['step_times'])
projection=100*(21*3*max32+3*raw['tail_forward_backward_seconds']+raw['final_budget']['elapsed_seconds'])+600
assert projection<18000
plan={'format':'MINIMAL_FIXED_STAGED_REFERENCE_PLAN_V1','actual_frozen_utc':now.isoformat(),
 'question':'One video-isolated full-chain fixed reference trainability and replay; no donor superiority comparison.',
 'phase_schedule':['gated_reference_first10_220steps','only_after_actual_D_B_CPU_and_resume_fresh_replay_continue11_to100'],
 'fold':0,'seed':91819,'epochs_total':100,'updates_per_epoch':22,'total_updates':2200,'batch_size':32,'tail_rows':23,
 'runtime_plan_sha256':sha(bundle/'runtime_candidate_plan.json'),
 'source_sha256':{p.name:sha(p) for p in bundle.glob('*.py')},
 'role_order_sha256':{n:sha(bundle/n) for n in runtime['role_file_sha256']},
 'actual_precheck_root':'/data/coding/minimal_fixed_fold0_activation_checkpoint_v6_actual_20261006T144244Z',
 'actual_precheck_receipt_sha256':sha(BASE/'a/original_small_files/run/out/actual_precheck_receipt.json'),
 'actual_precheck_exit_sha256':sha(BASE/'a/original_small_files/run/natural_exit.json'),
 'original_complete_joint_audit_sha256':sha(bundle/'original_complete_GPU_D_B_CPU_joint_audit.json'),
 'clean_initial_full_sha256':raw['initial_full']['file_sha256'],'clean_initial_state_sha256':raw['initial_full']['metadata']['state_sha256'],
 'fit_statistics_sha256':raw['initial_full']['metadata']['fit_statistics_sha256'],
 'clean_rng_sha256':{n:sha(BASE/'a/original_small_files/run/out'/n) for n in ('clean_initial_torch_rng.pt','clean_initial_python_numpy_rng.json')},
 'initial_restore':'All 374 parameters/buffers plus torch CPU/CUDA/Python/NumPy captured before precheck steps, not reseed or post-precheck state.',
 'optimizer':'Original source pinned author/control AdamW groups lr1e-5 warmup0.1 over2200 updates; optimizer then scheduler every actual batch including23 tail.',
 'precision':'Source/default FP32, no autocast or GradScaler; original author backend deterministic seed function. Log actual backend state.',
 'task_parameters':185402807,'parameter_tensors':364,'full_state_tensors':374,
 'matching_protocol':{'shared_first10_order_content_sha256':hashlib.sha256(orders[:10].tobytes()).hexdigest(),
  'exact_clean_initial_and_model_source_pinned':True,'actual_comparison_control_trained':False,
  'future_training_donor_ablation_only_if_separate_question':'Same constructed modules/initial state/roles/statistics/all orders/objective weights/optimizer/selection. Disable donor contribution; allow honest unused donor gradients, no dummy loss. Random stream strategy must separately freeze and verify.',
  'no_old_C2_plain_teacher_or_old_task_weight_initialization':True},
 'shared_first10_order_content_sha256':hashlib.sha256(orders[:10].tobytes()).hexdigest(),
 'selection':'Every epoch exactly original INNER153/4 video equal float64 MSE, strictly less selects new best, exact ties retain earliest. No hyperparameter or seed/loss changes at10.',
 'reference_roles':{'FIT':[695,30],'INNER':[153,4],'OUTER':[433,18]},
 'future_head_roles':{'role_reservation_plan_sha256':sha(bundle/'head_role_reservation_plan.json'),'head_development_FIT':[232,9],'head_development_EVAL':[201,9],
  'OUTER_disabled_through_training_and_this_runner':True,'head_training_not_authorized_by_this_plan':True,
  'legal_next_minimum_proposal':'After100 earliest best full replay/zero-label real candidate nondegeneracy, separate fixed protocol U/W affine h(1,pF,delta), same 3 parameters ridge.01 slopes only, FIT-only feature stats and globalW normalization; fixed free/interval/discrete reads pluszero/all-accept/constant. No EVAL selection or oldT0 c transfer.',
  'stop_before_head':'No actual delta output changes => stop candidate/W, no epsilon weights or direction/amplitude sweep.',
  'past_allTRAIN_exploration_not_fresh_confirmation_or_wholecrossfit':True},
 'mechanism_fit_rows':runtime['precheck_row_candidates']['normal'][:4],
 'mechanism':'One same four-row zero-label zero-donor check at predeclared stage10/final selected checkpoint;<=120s no sweep/extra formal steps/score claim.',
 'resume_check':'At saved220, continuous and fresh-disk-restored same next FIT batch each isolated one update. Exact complete model/Adam/scheduler/ending RNG digest and objective. Restore220; diagnostic2 steps never count formal221 or select.<=240s.',
 'conservative_100_projection_seconds':projection,'projection_method':'100*(21*3*max precheck normal/mixed seconds+3*backward-only tail seconds+entire76.5s precheck as INNER upper bound)+600. Update using full real epoch INCLUDINGtail optimizer and INNER, not just two-step minimum.',
 'actual_phase10_seconds_limit':1800,'actual_continue100_seconds_limit':18000,'whole_cumulative_GPU_bytes_limit':6*1024**3,'saving_reserve_seconds':7200,
 'space_gate_bytes':{'remote_free_before_start':12*1024**3,'D_free_before_start':12*1024**3,'remote_min_during_execution':4*1024**3},
 'space_ledger':'Keep existing initial/post-precheck originals; stage10 resume includesmodel+2Adam+best (~2.97GB) plusselectedbest~.742GB. Final100 another~3.71GB, new smallZIP/receipt/RNG and temp margin. Maxnew<10GiB each A,D,B. No all100 full weight snapshots; bestCPU RAM cloned, immutable boundary files. No old deletion.',
 'next_update_GPUBudget_included':True,'fresh_instance_strict_replay_required':True,
 'assets_must_be_actual_verified_before_launch':True,'actual_stage10_executed':False,'actual100_completed':False,'new_performance_scores':False}
(bundle/'staged_reference_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
for p in bundle.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
# Load pure validate without importing Torch or NumPy external environment.
s=(bundle/'minimal_fixed_staged_training_v3.py').read_text();tree=ast.parse(s)
allowed={n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.Import,ast.ImportFrom)) and (not isinstance(n,ast.FunctionDef) or n.name in ('validate','timestamp'))}
pure=ast.Module(body=[n for n in tree.body if n in allowed],type_ignores=[]);env={};exec(compile(pure,'source_validate_extract','exec'),env)
pp=dict(plan,_sha256=sha(bundle/'staged_reference_plan.json'))
example={'scope':'MINIMAL_FIXED_STAGED_REFERENCE_V1','human_provenance_verified':True,'lease_source':'DIRECT_HUMAN_NEW_P4_24H_20261006','actual_query_utc':now.isoformat(),'gpu_uuid':'GPU-53696803-875e-eec8-2231-29db63579891','compute_processes':[],'python_full_argv':[],'complete_assets_and_source_verified':True,'complete_GPU_D_B_precheck_verified':True,'remote_free_bytes':20*1024**3,'permanent_D_free_bytes':20*1024**3,'plan_sha256':pp['_sha256'],'lease_end_utc':(now+datetime.timedelta(hours=20)).isoformat()}
env['validate'](pp,example,now,'stage10');negative=0
for key,value in [('scope','wrong'),('human_provenance_verified',False),('actual_query_utc',(now-datetime.timedelta(minutes=6)).isoformat()),('gpu_uuid','old'),('compute_processes',[{'pid':1}]),('python_full_argv',None),('complete_GPU_D_B_precheck_verified',False),('remote_free_bytes',0),('permanent_D_free_bytes',0),('plan_sha256','0'*64),('lease_end_utc',(now+datetime.timedelta(hours=1)).isoformat())]:
 e=dict(example);e[key]=value
 try:env['validate'](pp,e,now,'stage10')
 except PermissionError:negative+=1
 else:raise AssertionError('NEGATIVE_GATE '+key)
try:env['validate'](pp,example,now,'continue100')
except PermissionError:negative+=1
else:raise AssertionError('CONTINUE_BEFORE_CPU')
receipt={'actual_frozen_utc':now.isoformat(),'source_only_AST_and_12_negative_gate_passed':True,'negative_cases':negative,'synthetic_positive_not_GPU_pass':True,'Torch_imported_or_model_constructed':False,'actual_stage10_started':False,'formal100_complete':False,'bundle':str(bundle),'plan_sha256':sha(bundle/'staged_reference_plan.json'),'source_sha256':plan['source_sha256'],'conservative_projection_seconds':projection}
(bundle/'local_freeze_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
pkg=bundle.with_suffix('.zip')
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for p in bundle.iterdir():z.write(p,p.name)
with zipfile.ZipFile(pkg) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for p in bundle.iterdir():assert hashlib.sha256(z.read(p.name)).hexdigest()==sha(p)
receipt['package_sha256']=sha(pkg);receipt['package']=str(pkg)
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('staged_reference_source_preparation_'+stamp);dest.mkdir(exist_ok=False)
shutil.copy2(pkg,dest/pkg.name);shutil.copytree(bundle,dest/'source')
for n in ('freeze_staged_reference_source_v1.py','minimal_fixed_staged_training_v1.py','minimal_fixed_staged_training_v2.py','prepare_minimal_fixed_staged_v2.py','prepare_minimal_fixed_staged_v3.py','resume_next_update_check_v1.py','finalize_second_lease_activation_checkpoint_actual_v1.py','finalize_second_lease_activation_checkpoint_actual_v2.py'):
 shutil.copy2(WORK/n,dest/n)
(dest/'preservation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
(OUT/'完整流单参考分段训练源冻结最新.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
