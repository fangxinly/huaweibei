import ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from fixed_reference_head_label_free_gate_v2 import allowed_head_inputs,candidate_gate
R=Path(__file__).parent.parent
old=R/'work/minimal_fixed_staged_reference_v1_20261006T151117Z'
refD=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference100_complete_actual_20261006T183917Z')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
joint=read(refD/'complete_reference100_GPU_D_B_CPU_joint_audit.json');original=read(refD/'a/run/out/actual_training_receipt.json')
assert joint['formal_updates']==2200 and joint['original_CPU_natural_exit']['exit_code']==0
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
bundle=R/'work'/('fixed_head_candidate_collection_v1_'+stamp);bundle.mkdir(exist_ok=False)
for p in old.iterdir():
 if p.is_file():shutil.copyfile(p,bundle/p.name)
names=['fixed_reference_head_label_free_gate_v2.py','fixed_head_input_collector_v1.py','fixed_head_collection_wrapper_v1.py','audit_fixed_head_candidate_CPU_v1.py','capture_fixed_head_candidate_v1.py','run_staged_CPU_original_wrapper_v1.py']
for n in names:shutil.copyfile(R/'work'/n,bundle/n)
checks=[]
for n in names:ast.parse((bundle/n).read_text(encoding='utf-8'));checks.append('AST:'+n)
rows=np.load(bundle/'head_development_fit_0.npy',allow_pickle=False);evalrows=np.load(bundle/'head_development_eval_0.npy',allow_pickle=False)
mapping=read(bundle/'train_row_video_mapping.json');allv=np.asarray([x['video_id'] for x in mapping]);v=allv[rows]
class Poison:
 def __getitem__(self,k):
  if k==1:raise AssertionError('TASK_LABEL_ACCESS')
  return ('raw_input',) if k==0 else 'metadata'
examples=[Poison() for _ in range(1281)]
assert len(allowed_head_inputs(examples,rows,rows,allv))==232;checks.append('POISON_LABELS_NEVER_ACCESSED')
try:allowed_head_inputs(examples,evalrows[:1],rows,allv)
except PermissionError:checks.append('201_INPUT_ROLE_DENIED')
else:raise AssertionError('EVAL_INPUT_NOT_DENIED')
f=np.linspace(-2.7,2.7,232,dtype=np.float32)
zero=candidate_gate(f,f.copy(),rows,v,f.copy());assert not zero['W_empirical_feasibility_pass'];checks.append('ZERO_W_STOP')
c=f.copy();c[v==v[0]]+=np.float32(.2)
one=candidate_gate(f,c,rows,v,f.copy());assert one['W_empirical_feasibility_pass'] and one['W_effective_video_mass']==1;checks.append('ESS_ONE_DIAGNOSTIC_DOES_NOT_CANCEL_CHEAP_FIT')
for name,args in [('NONFINITE',[f,np.full(232,np.nan,np.float32),rows,v,f.copy()]),('RESTORE_DRIFT',[f,f.copy(),rows,v,f+.01]),('WRONG_PRECISION',[f.astype(np.float64),f.copy(),rows,v,f.copy()])]:
 try:candidate_gate(*args)
 except ValueError:checks.append(name+'_DENIED')
 else:raise AssertionError(name)
new_sources={p.name:sha(p) for p in bundle.iterdir() if p.is_file()}
plan={'status':'FROZEN_HEADFIT232_ONE_CANDIDATE_ZERO_LABEL_COLLECTION_V1_NOT_HEAD_FIT','actual_frozen_utc':now.isoformat(),
 'source_and_role_sha256':new_sources,'original_runtime_plan_sha256':sha(old/'runtime_candidate_plan.json'),
 'reference_joint_sha256':sha(refD/'complete_reference100_GPU_D_B_CPU_joint_audit.json'),'checkpoint_file_sha256':original['selected_best_full']['sha256'],
 'checkpoint_state_sha256':joint['original_other_training_node_CPU']['states']['selected_best_full.pt']['state_sha256'],'FIT_statistics_sha256':original['fit_statistics_sha256'],
 'candidate_semantics':'Single fixed first-context intervention replaces all six incoming feedback vectors by exact zeros_like; original fixed_context formula and second Euler recomputed; no module/mode/input mask/channel/scale search. Temporary override finally removed.',
 'roles':'Original full100 referenceFIT695/INNER153 unchanged. Only original232 headFIT rows/9video inputs are permitted in this collector. All task labels blocked by constructor guard override and new raw-input access guard; only dummy0/7.201 headEVAL input/label and all development/TEST/CAL labels prohibited at this phase.',
 'constructor_override':'Public pinned constructor original guard class replaced locally by subclass whose fit_supervision delegates to original input-only/dummy guard. No actual FIT/INNER/head task labels indexed. No optimizer update. Whole unique best41 restored strict; original source and completed weights never edited. Trusted pickle deserialization contains unused labels in memory; row-wise task label entries are never indexed.',
 'collection':'One GPU instance; fixed32 batches [32]*7+[8] in frozen row order, perbatch F-C-C-F, source FP32. First32 F/C dummy0->7 checks exact0. No reseed/augment/source patch or dynamic precision/epsilon. Both paths repeat <=1e-6 independently, first Euler state same, one first-context six-direction intervention and two reader calls eachforward. Fullmodel/buffer/FITstats/RNG unchanged.',
 'arrays':'Original FP32 pF,pC,pC_repeat,pF_restored plus FP64 delta/Wraw4delta^2; original row/segment/video IDs, per-row repeat errors and fixed class endpoints. Schema contains no labels. Complete source/plan/parameter/input-role/array SHA plus actualPID fullargv naturalexit and cumulativeGPU/time budget.',
 'hard_gate':'Engineering mismatch/nonfinite/role/semantic/persistent-state or replay failure invalidates collection and prohibits head fitting; exact delta0 stops candidate selection/W without epsilon. ESS/concentration/coverage/near-collinearity are diagnostics only, no empirical minimum gate.',
 'v1_local_preparation_ESS3_rule':'Superseded before any actual head prediction or task label collection: independent reasoning accepts reviewer14 objection because two3parameter closed-form fits are trivial cost. v1 source-only original/D kept, no observed data used to lower a gate.',
 'later_matched_class':'Z=(pF,delta) same chosen function-class information, not full original x. h=a+b*standardizedpF+c*standardizeddelta. U videoequal residual r=y-pF; W proportional4delta^2 with global videoequal normalization, ridge .01 slopes. Exact zero feature variance standardized0/coefficient0; exactly constant absdelta nonzero makes U/W algebraically redundant, not approximate merge.',
 'later_readouts_and_comparators':'After this actual collection completeD+othernode CPU/capture and separate label-access/fit/evaluation freeze: F, wholeacceptC, headFIT videoequal residual constant, U/W each free/interval/discrete with tieF. All5metrics plus paired videoequalMSE reported; no201 head/reader/lambda/seed/feature choice. No head fitting or201 inference in current source.',
 'budget':{'GPU_seconds':1200,'cumulative_peak_bytes':6*1024**3,'remote_free_bytes':4*1024**3,'D_free_bytes':6*1024**3,'save_reserve_seconds':7200,'fresh_query_max_seconds':300},
 'human_asset_source':'Original human secondlease threeP4 credentials already authorized; lease24h approxOct7UTC13:35, conservative13:30 not platform verified. Fresh identity/compute/fullargv/source/assets/space still required before launch; JSON alone does not establish human provenance.',
 'actual_runtime_passed':False,'actual_head_inputs_predictions_or_task_labels_read':False,'actual_CPU_or_new_capture_passed':False,'actual_fitted_heads_or_performance':False}
write(bundle/'candidate_collection_plan.json',plan)
write(bundle/'local_AST_and_role_fixture_receipt.json',{'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checks':checks,'fixture_scope':'synthetic NumPy/poison role only, not runtime or real data predictions','Torch_imported':False,'actual_GPU':False,'exact_ESS_one_case_expected_feasible':True})
shutil.copyfile(refD/'complete_reference100_GPU_D_B_CPU_joint_audit.json',bundle/'reference100_completion_joint.json')
pkg=bundle.with_suffix('.zip')
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for p in bundle.iterdir():z.write(p,p.name)
with zipfile.ZipFile(pkg) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/('fixed_head_candidate_source_freeze_'+stamp);D.mkdir(exist_ok=False)
shutil.copyfile(pkg,D/'original_source_bundle.zip');shutil.copyfile(bundle/'candidate_collection_plan.json',D/'candidate_collection_plan.json')
receipt={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'bundle':str(bundle),'bundle_zip':str(pkg),'bundle_zip_sha256':sha(pkg),'plan_sha256':sha(bundle/'candidate_collection_plan.json'),'members':len(list(bundle.iterdir())),'D':str(D),'fresh_C_D':{k:shutil.disk_usage(k+':/').free for k in ['C','D']},'status':'SOURCE_ROLE_FIXTURES_FROZEN_NOT_ACTUAL_GPU_GATE','new_GPU_or_head_fit_or_performance':False}
write(D/'source_preservation_receipt.json',receipt);write(R/'outputs/完整流单候选232零标签采集源冻结最新.json',receipt)
print(json.dumps(receipt))
