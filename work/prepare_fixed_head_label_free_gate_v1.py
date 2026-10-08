import ast,datetime,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from fixed_reference_head_label_free_gate_v1 import allowed_head_inputs,candidate_gate
R=Path(__file__).parent.parent
source=R/'work/fixed_reference_head_label_free_gate_v1.py'
bundle=R/'work/minimal_fixed_staged_reference_v1_20261006T151117Z'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
local=R/'work'/('fixed_reference_head_label_free_gate_'+stamp);local.mkdir(exist_ok=False)
fit=np.load(bundle/'head_development_fit_0.npy',allow_pickle=False)
evaluation=np.load(bundle/'head_development_eval_0.npy',allow_pickle=False)
mapping=read(bundle/'train_row_video_mapping.json');video=np.asarray([x['video_id'] for x in mapping])
reserve=read(bundle/'head_role_reservation_plan.json')
assert sha(bundle/'head_development_fit_0.npy')==reserve['head_development_fit']['array_sha256']
assert sha(bundle/'head_development_eval_0.npy')==reserve['head_development_eval']['array_sha256']
assert set(fit.tolist()).isdisjoint(evaluation.tolist()) and len(fit)==232
joint=read(R/'outputs/完整流单参考100完成与完整保存实际结果.json')
assert joint['formal_updates']==2200 and joint['original_CPU_natural_exit']['exit_code']==0
plan={'actual_frozen_utc':now.isoformat(),'status':'LOCAL_LABEL_FREE_GATE_PREPARATION_NOT_EXECUTION_OR_FULL_HEAD_PROTOCOL',
 'source_sha256':sha(source),'reference_joint_sha256':sha(R/'outputs/完整流单参考100完成与完整保存实际结果.json'),
 'reference_checkpoint_sha256':joint['original_D_full_files']['selected_best_full.pt']['sha256'],
 'reference_state_sha256':joint['original_other_training_node_CPU']['states']['selected_best_full.pt']['state_sha256'],
 'role_reservation_sha256':sha(bundle/'head_role_reservation_plan.json'),
 'head_fit_sha256':sha(bundle/'head_development_fit_0.npy'),'head_eval_sha256':sha(bundle/'head_development_eval_0.npy'),
 'candidate':'One fixed condition: zero all six donor feedback inputs to the one fixed context update, matching the existing predeclared four-FIT mechanism. No channel/scale search; pC-pF, whole original weight unchanged, eval and dummy labels only.',
 'collection_scope':'232 head-FIT original inputs only. First stage forbids actual labels from every role and forbids201EVAL input inference; public constructor FIT labels are historical referenceFIT only, never head supervision.',
 'Z':'[pF, delta] with constant candidate identity and condition; all selection/scale flags fixed. No candidate picking, uncertainty threshold or teacher output. Capacity candidate h=a+b*standardized(pF)+c*standardized(delta).',
 'fit_later':'Not implemented: U/W same three-parameter closed-form ridge .01 on slopes; feature means/std from headFIT only using video equal weights. Normalize W=video_equal*delta^2 globally to total1; zero constant feature uses slope0, never epsilon to rescue candidate. True task residual r=y-pF; loss and fitting only after separate label-access protocol frozen and collection preserved.',
 'predeclared_empirical_W_stop':'Exact delta_squared_mass0 OR fewer than3 positive-mass videos OR effective mass ESS<3. Cost feasibility only, chosen before any new actual collection; not statistical independence or safety. Record complete spectrum; never lower gate or reweight by EVAL.',
 'readouts_later':'No labels or fitting now. Same h free, interval clip to [min(0,delta),max(0,delta)], discrete {0,delta} minimizing Qhat with tie F. Include F and a headFIT learned constant. All reported, none picked on201.',
 'five_metrics':'Frozen Acc7/Acc2/weightedF1/MAE/Pearson all same prediction, along with paired video equal squared-risk effects. Zero-label boundary diagnostics are counts, not accuracy/gain. No amplitude rescue if no boundary crosses.',
 'budgets_candidate_not_actual':'Future GPU collection20min/6GiB cumulativepeak/remote4GiB-D6GiB and >=2h preservation reserve; fresh5min UUID/fullargv/emptycompute/assets/source/space/current actual completed reference identity. Existing whole checkpoint D+othernode CPU pin, no new weights or optimizer updates; fresh original receipts/array CPU audit/new-root capture required before any head label fitting.',
 'actual_GPU_or_labels_or_scores_or_head_fit':False,'full_runtime_and_label_access_and_CPU_auditor_implemented':False,'review14_complete':False}
write(local/'local_gate_plan.json',plan)
ast.parse(source.read_text(encoding='utf-8'))
checks=[]
class Poison:
 def __getitem__(self,k):
  if k==1:raise AssertionError('LABEL_ACCESS_FORBIDDEN')
  return ('raw_inputs',) if k==0 else 'original_metadata'
examples=[Poison() for _ in range(1281)]
records=allowed_head_inputs(examples,fit,fit,video)
assert len(records)==232 and all(np.all(x[1]==0) for x in records);checks.append('poison_labels_never_indexed')
def denied(name,fn):
 try:fn()
 except (ValueError,PermissionError):checks.append(name);return
 raise AssertionError('NEGATIVE_CASE_NOT_DENIED: '+name)
denied('eval_row_input_denied',lambda:allowed_head_inputs(examples,evaluation[:1],fit,video))
denied('duplicate_row_denied',lambda:allowed_head_inputs(examples,np.repeat(fit[:1],2),fit,video))
denied('floating_row_denied',lambda:allowed_head_inputs(examples,fit.astype(float),fit,video))
v=video[fit];f=np.linspace(-2.7,2.7,232,dtype=np.float32);c=f+np.float32(.2)
positive=candidate_gate(f,c,fit,v,f.copy());assert positive['W_empirical_feasibility_pass'];checks.append('synthetic_nine_video_positive_not_actual_gate')
zero=candidate_gate(f,f.copy(),fit,v,f.copy());assert not zero['W_empirical_feasibility_pass'] and zero['W_effective_video_mass'] is None;checks.append('exact_zero_no_epsilon_or_nan')
c1=f.copy();c1[v==v[0]]+=np.float32(.2)
one=candidate_gate(f,c1,fit,v,f.copy());assert not one['W_empirical_feasibility_pass'] and one['W_effective_video_mass']==1;checks.append('one_video_W_gate_denied')
denied('nonfinite_scalar_denied',lambda:candidate_gate(f,np.full(232,np.nan,np.float32),fit,v,f.copy()))
denied('float64_scalar_denied',lambda:candidate_gate(f.astype(np.float64),c,fit,v,f.copy()))
denied('restore_drift_denied',lambda:candidate_gate(f,c,fit,v,f+np.float32(.001)))
receipt={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'AST_AND_POISON_ROLE_SYNTHETIC_NUMPY_GATE_FIXTURES_PASSED_NOT_GPU','checks':checks,'fixture_count':len(checks),'synthetic_positive':positive,'synthetic_zero':zero,'synthetic_one_video':one,'task_data_arrays_or_labels_read':False,'real_role_metadata_read':True,'Torch_imported':False,'GPU_gate_passed':False,'new_prediction_or_performance':False}
write(local/'local_fixture_receipt.json',receipt)
for p in [source,Path(__file__),bundle/'head_development_fit_0.npy',bundle/'head_development_eval_0.npy',bundle/'head_role_reservation_plan.json']:shutil.copyfile(p,local/p.name)
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/('fixed_reference_head_label_free_gate_local_'+stamp);D.mkdir(exist_ok=False)
for p in local.iterdir():shutil.copyfile(p,D/p.name)
mf={p.name:sha(p) for p in D.iterdir()};write(D/'manifest.json',{'actual_utc':receipt['actual_utc'],'members':mf,'fresh_C_D':{k:shutil.disk_usage(k+':/').free for k in ['C','D']}})
with zipfile.ZipFile(D/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in mf:z.write(D/n,n)
 z.write(D/'manifest.json','manifest.json')
with zipfile.ZipFile(D/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in mf.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
state={'actual_utc':receipt['actual_utc'],'scope':plan['status'],'local':str(local),'D':str(D),'plan_sha256':sha(local/'local_gate_plan.json'),'source_sha256':sha(source),'ZIP_SHA':sha(D/'records.zip'),'actual_labels_GPU_collection_head_fit':False,'fixture_count':len(checks),'next':'Complete fresh checkpoint-only collector, independent CPU array/capture audit, runtime guards and stage1 original-input dummy-label replay before any actual head labels or201 inference. Review14 pending, no dependency on waiting.'}
write(R/'outputs/完整流单候选零标签门本地准备接续.json',state)
(R/'outputs/完整流单候选零标签门本地准备接续.md').write_text('单候选零标签门本地准备\n\n'+json.dumps(state,ensure_ascii=False,indent=2)+'\n\n真实role元数据与10个合成NumPy/毒化标签fixture通过，仅本地准备。未导入Torch/建立模型/访问实际头标签/真实232预测或201输入，不冒GPU门已过。一个固定六供体同时关闭条件匹配原四FIT机制，不挑通道或幅度。W有效视频mass>=3仅事先经验成本门，不是风险/泛化保证；保持原单F和未来同信息U/W/读出匹配。完整runtime/拟合访问协议未实施，原完成100科学源及所有旧原件不变。\n',encoding='utf-8')
l=read(R/'outputs/研究建议交流接续.json');l['next_label_free_candidate_preparation']=state;write(R/'outputs/研究建议交流接续.json',l)
print(json.dumps(state))
