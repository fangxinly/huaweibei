"""Synthetic objective/role verification only, never official task-label fitting."""
import datetime,hashlib,json,shutil,zipfile,tempfile
from pathlib import Path
import numpy as np
from matched_fixed_residual_heads_v1 import fit,predict,features,HeadLabels
R=Path(__file__).parent.parent;stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');local=R/'work'/('matched_head_protocol_local_'+stamp);local.mkdir(exist_ok=False)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
rng=np.random.default_rng(730101);f=rng.normal(size=120);d=rng.normal(scale=.15,size=120);y=f+.2-.1*f+.4*d+rng.normal(scale=.1,size=120);v=np.repeat(np.arange(9),[8,9,10,11,12,13,14,15,28]);m=fit(f,d,y,v);X=features(f,d,m['mean'],m['std']);maxerr=0
for name,w in [('U',m['normalized_U_weights']),('W',m['normalized_W_weights'])]:
 A=np.vstack([np.sqrt(w)[:,None]*X,np.diag([0,.1,.1])]);b=np.r_[np.sqrt(w)*(y-f),np.zeros(3)];ind=np.linalg.lstsq(A,b,rcond=None)[0];err=float(np.max(np.abs(ind-m[name])));assert err<1e-12;maxerr=max(maxerr,err)
 sameQ=(d*d-2*d*(X@m[name])- (d*d-2*d*(y-f)))**2
 assert np.max(np.abs(sameQ-4*d*d*((X@m[name])-(y-f))**2))<1e-12
const=np.full(120,.125);mc=fit(f,const,y,v);assert mc['W_exact_algebraic_duplicate'] and np.array_equal(mc['U'],mc['W']) and mc['std'][1]==0 and mc['U'][2]==mc['W'][2]==0
zero_stopped=False
try:fit(f,np.zeros(120),y,v)
except ValueError as e:zero_stopped=str(e)=='EXACT_ZERO_DELTA_CANDIDATE_W_STOP_NO_EPSILON'
assert zero_stopped
res=predict(m,f,d);assert len(res)==9
for name in ['U','W']:
 h=X@m[name];a=res[name+'_interval']-f;assert np.max(np.abs(a))<=np.max(np.abs(d))+1e-15
 disc=res[name+'_discrete']-f;assert np.all(np.minimum(np.abs(disc),np.abs(disc-d))<1e-14)
class Poison:
 def __getitem__(self,key):raise AssertionError('UNAUTHORIZED_LABEL_ACCESSED')
ids=np.arange(232);ev=np.arange(232,433);guard=HeadLabels([Poison() for _ in range(1281)],ids,ev);negative=0
for call in [lambda:guard.get_fit(ev),lambda:guard.get_fit(ids[::-1]),lambda:guard.get_eval(ev,'missing.npz','0'*64,'missing.json','0'*64)]:
 try:call()
 except (PermissionError,FileNotFoundError):negative+=1
assert negative==3 and guard.journal==[]
source=R/'work/matched_fixed_residual_heads_v1.py';shutil.copyfile(source,local/source.name);shutil.copyfile(R/'work/sentiment_metrics_careflow_v1.py',local/'sentiment_metrics_careflow_v1.py');shutil.copyfile(__file__,local/Path(__file__).name)
parent=Path('D:/CodexBackups/selective_flow_20261003_1105/fixed_head_candidate_actual_20261006T191103Z/complete_label_free_candidate_GPU_D_B_CPU_joint_audit.json');assert sha(parent)=='da44e83279048a7e615ca6ee5cc2cd20e3d251a054c5812f79a857972154aed0'
plan={'status':'LOCAL_MATCHED_HEAD_CORE_AND_PROTOCOL_DRAFT_NOT_EXECUTION_FREEZE','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'parent_actual_candidate_joint_sha256':sha(parent),'source_sha256':sha(source),
 'shared_information':'Z=(pF,delta); target r=y-pF; common 3 slots intercept/standardizedpF/standardizeddelta; no extra inputs',
 'FIT_only_statistics':'Both heads use identical headFIT232 nine-video-equal mean/std; population variance, exact constant feature structurally0/coefficient0; no threshold epsilon. Statistics never refit with W or201.',
 'objectives':'U sum(a*(h-r)^2)+.01*(b^2+c^2); W sum(a*4delta²/global_c*(h-r)^2)+same ridge. a=1/(9*n_v),global_c=sum(a*4delta²). Intercept unpenalized, weights sum1; factor4 cancels only after this normalization.',
 'baseline_and_readouts':['F','C','constant_free','U_free','U_interval','U_discrete','W_free','W_interval','W_discrete'],
 'readout':'free=pF+h; interval=pF+clip(h,min(0,delta),max(0,delta)); discrete acceptsC only when delta²-2delta*h<0, tieF. Fixed no amplitude scan.',
 'label_roles':'Only232 headFIT labels after complete protocol+source freeze and original candidate D/CPU audit; fit heads/constant once. Whole head state and stats SHA frozen before201 input inference. All9 outputs201 finite/row/order/own-headSHA saved and SHA verified before one201 label read. No model/head updates after evaluation.',
 'metrics':'All9 outputs reported together with pooled Acc7/Acc2/weightedF1/MAE/Pearson, pooledMSE and videoequalMSE plus paired9 per-video differences. Allfive must pointwisebeat matchedF and simpleconstant to motivate further method claims; never use201 to choose one of9 for publication. This is development, not independent confirmation or officialCaReFlowbenchmark.',
 'stop':'Engineering/source/role/nonfinite/replay mismatches invalidate run; exactdelta0 W/selection stop no epsilon. exactabsdelta constant U/W identical. No newESS/coverage/concentration/condition threshold; report allpredeclaredcomparisons. No expansion if matched weighted or constrained outputs fail simplefree/constant; new features need separate protocol.',
 'still_missing_before_actual_execution':['source-pinned original-label extraction/runtime chronology and receipt wrapper','201 input-only collector with newrole gate and sameF-C-C-F/dummy replay','complete original arrays/head state/prediction hashes and one-label evaluation runner','independentCPU original-objective/replay/metric audit and capture newroots/fullargv','fresh human-backed assets/lease/source/space/budget evidence and D+othernodeCPU saving'],
 'actual_official_labels_or_GPU_or_head_fit_or201_prediction_or_scores':False,'no_new_remote_or_othernodeCPU_claim':True}
write(local/'protocol_candidate_plan.json',plan);write(local/'synthetic_core_verification.json',{'actual_utc':plan['actual_utc'],'independent_augmented_lstsq_vs_normal_equation_maxerr':maxerr,'structuredQ_weighted_identity_pass':True,'constant_absdelta_UW_exact_duplicate':True,'constant_feature_std_and_coefficient_exact0':True,'zero_delta_stopped':zero_stopped,'poison_label_negative_roles':negative,'fixture_only_not_task_data':True})
D=Path('D:/CodexBackups/selective_flow_20261003_1105')/('matched_head_protocol_local_'+stamp);D.mkdir(exist_ok=False)
mf={p.name:sha(p) for p in local.iterdir()};write(local/'manifest.json',{'actual_utc':plan['actual_utc'],'members':mf,'fresh_C_D':{x:shutil.disk_usage(x+':/').free for x in ['C','D']}})
with zipfile.ZipFile(local/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in mf:z.write(local/n,n)
 z.write(local/'manifest.json','manifest.json')
with zipfile.ZipFile(local/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in mf.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
for p in local.iterdir():shutil.copyfile(p,D/p.name)
record={'status':plan['status'],'actual_utc':plan['actual_utc'],'local':str(local),'D':str(D),'source_SHA':sha(source),'plan_SHA':sha(local/'protocol_candidate_plan.json'),'ZIP_SHA':sha(D/'records.zip'),'independent_synthetic_max_error':maxerr,'actual_task_fit_or_GPU_or201_evaluation':False,'next_required':plan['still_missing_before_actual_execution']}
write(R/'outputs/完整流匹配U_W头协议本地准备接续.json',record)
(R/'outputs/完整流匹配U_W头协议本地准备接续.md').write_text('匹配U/W闭式核心与标签访问协议仅本地准备\n\n'+json.dumps(record,ensure_ascii=False,indent=2)+'\n\n共同FIT-only视频等权标准化，两目标归一后同岭.01。独立增广最小二乘与正规方程、结构化Q平方权重恒等式、恒定absdelta/零feature、越权毒化门通过，仅合成。正式运行/完整201守卫/自然退出/原CPU与捕获保存仍待实现和冻结，禁止据此读取新标签或称任务成绩。\n',encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
