"""Post-preservation, one fixed selected prediction and authorized INNER labels only."""
import datetime,hashlib,io,json,shutil,zipfile
from pathlib import Path
import numpy as np
from sentiment_metrics_careflow_v1 import metrics,SEMANTICS
R=Path(__file__).parent.parent
D=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference100_complete_actual_20261006T183917Z')
P=Path('D:/CodexBackups/selective_flow_20261003_1105/plain_scalar_residual_actual_20261006T115633Z/original_package.zip')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,j:Path(p).write_text(json.dumps(j,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
joint=read(D/'complete_reference100_GPU_D_B_CPU_joint_audit.json')
assert joint['status']=='ACTUAL_REFERENCE100_GPU_D_B_COMPLETE_STATE_SOURCE_ORDER_ARRAY_CAPTURE_JOINT_PASSED'
assert joint['epochs']==100 and joint['formal_updates']==2200 and joint['best_epoch']==41 and joint['strict_fresh_full_pt_replay_error']==0
r=read(D/'a/run/out/actual_training_receipt.json');predfile=D/'a/run/out/fresh_selected_best_inner_replay.npz'
original=read(P.parent/'preservation_receipt.json')
assert sha(P)==original['package_sha256']=='481b6599c93383f25e211d673ef14b22a6ab798ca8dfd17afd0880bd6aa853da'
dest=D/'selected_INNER_five_metrics';dest.mkdir(exist_ok=False)
plan={'actual_frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':sha(__file__),'module_sha256':sha(R/'work/sentiment_metrics_careflow_v1.py'),'full_preservation_joint_sha256':sha(D/'complete_reference100_GPU_D_B_CPU_joint_audit.json'),'unique_checkpoint_sha256':r['selected_best_full']['sha256'],'prediction_file_sha256':sha(predfile),'label_scoped_original_package_sha256':sha(P),'roles':'INNER153/4 only, already used for checkpoint selection and previous same-fold teacher/scalar exploration. Descriptive selection metrics, not OUTER/DEV/Test or independent confirmation. Same checkpoint/prediction for all five, no metric-specific optimization.', 'semantics':SEMANTICS,'no_new_model_forward_or_fit':True,'FIT_OUTER_CAL_EVAL_DEV_TEST_labels_disabled':True}
write(dest/'frozen_metric_plan.json',plan)
with np.load(predfile,allow_pickle=False) as z:
 rows=z['row_ids'].copy();pred=z['prediction'].copy();state=str(z['model_state_sha256'])
assert state==joint['original_other_training_node_CPU']['states']['selected_best_full.pt']['state_sha256']
with zipfile.ZipFile(P) as z:
 # Existing original package: only the already authorized scoped INNER members.
 name=next(n for n in z.namelist() if n.endswith('scoped_inputs.npz'))
 raw=z.read(name)
 assert hashlib.sha256(raw).hexdigest()==original['members'][name]['sha256']=='c6500dd82ac908eabe92dece6e8dae8441c2809a916ffb3142a72e123271ab11'
 with np.load(io.BytesIO(raw),allow_pickle=False) as scoped:
  oldrows=scoped['inner_rows'].copy();y=scoped['inner_y'].copy();videos=scoped['inner_video'].copy()
assert rows.shape==(153,) and np.array_equal(rows,oldrows) and len(np.unique(videos))==4
vMSE=float(np.mean([np.mean((pred[videos==v].astype(np.float64)-y[videos==v])**2) for v in sorted(np.unique(videos))]))
assert abs(vMSE-r['INNER_video_equal_MSE_selection_only'])<1e-12,'Cached authorized labels do not reproduce original selection metric; do not bypass'
res={'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'frozen_plan_sha256':sha(dest/'frozen_metric_plan.json'),'scope':plan['roles'],'checkpoint_best_epoch':41,'metrics':metrics(pred,y),'video_equal_MSE_selection_only':vMSE,'original_receipt_metric_error':abs(vMSE-r['INNER_video_equal_MSE_selection_only']),'checkpoint_state_sha256':state,'label_scoped_member_sha256':hashlib.sha256(raw).hexdigest(),'new_model_forward':False,'new_other_node_CPU':False,'new_generic_performance_or_CaReFlow_superiority':False}
write(dest/'actual_metrics.json',res);np.savez_compressed(dest/'original_allowed_INNER_arrays.npz',row_ids=rows,prediction=pred,label=y,video=videos)
shutil.copyfile(__file__,dest/Path(__file__).name);shutil.copyfile(R/'work/sentiment_metrics_careflow_v1.py',dest/'sentiment_metrics_careflow_v1.py')
mf={p.name:sha(p) for p in dest.iterdir() if p.is_file()};write(dest/'manifest.json',{'actual_utc':res['actual_utc'],'members':mf,'fresh_C':shutil.disk_usage('C:/').free,'fresh_D':shutil.disk_usage('D:/').free})
with zipfile.ZipFile(dest/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in mf:z.write(dest/n,n)
 z.write(dest/'manifest.json','manifest.json')
with zipfile.ZipFile(dest/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in mf.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
write(R/'outputs/完整流100唯一best五指标INNER选择用途实际结果.json',res)
row=res['metrics'];report='完整流100唯一best五指标：仅INNER选择用途\n\n'+res['actual_utc']+'。第41轮唯一best，完成全100/2200、完整D/B原CPU/双capture保存与fresh整pt重放门后补算。\n\nAcc7 '+str(row['Acc7'])+'；Acc2 '+str(row['Acc2'])+'；F1 '+str(row['F1'])+'；MAE '+str(row['MAE'])+'；Corr '+str(row['Corr'])+'。全部153行，分类非零'+str(row['nonzero_samples'])+'行。原始视频等权MSE '+str(vMSE)+'重现原GPU选模数，误差'+str(res['original_receipt_metric_error'])+'。\n\n这里只是已用于选模的INNER153/4，且同折历史已探索；不同于官方DEV229、论文Test或未来OUTER/head评价，不能横比宣布超过CaReFlow。无新forward/拟合/异节点CPU。只读既有允许INNER标签；FIT/OUTER/CAL/EVAL/DEV/TEST标签未取。所有指标同一预测，未改变训练/选模规则。\n\nD '+str(dest)+'，完整冻结计划与原数组SHA/ZIPCRC保存。\n'
(R/'outputs/完整流100唯一best五指标INNER选择用途实际结果.md').write_text(report,encoding='utf-8')
print(json.dumps(res))
