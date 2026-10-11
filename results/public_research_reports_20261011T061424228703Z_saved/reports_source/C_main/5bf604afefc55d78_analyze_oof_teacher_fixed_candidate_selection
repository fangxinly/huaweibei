"""CPU frozen-path selector diagnostic; no optimization or new message generation."""
from pathlib import Path
import datetime,json,hashlib,numpy as np
base=Path(__file__).resolve().parent.parent
oracle=Path('D:/CodexBackups/selective_flow_20261003_1105/train_oracle_completed_20261006T0158Z/c/out/diagnostics.npz')
oof=Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_oof_20261006T0308Z/train_scalar_oof.npz')
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(oracle)=='92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658'
assert sha(oof)=='cff1a52f7e828c96cacf070ae4e051a5c34042ebb9435f44f5b3242cad91a7ea'
def selector(pool,mean):
 return np.argmin((pool-mean[:,None])**2,axis=1)
with np.load(oracle,allow_pickle=False) as z,np.load(oof,allow_pickle=False) as t:
 assert np.array_equal(z['row'],t['row_ids']) and np.max(np.abs(z['y'].astype(np.float64)-t['y']))==0
 y=z['y'].astype(np.float64);raw=z['raw_prediction'].astype(np.float64);path=z['learned_prediction_path'].astype(np.float64)
 assert path.shape==(1281,4) and np.max(np.abs(path[:,0]-raw))<2e-6
 pool=np.column_stack([raw,path[:,1:]])
 means={'old_learned_residual':z['p0'].astype(np.float64)-z['estimated_residual'].astype(np.float64),'video_OOF_teacher':t['mu'].astype(np.float64),'fit_only_mean_negative_control':t['fit_mean_baseline'].astype(np.float64)}
 fold=t['fold'].copy()
indices={name:selector(pool,mean) for name,mean in means.items()}
result={};baseline_mse=float(np.mean((raw-y)**2))
for name,idx in indices.items():
 pred=pool[np.arange(1281),idx];mean=means[name];qhat=(pred-mean)**2-(raw-mean)**2;observed=(pred-y)**2-(raw-y)**2;delta=pred-raw
 identity=observed-qhat-2*(mean-y)*delta;assert np.max(np.abs(identity))<1e-12 and np.max(qhat)<1e-12
 changed=idx!=0;result[name]={'selected_index_counts':np.bincount(idx,minlength=4).tolist(),'changed_rows':int(changed.sum()),'mse':float(np.mean((pred-y)**2)),'mae':float(np.mean(np.abs(pred-y))),'relative_mse_change_from_raw':float(np.mean(observed)/baseline_mse),'changed_observed_harm_fraction':float(np.mean(observed[changed]>1e-12)) if changed.any() else None,'proxy_vs_observed_risk_identity_max_error':float(np.max(np.abs(identity))),'by_fold':[{'fold':int(k),'rows':int((fold==k).sum()),'mse':float(np.mean((pred[fold==k]-y[fold==k])**2)),'raw_mse':float(np.mean((raw[fold==k]-y[fold==k])**2))} for k in range(3)]}
report={'status':'OOF_TEACHER_VS_OLD_RESIDUAL_FROZEN_LEARNED_PATH_CPU_SELECTOR_DIAGNOSTIC','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':1281,'candidate_pool':'Original F plus three pre-existing learned-residual path steps; excludes all label-known oracle-path candidates. No new vector or gradient optimization.','source_sha256':sha(__file__),'oracle_npz_sha256':sha(oracle),'teacher_oof_npz_sha256':sha(oof),'raw_mse':baseline_mse,'learned_last_step_mse':float(np.mean((pool[:,-1]-y)**2)),'results':result,'new_GPU_execution':False,'model_parameter_update':False,'dev_or_test_read':False,'whole_pipeline_crossfit':False,'scope':'Fixed reference and candidate paths were fullTRAINfit/DEVselected. Teacher scalar only was video isolated. Original TRAIN targets used to audit realized selector outcomes, never choose candidates; these outcomes are not new DEV performance or semantic truth.'}
out=base/'outputs'/'视频隔离教师对既有有限候选效用判断CPU诊断.json';assert not out.exists();out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report))
