"""Independent reconstruction of observed fixed-pool selector metrics."""
from pathlib import Path
import json,datetime,numpy as np,hashlib
root=Path(__file__).resolve().parent.parent
old=Path('D:/CodexBackups/selective_flow_20261003_1105/train_oracle_completed_20261006T0158Z/c/out/diagnostics.npz');oof=Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_oof_20261006T0308Z/train_scalar_oof.npz')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();original=root/'outputs/视频隔离教师对既有有限候选效用判断CPU诊断.json';r=json.loads(original.read_text(encoding='utf-8'));assert sha(old)==r['oracle_npz_sha256'] and sha(oof)==r['teacher_oof_npz_sha256']
with np.load(old) as a,np.load(oof) as t:
 raw=a['raw_prediction'].astype(np.float64);y=a['y'].astype(np.float64);candidate=np.column_stack((raw,a['learned_prediction_path'][:,1:].astype(np.float64)))
 assert np.array_equal(a['row'],t['row_ids']) and np.array_equal(y,t['y'])
 means={'old_learned_residual':a['p0'].astype(np.float64)-a['estimated_residual'].astype(np.float64),'video_OOF_teacher':t['mu'].astype(np.float64),'fit_only_mean_negative_control':t['fit_mean_baseline'].astype(np.float64)}
 bmse=float(np.average((raw-y)**2));assert abs(bmse-r['raw_mse'])<1e-12
 for name,mean in means.items():
  idx=np.argmin((candidate-mean.reshape(-1,1))**2,axis=1);chosen=candidate[np.arange(len(y)),idx];report=r['results'][name];changed=idx>0;observed=(chosen-y)**2-(raw-y)**2
  assert report['selected_index_counts']==np.bincount(idx,minlength=4).tolist() and report['changed_rows']==int(changed.sum())
  values={'mse':float(np.average((chosen-y)**2)),'mae':float(np.average(np.abs(chosen-y))),'relative_mse_change_from_raw':float(np.average(observed)/bmse),'changed_observed_harm_fraction':float(np.average(observed[changed]>1e-12))}
  assert all(abs(values[k]-report[k])<1e-12 for k in values)
  for f in range(3):
   ids=t['fold']==f;assert abs(report['by_fold'][f]['mse']-np.average((chosen[ids]-y[ids])**2))<1e-12
out={'status':'NEW_OOF_FIXED_CANDIDATE_SELECTOR_ORIGINAL_ARRAYS_COUNTS_AND_OBSERVED_METRICS_INDEPENDENTLY_RECONSTRUCTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_report_sha256':sha(original),'auditor_sha256':sha(__file__),'rows':1281,'no_new_GPU_or_parameter_update':True,'no_DEV_or_TEST_labels':True,'scope':'Same old learned-residual candidate pool diagnostic; does not establish performance of a new teacher-driven vector solver or whole-pipeline crossfit.'}
dest=root/'outputs/视频隔离教师有限候选选择CPU独立核验.json';assert not dest.exists();dest.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(out['status'])
