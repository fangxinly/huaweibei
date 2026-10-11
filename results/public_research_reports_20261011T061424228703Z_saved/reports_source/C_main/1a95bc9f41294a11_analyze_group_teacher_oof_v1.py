"""TRAIN scalar OOF diagnostics only after three full D and independent CPU saves."""
from pathlib import Path
import argparse,datetime,hashlib,json,pickle
import numpy as np
from audit_group_teacher_completed_files_v1 import audit,sha

p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out-directory',type=Path,required=True);a=p.parse_args()
assert not a.out_directory.exists()
assert sha(a.dataset)=='5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b'
verified=[]
for fold,node in enumerate(['a','b','c']):
 d=a.directory/node
 r=audit(d,fold);cpu=json.loads((d/'cpu_preservation_receipt.json').read_text(encoding='utf-8'));m=json.loads((d/'preservation_manifest.json').read_text(encoding='utf-8'))
 assert cpu['status']=='INDEPENDENT_TRAINING_AND_ASSEMBLY_HOST_COMPLETED_TEACHER_SEVEN_FILES_EXTRAS_CPU_SHA_ZIP_TENSORS_ARRAYS_VERIFIED'
 assert cpu['training_node']==cpu['assembly_node']==node and cpu['target_cpu_node']!=node and not cpu['cuda_initialized']
 assert cpu['manifest_sha256']==sha(d/'preservation_manifest.json')
 assert cpu['required_seven_files']==m['required_seven_files'] and cpu['extra_files']==m['extra_files']
 for group in ['required_seven_files','extra_files']:
  for name,meta in m[group].items():assert (d/name).stat().st_size==meta['bytes'] and sha(d/name)==meta['sha256']
 verified.append((d,r,cpu))
# Consume TRAIN outcome targets only after all three selected models and original
# arrays are fixed and locally/independently preserved.
train=pickle.load(a.dataset.open('rb'))['train'];assert len(train)==1281
y=np.array([float(np.asarray(x[1]).reshape(-1)[0]) for x in train],dtype=np.float64);assert np.isfinite(y).all()
mu=np.full(1281,np.nan);baseline=np.full(1281,np.nan);fold_ids=np.full(1281,-1,dtype=np.int64);fold_reports=[]
mapping=json.loads((verified[0][0]/'train_row_video_mapping.json').read_text(encoding='utf-8'))
assert all(row['row']==i and row['segment_id']==train[i][2] for i,row in enumerate(mapping))
video=np.array([r['video_id'] for r in mapping]);assert len(set(video))==52
for fold,node in enumerate(['a','b','c']):
 d,r,cpu=verified[fold]
 assert sha(d/'train_row_video_mapping.json')==sha(verified[0][0]/'train_row_video_mapping.json')
 outer=np.load(d/f'outer_{fold}.npy',allow_pickle=False);fit=np.load(d/f'fit_{fold}.npy',allow_pickle=False)
 with np.load(d/'run_v1/outer_scalar_predictions.npz',allow_pickle=False) as z:
  assert np.array_equal(z['row_ids'],outer) and np.all(fold_ids[outer]==-1)
  mu[outer]=z['mu'].astype(np.float64);baseline[outer]=np.mean(y[fit]);fold_ids[outer]=fold
 with np.load(d/'run_v1/selected_inner_predictions.npz',allow_pickle=False) as z:assert np.array_equal(z['label'],y[z['row_ids']].astype(z['label'].dtype))
 fold_reports.append({'fold':fold,'node':node,'outer_rows':len(outer),'best_epoch':r['best_epoch'],'inner_mse':r['best_inner_mse'],'outer_mse':float(np.mean((mu[outer]-y[outer])**2)),'outer_mae':float(np.mean(np.abs(mu[outer]-y[outer]))),'fit_only_mean_baseline_mse':float(np.mean((baseline[outer]-y[outer])**2)),'CPU_receipt_sha256':sha(d/'cpu_preservation_receipt.json')})
assert np.all(fold_ids>=0) and np.isfinite(mu).all() and np.isfinite(baseline).all()
videos=[]
for name in sorted(set(video)):
 ids=np.flatnonzero(video==name);assert len(set(fold_ids[ids]))==1
 videos.append({'video_id':name,'fold':int(fold_ids[ids][0]),'rows':len(ids),'teacher_squared_sum':float(np.sum((mu[ids]-y[ids])**2)),'fit_mean_squared_sum':float(np.sum((baseline[ids]-y[ids])**2))})
result={'status':'THREE_PRESERVED_COMPLETED100_TRAIN_VIDEO_ISOLATED_SCALAR_OOF_DIAGNOSTICS','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'train_rows':1281,'dev_or_test_labels_consumed':False,'teacher_mse':float(np.mean((mu-y)**2)),'teacher_mae':float(np.mean(np.abs(mu-y))),'fit_only_mean_baseline_mse':float(np.mean((baseline-y)**2)),'fit_only_mean_baseline_mae':float(np.mean(np.abs(baseline-y))),'folds':fold_reports,'whole_pipeline_crossfit':False,'scope':'Only scalar teacher training/normalization/selection isolated by TRAIN video. Existing A features remain fullTRAINfit/DEVselected. This single-seed OOF diagnostic is not five-seed stability, test performance, a new directional strategy result, or semantic shared/complementary/interference truth.'}
result['video_rows']=videos
result['video_macro_teacher_mse']=float(np.mean([v['teacher_squared_sum']/v['rows'] for v in videos]));result['videos_teacher_better_than_fit_mean']=sum(v['teacher_squared_sum']<v['fit_mean_squared_sum'] for v in videos)
a.out_directory.mkdir();np.savez(a.out_directory/'train_scalar_oof.npz',row_ids=np.arange(1281,dtype=np.int64),fold=fold_ids,mu=mu,y=y,fit_mean_baseline=baseline)
(a.out_directory/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({'status':result['status'],'mse':result['teacher_mse'],'mae':result['teacher_mae']}))
