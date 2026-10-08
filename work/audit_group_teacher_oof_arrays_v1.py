"""Independent NumPy reconstruction of preserved scalar OOF diagnostic arrays."""
from pathlib import Path
import argparse,json,pickle,hashlib,datetime
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--completed',type=Path,required=True);p.add_argument('--oof',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert sha(a.dataset)=='5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b'
train=pickle.load(a.dataset.open('rb'))['train'];assert len(train)==1281
y=np.array([float(np.asarray(row[1]).ravel()[0]) for row in train]);mapping=json.loads((a.completed/'a/train_row_video_mapping.json').read_text());assert len(mapping)==1281 and all(v['row']==i and v['segment_id']==train[i][2] for i,v in enumerate(mapping))
mu=np.full(1281,np.nan);baseline=np.full(1281,np.nan);folds=np.full(1281,-1,dtype=np.int64);details=[]
for fold,node in enumerate(['a','b','c']):
 d=a.completed/node;outer=np.load(d/f'outer_{fold}.npy');fit=np.load(d/f'fit_{fold}.npy');assert np.all(folds[outer]==-1)
 with np.load(d/'run_v1/outer_scalar_predictions.npz') as z:assert np.array_equal(z['row_ids'],outer);mu[outer]=z['mu'];folds[outer]=fold
 baseline[outer]=np.mean(y[fit]);details.append({'fold':fold,'rows':len(outer),'teacher_mse':float(np.mean((mu[outer]-y[outer])**2)),'fit_mean_mse':float(np.mean((baseline[outer]-y[outer])**2))})
assert np.all(folds>=0) and np.isfinite(mu).all()
with np.load(a.oof/'train_scalar_oof.npz') as z:
 assert set(z.files)=={'row_ids','fold','mu','y','fit_mean_baseline'} and np.array_equal(z['row_ids'],np.arange(1281))
 for n,value in [('fold',folds),('mu',mu),('y',y),('fit_mean_baseline',baseline)]:assert np.array_equal(z[n],value)
r=json.loads((a.oof/'analysis.json').read_text());assert r['status']=='THREE_PRESERVED_COMPLETED100_TRAIN_VIDEO_ISOLATED_SCALAR_OOF_DIAGNOSTICS'
metrics={'teacher_mse':float(np.mean((mu-y)**2)),'teacher_mae':float(np.mean(np.abs(mu-y))),'fit_only_mean_baseline_mse':float(np.mean((baseline-y)**2)),'fit_only_mean_baseline_mae':float(np.mean(np.abs(baseline-y)))}
for n,value in metrics.items():assert abs(r[n]-value)<1e-12
videos=[]
for name in sorted({v['video_id'] for v in mapping}):
 ids=np.array([v['row'] for v in mapping if v['video_id']==name]);assert len(set(folds[ids]))==1
 videos.append({'video_id':name,'fold':int(folds[ids][0]),'rows':len(ids),'teacher_squared_sum':float(np.sum((mu[ids]-y[ids])**2)),'fit_mean_squared_sum':float(np.sum((baseline[ids]-y[ids])**2))})
assert r['video_rows']==videos and len(videos)==52 and r['videos_teacher_better_than_fit_mean']==sum(v['teacher_squared_sum']<v['fit_mean_squared_sum'] for v in videos)
assert abs(r['video_macro_teacher_mse']-np.mean([v['teacher_squared_sum']/v['rows'] for v in videos]))<1e-12
gain=(metrics['fit_only_mean_baseline_mse']-metrics['teacher_mse'])/metrics['fit_only_mean_baseline_mse']
out={'status':'PRESERVED_THREE_TEACHER_TRAIN_VIDEO_OOF_ORIGINAL_ARRAYS_ROW_MAP_LABELS_BASELINE_METRICS_INDEPENDENTLY_RECONSTRUCTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':1281,'videos':52,'metrics':metrics,'relative_mse_reduction_from_fit_mean':float(gain),'folds':details,'teacher_better_videos':r['videos_teacher_better_than_fit_mean'],'dataset_sha256':sha(a.dataset),'OOF_npz_sha256':sha(a.oof/'train_scalar_oof.npz'),'original_analysis_sha256':sha(a.oof/'analysis.json'),'dev_or_test_labels_consumed':False,'whole_pipeline_crossfit':False,'scope':'Teacher scalar quality diagnostic. No new control policy benefit, semantic truth or multiple-seed stability is established.'}
assert not a.out.exists();a.out.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(out))
