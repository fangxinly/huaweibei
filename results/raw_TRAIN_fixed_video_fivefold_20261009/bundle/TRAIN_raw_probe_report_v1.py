"""Synthetic-qualified TRAIN diagnostic reporter; no actual-file entry point yet."""
import json,math,sys
import numpy as np
PATHS=('T','T_A','T_V','T_A_V')
PAIRS=(('T','T_A'),('T','T_V'),('T_A','T_A_V'),('T_V','T_A_V'))
def report(row_ids,videos,folds,labels,predictions):
 n=len(row_ids);assert len(set(row_ids))==n and n>0 and len(videos)==len(folds)==len(labels)==n
 labels=np.asarray(labels,dtype=np.float64);assert labels.shape==(n,) and np.isfinite(labels).all()
 assert set(predictions)==set(PATHS);by_video={}
 for i,(v,f) in enumerate(zip(videos,folds)):
  assert type(f) is int and 0<=f<5
  if v not in by_video:by_video[v]=dict(fold=f,indices=[])
  assert by_video[v]['fold']==f;by_video[v]['indices'].append(i)
 assert {x['fold'] for x in by_video.values()}==set(range(5))
 errors={}
 for k in PATHS:
  p=np.asarray(predictions[k],dtype=np.float64);assert p.shape==(n,) and np.isfinite(p).all()
  errors[k]=[(float(p[i])-float(labels[i]))**2 for i in range(n)]
  assert all(math.isfinite(e) for e in errors[k])
 mse={k:math.fsum(v)/n for k,v in errors.items()}
 pairs=[dict(baseline=a,augmented=b,pooled_row_MSE_reduction=mse[a]-mse[b]) for a,b in PAIRS]
 video_rows=[]
 for video in sorted(by_video):
  item=by_video[video];indices=item['indices'];vmse={k:math.fsum(errors[k][i] for i in indices)/len(indices) for k in PATHS}
  video_rows.append(dict(video_id=video,fold=item['fold'],rows=len(indices),MSE=vmse,paired_MSE_reductions=[dict(baseline=a,augmented=b,reduction=vmse[a]-vmse[b]) for a,b in PAIRS]))
 return dict(status='FINITE_RAW_TRAIN_ESTIMATOR_DESCRIPTIVE_REPORT',rows=n,videos=len(by_video),pooled_row_weighted_MSE=mse,paired_comparisons=pairs,per_video=video_rows,positive_reduction_definition='baseline MSE minus augmented MSE; positive means smaller squared prediction error for this fixed estimator.',not_claimed=['Official VAL/TEST five metrics','Mutual information or PID','Bayes ceiling','A/B flow mechanism','Statistical significance or stable benefit','Upstream raw-data construction isolation'])
def qualify():
 videos=sum(([str(f)]*(f+1) for f in range(5)),[]);folds=[int(v) for v in videos];n=len(videos);ids=[str(i) for i in range(n)];labels=np.linspace(-1.,1.,n)
 predictions={k:labels+offset for k,offset in zip(PATHS,[2.,1.,3.,.5])};actual=report(ids,videos,folds,labels,predictions)
 expected={'T':4.,'T_A':1.,'T_V':9.,'T_A_V':.25}
 assert all(abs(actual['pooled_row_weighted_MSE'][k]-v)<1e-14 for k,v in expected.items())
 for pair in actual['paired_comparisons']:assert abs(pair['pooled_row_MSE_reduction']-(expected[pair['baseline']]-expected[pair['augmented']]))<1e-14
 varied={k:labels+np.array(folds,dtype=float) for k in PATHS};weighted=report(ids,videos,folds,labels,varied)
 assert abs(weighted['pooled_row_weighted_MSE']['T']-math.fsum(f*f for f in folds)/n)<1e-14
 assert abs(weighted['pooled_row_weighted_MSE']['T']-math.fsum(x['MSE']['T'] for x in weighted['per_video'])/5)>1e-2
 refused={}
 wrong=[dict(ids=ids,vs=videos,fs=[1]+folds[1:],ps=predictions,name='video_cross_fold'),dict(ids=ids[:-1]+[ids[0]],vs=videos,fs=folds,ps=predictions,name='duplicate_row'),dict(ids=ids,vs=videos,fs=folds,ps={**predictions,'T':np.full(n,np.nan)},name='nonfinite_prediction')]
 # First video has one row; move one row from a multirow video to split it.
 wrong[0]['fs']=list(folds);wrong[0]['fs'][1]=2
 for x in wrong:
  try:report(x['ids'],x['vs'],x['fs'],labels,x['ps'])
  except AssertionError:refused[x['name']]=True
  else:raise AssertionError(x['name']+' accepted')
 return dict(status='SYNTHETIC_TRAIN_REPORTER_QUALIFIED',known_MSE_and_paired_delta_passed=True,unequal_video_size_weighting_distinguished=True,standard_library_fsum_crosscheck_passed=True,rejections=refused,real_data_loaded=False,real_probe_scored=False)
if __name__=='__main__':
 assert sys.argv[1:]==['--qualify'],'Actual prediction-file scoring is not enabled in this source.'
 print(json.dumps(qualify(),allow_nan=False))
