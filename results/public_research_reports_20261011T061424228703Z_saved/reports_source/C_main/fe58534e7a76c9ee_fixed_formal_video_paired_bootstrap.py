"""Joint video-cluster uncertainty for already scored, fixed formal F/C predictions."""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
import numpy as np
NAMES=['Acc7','Acc2','F1','MAE','Corr']
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sufficient(p,y):
 nz=y!=0;yt=y[nz]>0;pt=p[nz]>=0
 tn=np.sum(~yt&~pt);fp=np.sum(~yt&pt);fn=np.sum(yt&~pt);tp=np.sum(yt&pt)
 return np.array([len(y),np.sum(np.round(np.clip(p,-3,3))==np.round(np.clip(y,-3,3))),tn,fp,fn,tp,np.sum(abs(p-y)),y.sum(),p.sum(),y@y,p@p,y@p],dtype=np.float64)
def five(s):
 n,correct,tn,fp,fn,tp,mae,sy,sp,sy2,sp2,syp=s.T
 nz=tn+fp+fn+tp
 def divide(a,b):return np.divide(a,b,out=np.zeros_like(a),where=b!=0)
 F1=divide((tn+fp)*divide(2*tn,2*tn+fp+fn)+(tp+fn)*divide(2*tp,2*tp+fp+fn),nz)
 denominator=np.sqrt(np.maximum(0,sy2-sy*sy/n)*np.maximum(0,sp2-sp*sp/n))
 corr=np.divide(syp-sy*sp/n,denominator,out=np.full_like(n,np.nan),where=denominator!=0)
 return np.stack([correct/n,divide(tn+tp,nz),F1,mae/n,corr],axis=-1)
def main():
 p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 assert sha(a.plan)==a.plan_sha;plan=json.loads(a.plan.read_text(encoding='utf8'));base=a.plan.parent
 for name,h in plan['source_sha256'].items():assert sha(base/name)==h
 for name,h in plan['input_sha256'].items():assert sha(base/name)==h
 assert not a.output.exists();a.output.mkdir();sys.path.insert(0,str(base));from sentiment_metrics_careflow_v1 import metrics
 result=dict(status='FIXED_FORMAL_VAL_TEST_VIDEO_PAIRED_BOOTSTRAP_COMPLETE',actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,protocol_SHA=a.plan_sha,scope=plan['scope'],roles={},new_model_forward=False,new_scoring_selection=False)
 for role in ('val','test'):
  with np.load(base/(role+'_fixed_pair.npz'),allow_pickle=False) as z:ids=z['row_ids'];y=z['labels'];F=z['minimal_fixed_F'].astype(np.float64);C=z['careflow'].astype(np.float64)
  videos=np.array([str(v).split('[',1)[0] for v in ids]);unique=sorted(set(videos));V=len(unique)
  sf=np.stack([sufficient(F[videos==v],y[videos==v]) for v in unique]);sc=np.stack([sufficient(C[videos==v],y[videos==v]) for v in unique])
  points={name:metrics(values,y) for name,values in [('minimal_fixed_F',F),('careflow',C)]}
  for name,values,s in [('minimal_fixed_F',F,sf),('careflow',C,sc)]:
   expected=np.array([points[name][k] for k in NAMES]);assert np.max(abs(five(s.sum(0))-expected))<1e-12
   for k in NAMES:assert abs(points[name][k]-plan['expected_fixed_points'][role][name][k])<1e-12
  rng=np.random.default_rng(plan['seed']+(role=='test'));draw=rng.integers(0,V,size=(plan['draws'],V));diff=five(sf[draw].sum(1))-five(sc[draw].sum(1))
  # Confirm repeated-video rows against independent existing metric implementation.
  errors=[]
  for row in draw[:10]:
   indices=np.concatenate([np.flatnonzero(videos==unique[i]) for i in row])
   direct=np.array([metrics(F[indices],y[indices])[k]-metrics(C[indices],y[indices])[k] for k in NAMES])
   from_stats=five(sf[row].sum(0))-five(sc[row].sum(0));errors.append(float(np.max(abs(direct-from_stats))))
  assert max(errors)<1e-12
  valid=np.isfinite(diff).all(1);d=diff[valid];point=np.array([points['minimal_fixed_F'][k]-points['careflow'][k] for k in NAMES])
  benefits=d*np.array([1,1,1,-1,1])
  result['roles'][role]=dict(rows=len(y),videos=V,video_ids=unique,fixed_points=points,F_minus_C={k:dict(point=float(point[i]),percentile95=np.quantile(d[:,i],[.025,.975]).tolist(),standard_error=float(d[:,i].std(ddof=1))) for i,k in enumerate(NAMES)},valid_joint_draws=int(valid.sum()),repeated_video_sufficient_statistics_max_error=max(errors),joint_all5_F_better_fraction=float(np.mean((benefits>0).all(1))),metric_difference_correlation=np.corrcoef(benefits.T).tolist(),no_epoch_or_model_selection_repeated=True)
  np.savez(a.output/(role+'_joint_bootstrap_draws.npz'),F_minus_C=d,metric_names=np.asarray(NAMES),video_indices=draw)
 (a.output/'actual_stage_receipt.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
 print(json.dumps(result))
if __name__=='__main__':main()
