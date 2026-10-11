"""Prospectively fixed exploratory diagnostic on saved INNER b/p, not donor-message p0/p1."""
import argparse, csv, datetime, hashlib, json, os, sys
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def sigmoid(z):
 z=np.asarray(z,dtype=np.float64)
 return np.exp(-np.logaddexp(0.,-z))
def objective(theta,X,b,d,y,a,ridge):
 g=sigmoid(X@theta);e=b+g*d-y
 penalty=theta.copy();penalty[0]=0
 loss=float(a@(e*e)+ridge*(penalty@penalty))
 grad=2*X.T@(a*e*d*g*(1-g))+2*ridge*penalty
 return loss,grad
def fit_gate(X,b,d,y,a,ridge,limit=600,tol=1e-7):
 theta=np.zeros(X.shape[1]);H=np.eye(len(theta));value,grad=objective(theta,X,b,d,y,a,ridge)
 for iteration in range(limit):
  if np.linalg.norm(grad,np.inf)<tol:break
  direction=-H@grad
  if grad@direction>=0:H=np.eye(len(theta));direction=-grad
  step=1.
  for trial in range(60):
   candidate=theta+step*direction;v2,g2=objective(candidate,X,b,d,y,a,ridge)
   if v2<=value+1e-4*step*(grad@direction):break
   step*=.5
  else:raise RuntimeError('Armijo line search failed')
  s=candidate-theta;t=g2-grad;curvature=float(s@t)
  if curvature>1e-12:
   Q=np.eye(len(theta))-np.outer(s,t)/curvature
   H=Q@H@Q.T+np.outer(s,s)/curvature
  else:H=np.eye(len(theta))
  theta,value,grad=candidate,v2,g2
 return theta,dict(iterations=iteration+1,objective=value,gradient_inf=float(np.linalg.norm(grad,np.inf)),converged=bool(np.linalg.norm(grad,np.inf)<tol))
def video_weights(v):
 _,idx,counts=np.unique(v,return_inverse=True,return_counts=True)
 return 1./(len(counts)*counts[idx])
def features(b,d):return np.stack([np.abs(b),np.abs(d),np.sign(b)*np.sign(d)],axis=1)
def lovo(b,p,y,v,ridge):
 d=p-b;F=features(b,d);output={k:np.empty_like(b) for k in ('constant','feature')};gates={k:np.empty_like(b) for k in output};fits=[]
 for heldout in sorted(set(v)):
  test=v==heldout;train=~test;a=video_weights(v[train]);denom=float(a@(d[train]**2))
  constant=float(np.clip((a@(d[train]*(y[train]-b[train])))/denom,0,1)) if denom else 0.
  mu=a@F[train];std=np.sqrt(a@((F[train]-mu)**2));std=np.where(std>0,std,1.)
  X=np.column_stack([np.ones(train.sum()),(F[train]-mu)/std])
  w,trace=fit_gate(X,b[train],d[train],y[train],a,ridge)
  g=sigmoid(np.column_stack([np.ones(test.sum()),(F[test]-mu)/std])@w)
  for name,gate in [('constant',np.full(test.sum(),constant)),('feature',g)]:
   gates[name][test]=gate;output[name][test]=b[test]+gate*d[test]
  fits.append(dict(heldout_video=str(heldout),train_videos=int(len(set(v[train]))),train_rows=int(train.sum()),heldout_rows=int(test.sum()),constant_gate=constant,feature_theta=w.tolist(),feature_mu=mu.tolist(),feature_std=std.tolist(),optimizer=trace))
 return output,gates,fits
def corr(x,y):
 if len(x)<2 or np.std(x)==0 or np.std(y)==0:return None
 return float(np.corrcoef(x,y)[0,1])
def summary(b,p,y,metric):
 d=p-b;r=y-b;U=r*r-(y-p)**2
 g=np.zeros_like(d);np.divide(r,d,out=g,where=d!=0);g=np.clip(g,0,1);oracle=b+g*d
 error=float(np.max(np.abs(U-(2*r*d-d*d))))
 assert error<1e-10 and np.mean((oracle-y)**2)<=min(np.mean((b-y)**2),np.mean((p-y)**2))+1e-12
 result={}
 for name,mask in [('all',np.ones(len(y),dtype=bool)),('weak',abs(y)<=1),('strong',abs(y)>1)]:
  yy=y[mask];dd=d[mask];rr=r[mask];uu=U[mask]
  result[name]=dict(rows=int(mask.sum()),corr_delta_residual=corr(dd,rr),positive_utility_fraction=float(np.mean(uu>0)),
   delta_mean=float(np.mean(dd)),delta_std=float(np.std(dd)),delta_quantiles=np.quantile(dd,[0,.1,.5,.9,1]).tolist(),
   label_std=float(np.std(yy)),b_std=float(np.std(b[mask])),p_std=float(np.std(p[mask])),
   b=metric(b[mask],yy),p=metric(p[mask],yy),continuous_oracle=metric(oracle[mask],yy),
   binary_oracle_MSE=float(np.mean(np.minimum((b[mask]-yy)**2,(p[mask]-yy)**2))),
   continuous_oracle_MSE_gain_vs_p=float(np.mean((p[mask]-yy)**2)-np.mean((oracle[mask]-yy)**2)))
 return result,oracle,g,error
def bootstrap(b,pred,y,v,draws,seed):
 videos=sorted(set(v));rng=np.random.default_rng(seed);sample=rng.integers(0,len(videos),size=(draws,len(videos)))
 result={}
 for method,q in pred.items():
  regions={}
  for name,mask in [('all',np.ones(len(y),bool)),('weak',abs(y)<=1),('strong',abs(y)>1)]:
   counts=np.array([np.sum((v==video)&mask) for video in videos])
   den=counts[sample].sum(axis=1);valid=den>0
   measures={}
   for loss,fn in [('MAE',lambda x:abs(x-y)),('MSE',lambda x:(x-y)**2)]:
    difference=fn(q)-fn(b)
    sums=np.array([difference[(v==video)&mask].sum() for video in videos])
    boots=sums[sample].sum(axis=1)[valid]/den[valid]
    measures[loss]=dict(point_difference_gate_minus_p=float(difference[mask].mean()),percentile95=np.quantile(boots,[.025,.975]).tolist(),standard_error=float(np.std(boots,ddof=1)),valid_draws=int(valid.sum()),conditional_probability_draw_difference_below_zero=float(np.mean(boots<0)))
   regions[name]=measures
  result[method]=regions
 return result
def tests():
 rng=np.random.default_rng(178);X=np.column_stack([np.ones(40),rng.normal(size=(40,3))]);b=rng.normal(size=40);d=rng.normal(size=40);y=rng.normal(size=40);a=np.full(40,1/40);theta=rng.normal(size=4)
 f,g=objective(theta,X,b,d,y,a,.01);eps=1e-6
 numeric=np.array([(objective(theta+np.eye(4)[j]*eps,X,b,d,y,a,.01)[0]-objective(theta-np.eye(4)[j]*eps,X,b,d,y,a,.01)[0])/(2*eps) for j in range(4)])
 assert np.max(abs(g-numeric))<1e-8
 v=np.repeat(['a','b','c','d'],10);y=b+.3*d;pred,gates,fits=lovo(b,b+d,y,v,.01)
 assert np.max(abs(gates['constant']-.3))<1e-12
 yy=y.copy();yy[v=='b']+=900;other,_,_=lovo(b,b+d,yy,v,.01)
 for k in pred:assert np.array_equal(pred[k][v=='b'],other[k][v=='b'])
 assert all(t['optimizer']['converged'] for t in fits)
 return dict(finite_difference_max_error=float(np.max(abs(g-numeric))),closed_constant_exact=True,heldout_labels_do_not_affect_heldout_gate=True,synthetic_does_not_establish_empirical_gain=True)
def main():
 arg=argparse.ArgumentParser();arg.add_argument('--plan',type=Path,required=True);arg.add_argument('--plan-sha',required=True);arg.add_argument('--output',type=Path,required=True);a=arg.parse_args()
 assert sha(a.plan)==a.plan_sha;plan=json.loads(a.plan.read_text(encoding='utf8'));bundle=a.plan.parent
 assert sha(__file__)==plan['source_sha256'][Path(__file__).name]
 for n,h in plan['source_sha256'].items():assert sha(bundle/n)==h
 for n,h in plan['input_sha256'].items():assert sha(bundle/n)==h
 assert not a.output.exists();a.output.mkdir()
 sys.path.insert(0,str(bundle));from sentiment_metrics_careflow_v1 import metrics
 result=dict(status='SAVED_INNER_OUTPUT_GATE_LOVO_EXPLORATORY_COMPLETE',actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,plan_sha256=a.plan_sha,synthetic_tests=tests(),scope=plan['scope'],no_GPU_or_model_forward=True)
 for role in ('fit','inner'):
  with np.load(bundle/(role+'_bfp.npz'),allow_pickle=False) as z:
   b=np.asarray(z['b'],dtype=np.float64).reshape(-1);p=np.asarray(z['p'],dtype=np.float64).reshape(-1);ids=z['row_ids'].astype(str)
  y=np.load(bundle/(role+'_evaluation_targets.npy'),allow_pickle=False).astype(np.float64).reshape(-1)
  v=np.array([i.split('[',1)[0] for i in ids]);assert len(set(ids))==len(ids)==len(y)
  s,oracle,og,error=summary(b,p,y,metrics);result[role]=dict(videos=len(set(v)),summary=s,utility_identity_max_error=error)
  if role=='inner':
   q,g,fits=lovo(b,p,y,v,plan['ridge']);result[role]['lovo_fits']=fits
   result[role]['gate_scores']={k:{region:metrics(value[mask],y[mask]) for region,mask in [('all',np.ones(len(y),bool)),('weak',abs(y)<=1),('strong',abs(y)>1)]} for k,value in q.items()}
   result[role]['paired_video_bootstrap']=bootstrap(p,q,y,v,plan['bootstrap_draws'],plan['bootstrap_seed'])
   with (a.output/'inner_y_b_p_video.csv').open('w',encoding='utf8',newline='') as f:
    writer=csv.writer(f);writer.writerow(['row_id','video','y','b','p','constant_lovo','feature_lovo','oracle_continuous'])
    writer.writerows(zip(ids,v,y,b,p,q['constant'],q['feature'],oracle))
   np.savez(a.output/'fixed_lovo_predictions.npz',row_ids=ids,video=v,y=y,b=b,p=p,**q,oracle_continuous=oracle,constant_gate=g['constant'],feature_gate=g['feature'])
 write(a.output/'actual_stage_receipt.json',result)
 print(json.dumps({k:result[k] for k in ['status','actual_utc','synthetic_tests']}))
 print(json.dumps(result['inner']['gate_scores']))
 print(json.dumps(result['inner']['paired_video_bootstrap']))
if __name__=='__main__':main()
