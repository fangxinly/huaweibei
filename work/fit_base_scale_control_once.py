"""Single FIT-only nonnegative affine control; never access official VAL/TEST."""
import argparse,datetime,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def gate(a):
 p=read(a.plan);assert sha(a.plan)==a.plan_sha and p['status']=='FIT_ONLY_BASE_AFFINE_CONTROL_SINGLE_FROZEN'
 root=a.plan.parent
 for n,h in p['source_sha256'].items():assert sha(root/n)==h
 for n,h in p['input_sha256'].items():assert sha(root/n)==h
 sys.path.insert(0,str(root));return p,root
def worker(a):
 p,root=gate(a);assert not a.out.exists();a.out.mkdir()
 from sentiment_metrics_careflow_v1 import metrics
 with np.load(root/'fit_bfp.npz',allow_pickle=False) as z:fit_b=z['b'].astype(np.float64);fit_ids=z['row_ids']
 assert len(fit_b)==1494
 fit_y=np.load(root/'fit_evaluation_targets.npy',allow_pickle=False).astype(np.float64).reshape(-1)
 assert fit_y.shape==fit_b.shape and np.isfinite(fit_y).all()
 mean_b=float(fit_b.mean());mean_y=float(fit_y.mean());bc=fit_b-mean_b;yc=fit_y-mean_y
 variance=float(np.dot(bc,bc));alpha=max(0.,float(np.dot(bc,yc)/variance)) if variance>0 else 0.;beta=mean_y-alpha*mean_b
 write(a.out/'coefficient_freeze.json',dict(actual_utc=utc(),alpha=alpha,beta=beta,fit_rows=1494,fit_predictions_sha256=sha(root/'fit_bfp.npz'),fit_targets_sha256=sha(root/'fit_evaluation_targets.npy'),only_FIT_labels_used=True,no_hyperparameter_search=True))
 with np.load(root/'inner_bfp.npz',allow_pickle=False) as z:inner_b=z['b'].astype(np.float64);inner_ids=z['row_ids']
 assert len(inner_b)==264 and not set(fit_ids)&set(inner_ids)
 pred_sha={}
 for role,b,ids in [('fit',fit_b,fit_ids),('inner',inner_b,inner_ids)]:
  np.savez(a.out/(role+'_q_prediction_only.npz'),prediction=alpha*b+beta,row_ids=ids)
  pred_sha[role]=sha(a.out/(role+'_q_prediction_only.npz'))
 write(a.out/'prediction_freeze.json',dict(actual_utc=utc(),sha256=pred_sha,coefficient_sha256=sha(a.out/'coefficient_freeze.json'),before_INNER_label_load=True))
 scores={};weak_strong={}
 for role,y in [('fit',fit_y),('inner',np.load(root/'inner_evaluation_targets.npy',allow_pickle=False).astype(np.float64).reshape(-1))]:
  with np.load(a.out/(role+'_q_prediction_only.npz'),allow_pickle=False) as z:q=z['prediction']
  scores[role]=metrics(q,y)
  weak_strong[role]={name:dict(rows=int(mask.sum()),MAE=float(np.abs(q[mask]-y[mask]).mean())) for name,mask in [('weak',np.abs(y)<=1),('strong',np.abs(y)>1)]}
 write(a.out/'actual_stage_receipt.json',dict(status='FIT_ONLY_BASE_AFFINE_CONTROL_COMPLETE_LOCAL_CPU',actual_utc=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,plan_sha256=a.plan_sha,alpha=alpha,beta=beta,scores=scores,weak_strong=weak_strong,prediction_sha256=pred_sha,no_GPU_forward=True,no_official_VAL_TEST_access=True,no_MODEL_weights_changed=True,scope='Fixed p-selected20 b calibration development control; b is not separately trained no-flow model'))
def audit(a):
 p,root=gate(a);assert not a.out.exists();a.out.mkdir();g=read(a.original/'actual_stage_receipt.json');assert g['plan_sha256']==a.plan_sha
 with np.load(root/'fit_bfp.npz',allow_pickle=False) as z:b=z['b'].astype(np.float64)
 y=np.load(root/'fit_evaluation_targets.npy',allow_pickle=False).astype(np.float64).reshape(-1)
 slope,intercept=np.linalg.lstsq(np.column_stack([b,np.ones(len(b))]),y,rcond=None)[0]
 if slope<0:slope=0.;intercept=float(y.mean())
 assert max(abs(slope-g['alpha']),abs(intercept-g['beta']))<1e-12
 errors={}
 for role in ('fit','inner'):
  assert sha(a.original/(role+'_q_prediction_only.npz'))==g['prediction_sha256'][role]
  with np.load(a.original/(role+'_q_prediction_only.npz'),allow_pickle=False) as z:q=z['prediction'].astype(np.float64)
  y=np.load(root/(role+'_evaluation_targets.npy'),allow_pickle=False).astype(np.float64).reshape(-1);mask=y!=0;t=(y[mask]>=0).astype(int);v=(q[mask]>=0).astype(int)
  confusion=np.array([[np.sum((t==i)&(v==j)) for j in (0,1)] for i in (0,1)]);support=confusion.sum(1);den=support+confusion.sum(0)
  f1=sum(float(support[i]*2*confusion[i,i]/den[i]) if den[i] else 0. for i in (0,1))/len(t)
  ref=dict(Acc7=float(np.equal(np.rint(np.minimum(3,np.maximum(-3,q))),np.rint(np.minimum(3,np.maximum(-3,y)))).mean()),Acc2=float((t==v).mean()),F1=f1,MAE=float(np.abs(q-y).sum()/len(y)),Corr=float(np.corrcoef(q,y)[0,1]),MSE=float(np.square(q-y).sum()/len(y)))
  errors[role]={k:abs(ref[k]-g['scores'][role][k]) for k in ref};assert max(errors[role].values())<1e-12
 write(a.out/'actual_stage_receipt.json',dict(status='INDEPENDENT_CPU_OLS_AND_FIVE_METRICS_AUDIT_COMPLETE',actual_utc=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,plan_sha256=a.plan_sha,original_receipt_sha256=sha(a.original/'actual_stage_receipt.json'),errors=errors,alpha_error=abs(slope-g['alpha']),beta_error=abs(intercept-g['beta']),no_model_forward=True))
def run(a):
 gate(a);assert not a.out.exists();a.out.mkdir();cmd=[sys.executable,str(Path(__file__).resolve()),a.worker,'--plan',str(a.plan),'--plan-sha',a.plan_sha,'--out',str(a.child_out)]
 if a.original:cmd+=['--original',str(a.original)]
 with (a.out/'stdout.log').open('wb') as out,(a.out/'stderr.log').open('wb') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err);write(a.out/'actual_child.json',dict(pid=child.pid,fullargv=cmd,actual_start_utc=utc()));code=child.wait(timeout=60)
 write(a.out/'natural_exit.json',dict(pid=child.pid,fullargv=cmd,natural_exit=code,actual_exit_utc=utc()));raise SystemExit(code)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','audit','run']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--original',type=Path);p.add_argument('--worker',choices=['worker','audit']);p.add_argument('--child-out',type=Path);a=p.parse_args();globals()[a.action](a)
