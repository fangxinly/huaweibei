"""Fixed prediction-only smooth correction gate; one development evaluation."""
import argparse,datetime,hashlib,json,os,subprocess,sys
from pathlib import Path
import numpy as np
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def gate(a):
 plan=read(a.plan);assert sha(a.plan)==a.plan_sha and plan['status']=='FIXED_PREDICTION_ONLY_GATE_DEVELOPMENT_PROTOCOL_FROZEN'
 root=a.plan.parent
 for n,h in plan['source_sha256'].items():assert sha(root/n)==h
 for n,h in plan['input_sha256'].items():assert sha(root/n)==h
 sys.path.insert(0,str(root));return plan,root
def worker(a):
 plan,root=gate(a);assert not a.out.exists();a.out.mkdir()
 from sentiment_metrics_careflow_v1 import metrics
 arrays={};pred_sha={};ids={}
 for role,count in [('fit',1494),('inner',264)]:
  with np.load(root/(role+'_bfp.npz'),allow_pickle=False) as z:b=z['b'].astype(np.float64).reshape(-1);p=z['p'].astype(np.float64).reshape(-1);ids[role]=z['row_ids']
  assert len(b)==count and b.shape==p.shape and np.isfinite(b).all() and np.isfinite(p).all()
  weight=b*b/(1+b*b);q=b+weight*(p-b);arrays[role]=(b,p,q,weight)
  np.savez(a.out/(role+'_gated_prediction_only.npz'),prediction=q,gate=weight,row_ids=ids[role]);pred_sha[role]=sha(a.out/(role+'_gated_prediction_only.npz'))
 assert not set(ids['fit'])&set(ids['inner'])
 write(a.out/'prediction_freeze.json',dict(actual_utc=utc(),prediction_sha256=pred_sha,before_this_run_label_load=True,formula='q=b+(b^2/(1+b^2))*(p-b)',not_using_true_label_gate=True))
 result={};regions={}
 for role in ('fit','inner'):
  y=np.load(root/(role+'_evaluation_targets.npy'),allow_pickle=False).astype(np.float64).reshape(-1);b,p,q,w=arrays[role]
  result[role]=metrics(q,y);regions[role]={}
  for name,mask in [('weak',np.abs(y)<=1),('strong',np.abs(y)>1)]:
   regions[role][name]=dict(rows=int(mask.sum()),b_MAE=float(np.abs(b[mask]-y[mask]).mean()),p_MAE=float(np.abs(p[mask]-y[mask]).mean()),q_MAE=float(np.abs(q[mask]-y[mask]).mean()),mean_gate=float(w[mask].mean()),base_abs_gt1_rows=int((np.abs(b[mask])>1).sum()))
 write(a.out/'actual_stage_receipt.json',dict(status='FIXED_PREDICTION_GATE_DEVELOPMENT_LOCAL_CPU_COMPLETE',actual_utc=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,plan_sha256=a.plan_sha,scores=result,regions=regions,prediction_sha256=pred_sha,no_GPU_forward=True,no_official_VAL_TEST_access=True,no_fit_or_parameter_update=True,history_FIT_INNER_already_explored=True,no_new_independent_confirmation=True))
def audit(a):
 plan,root=gate(a);assert not a.out.exists();a.out.mkdir();g=read(a.original/'actual_stage_receipt.json');assert g['plan_sha256']==a.plan_sha
 errors={}
 for role in ('fit','inner'):
  with np.load(root/(role+'_bfp.npz'),allow_pickle=False) as z:b=z['b'].astype(float).reshape(-1);p=z['p'].astype(float).reshape(-1)
  ref_q=p-(p-b)/(1+np.square(b))
  path=a.original/(role+'_gated_prediction_only.npz');assert sha(path)==g['prediction_sha256'][role]
  with np.load(path,allow_pickle=False) as z:q=z['prediction']
  formula_error=float(np.max(np.abs(q-ref_q)));assert formula_error<1e-12
  y=np.load(root/(role+'_evaluation_targets.npy'),allow_pickle=False).astype(float).reshape(-1);nz=y!=0;t=(y[nz]>=0).astype(int);v=(q[nz]>=0).astype(int)
  matrix=np.array([[np.sum((t==i)&(v==j)) for j in (0,1)] for i in (0,1)]);support=matrix.sum(1);den=support+matrix.sum(0)
  f1=sum(float(support[i]*2*matrix[i,i]/den[i]) if den[i] else 0. for i in (0,1))/len(t)
  ref=dict(Acc7=float(np.equal(np.rint(np.minimum(3,np.maximum(-3,q))),np.rint(np.minimum(3,np.maximum(-3,y)))).mean()),Acc2=float((t==v).mean()),F1=f1,MAE=float(np.abs(q-y).sum()/len(y)),Corr=float(np.corrcoef(q,y)[0,1]),MSE=float(np.square(q-y).sum()/len(y)))
  errors[role]={k:abs(ref[k]-g['scores'][role][k]) for k in ref};errors[role]['prediction_formula']=formula_error;assert max(errors[role].values())<1e-12
 write(a.out/'actual_stage_receipt.json',dict(status='INDEPENDENT_GATE_FORMULA_AND_FIVE_METRICS_CPU_AUDIT_COMPLETE',actual_utc=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,plan_sha256=a.plan_sha,original_receipt_sha256=sha(a.original/'actual_stage_receipt.json'),errors=errors,no_model_forward=True))
def run(a):
 gate(a);assert not a.out.exists();a.out.mkdir();cmd=[sys.executable,str(Path(__file__).resolve()),a.worker,'--plan',str(a.plan),'--plan-sha',a.plan_sha,'--out',str(a.child_out)]
 if a.original:cmd+=['--original',str(a.original)]
 with (a.out/'stdout.log').open('wb') as out,(a.out/'stderr.log').open('wb') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err);write(a.out/'actual_child.json',dict(pid=child.pid,fullargv=cmd,actual_start_utc=utc()));code=child.wait(timeout=60)
 write(a.out/'natural_exit.json',dict(pid=child.pid,fullargv=cmd,natural_exit=code,actual_exit_utc=utc()));raise SystemExit(code)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['worker','audit','run']);p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--original',type=Path);p.add_argument('--worker',choices=['worker','audit']);p.add_argument('--child-out',type=Path);a=p.parse_args();globals()[a.action](a)
