"""Independent augmented least-squares audit, using original permitted FIT labels only."""
import argparse,datetime,hashlib,json,os,sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--out',required=True);a=p.parse_args();root=Path(a.run)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
r=read(root/'out/actual_matched_head_fit_receipt.json');ex=read(root/'natural_exit.json');head=read(root/'out/head_state.json');m=head['model']
assert ex['exit_code']==0 and ex['natural_wait_verified'] and ex['child_pid']==r['pid'] and ex['child_full_argv'][1:]==r['argv'] and ex['original_receipt_sha256']==sha(root/'out/actual_matched_head_fit_receipt.json')
assert sha(root/'out/head_state.json')==r['head_state_sha256'] and sha(root/'out/original_headFIT232_supervised.npz')==r['original_supervised_array_sha256']
assert not r['GPU_used'] and not r['headEVAL201_inputs_or_labels_used'] and not r['all_other_task_label_roles_used'] and not r['new_development_or_benchmark_scores']
with np.load(root/'out/original_headFIT232_supervised.npz',allow_pickle=False) as z:
 ids=z['row_ids'];v=z['video_ids'];f=z['pF'];c=z['pC'];d=z['delta'];y=z['labels'];res=z['residual'];assert np.array_equal(res,y-f) and np.array_equal(c-f,d) and len(ids)==232 and len(np.unique(v))==9
 assert str(z['head_state_sha256'].item())==r['head_state_sha256']
 assert ids.tolist()==head['fit_row_ids']==r['task_label_journal'][0]['row_ids'] and len(r['task_label_journal'])==1
 u=np.asarray([1/(9*np.sum(v==x)) for x in v]);raw=np.column_stack([f,d]);mean=u@raw;sd=np.sqrt(u@((raw-mean)**2));constant=np.all(raw==raw[0],axis=0);mean[constant]=raw[0,constant];sd[constant]=0
 assert np.array_equal(mean,m['mean']) and np.array_equal(sd,m['std'])
 X=np.column_stack([np.ones(232),np.divide(raw-mean,sd,out=np.zeros_like(raw),where=sd>0)]);active=np.r_[True,sd>0];rawW=4*d*d;norm=float(u@rawW);w=u*rawW/norm
 assert np.array_equal(u,m['normalized_U_weights']) and np.array_equal(w,m['normalized_W_weights']) and norm==m['W_global_normalizer'] and float(u@res)==m['constant']
 maxerr=0.;normal_residual={}
 for name,weights in [('U',u),('W',w)]:
  beta=np.asarray(m[name]);A=np.vstack([np.sqrt(weights)[:,None]*X,np.diag([0,.1,.1])]);b=np.r_[np.sqrt(weights)*res,np.zeros(3)]
  independent=np.zeros(3);independent[active]=np.linalg.lstsq(A[:,active],b,rcond=None)[0];err=float(np.max(np.abs(beta-independent)));assert err<1e-10;maxerr=max(maxerr,err)
  normal_residual[name]=float(np.max(np.abs(X.T@(weights*(X@beta-res))+np.diag([0,.01,.01])@beta)));assert normal_residual[name]<1e-10
  h=X@beta
  for label,value in [(name+'_free',f+h),(name+'_interval',f+np.minimum(np.maximum(h,np.minimum(0,d)),np.maximum(0,d))),(name+'_discrete',np.where(d*d-2*d*h<0,c,f))]:assert np.array_equal(z[label],value)
 assert sha(root/'out/before_label_common_design.npz')==r['before_label_common_design_sha256']
 with np.load(root/'out/before_label_common_design.npz',allow_pickle=False) as design:
  for n,value in [('row_ids',ids),('video_ids',v),('X',X),('mean',mean),('std',sd),('normalized_U_weights',u),('normalized_W_weights',w),('W_raw',rawW)]:assert np.array_equal(design[n],value)
 assert np.array_equal(z['constant_free'],f+m['constant'])
record={'status':'ACTUAL_MATCHED_HEADFIT232_ORIGINAL_FULL_STATE_ARRAY_OBJECTIVE_CPU_AUDIT_PASSED_NOT201_EVAL','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid(),'argv':sys.argv,'source_sha256':sha(__file__),'original_fit_receipt_sha256':sha(root/'out/actual_matched_head_fit_receipt.json'),'head_state_sha256':r['head_state_sha256'],'original_supervised_array_sha256':r['original_supervised_array_sha256'],'independent_augmented_lstsq_vs_normal_equation_max_error':maxerr,'normal_equation_residuals':normal_residual,'no_additional_official_label_asset_access':True,'GPU_used':False,'CPU_model_forward':False,'headEVAL201_used':False}
Path(a.out).write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
