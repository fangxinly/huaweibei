"""Synthetic NumPy preparation checks; no official arrays, labels or GPU."""
import hashlib,json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from matched_fixed_residual_heads_v2 import design,fit,predict

rng=np.random.default_rng(91819)
v=np.repeat(np.arange(9),[2,3,4,5,6,7,8,9,10]);f=rng.normal(size=len(v));c=f+rng.normal(scale=.3,size=len(v));d=c-f
y=f+.13+.2*f-.3*d+rng.normal(scale=.1,size=len(v))
common=design(f,d,v);m=fit(f,d,y,v,common);X=common['X'];r=y-f;errors=[]
for n,q in [('U',common['normalized_U_weights']),('W',common['normalized_W_weights'])]:
 A=np.vstack([np.sqrt(q)[:,None]*X,np.diag([0,.1,.1])]);b=np.r_[np.sqrt(q)*r,np.zeros(3)]
 independent=np.linalg.lstsq(A,b,rcond=None)[0];errors.append(float(np.max(np.abs(independent-m[n]))))
 assert errors[-1]<1e-12
 assert abs(float(q@(X@m[n]-r)))<1e-12
 diag=m['fit_objective_diagnostics'][n]
 assert abs(diag['total_objective']-(float(q@((X@m[n]-r)**2))+.01*float(m[n][1:]@m[n][1:])))<1e-14
rawq=d*d-2*d*r;estimated=d*d-2*d*(X@m['W']);a=common['normalized_U_weights'];scale=common['W_global_normalizer']
identity_error=abs(float(a@((estimated-rawq)**2))-scale*m['fit_objective_diagnostics']['W']['data_loss'])
assert identity_error<1e-13
raw_regularized=float(a@((estimated-rawq)**2))+scale*.01*float(m['W'][1:]@m['W'][1:])
assert abs(raw_regularized-scale*m['fit_objective_diagnostics']['W']['total_objective'])<1e-13
outputs=predict(m,f,d,c);assert len(outputs)==9
for n in ('U','W'):
 h=X@m[n];assert np.array_equal(outputs[n+'_discrete'],np.where(d*d-2*d*h<0,c,f))
 assert np.all(outputs[n+'_interval']>=np.minimum(f,c)) and np.all(outputs[n+'_interval']<=np.maximum(f,c))
stored={k:(x.tolist() if isinstance(x,np.ndarray) else x) for k,x in m.items()}
with tempfile.TemporaryDirectory() as directory:
 p=Path(directory)/'head.json';p.write_text(json.dumps(stored));reloaded=json.loads(p.read_text())
 assert all(np.array_equal(outputs[k],predict(reloaded,f,d,c)[k]) for k in outputs)
 # Neither future entry point may advance past an incomplete protocol.
 bundle=Path(directory)/'bundle';bundle.mkdir();(bundle/'matched_head_execution_plan.json').write_text(json.dumps({'status':'LOCAL_PREPARATION_ONLY'}))
 for source,extra in [('fit_matched_fixed_heads_original_v1.py',['--out',str(Path(directory)/'out')]),('matched_head_fit_stage_wrapper_v1.py',['--root',str(Path(directory)/'run')])]:
  args=[sys.executable,str(Path(__file__).parent/source),'--bundle',str(bundle),'--asset-base','MUST_NOT_OPEN','--candidate-arrays','MUST_NOT_OPEN','--parent-joint','MUST_NOT_OPEN','--evidence','MUST_NOT_OPEN']+extra
  result=subprocess.run(args,capture_output=True,text=True)
  assert result.returncode!=0 and 'PROTOCOL_NOT_READY' in result.stderr and not (Path(directory)/'out').exists() and not (Path(directory)/'run').exists()
# Exact abs(delta) constant: preserve declared algebraic duplicate.
f2=np.zeros(len(v));d2=np.where(np.arange(len(v))%2,.125,-.125);c2=d2.copy();y2=rng.normal(size=len(v))
m2=fit(f2,d2,y2,v,design(f2,d2,v));assert np.array_equal(m2['U'],m2['W']) and m2['std'][0]==0 and m2['U'][1]==0
f3=np.full(len(v),.2);d3=np.full(len(v),.125);m3=fit(f3,d3,y2,v,design(f3,d3,v));assert np.array_equal(m3['U'][1:],np.zeros(2))
negative_checks=0
bad={k:(val.copy() if isinstance(val,np.ndarray) else val) for k,val in common.items()};bad['X'][0,0]=2
for call in [lambda:fit(f,d,y,v,bad),lambda:fit(f,d,y,v,common,ridge=.02),lambda:design(f,np.zeros_like(d),v),lambda:predict(m,f,d,c+.01)]:
 try:call()
 except (ValueError,PermissionError):negative_checks+=1
 else:raise AssertionError('EXPECTED_FIXED_PROTOCOL_REJECTION')
record={'status':'SYNTHETIC_PREPARATION_CHECKS_PASSED_NOT_TASK_EXPERIMENT','independent_augmented_lstsq_max_error':max(errors),'structured_Q_data_identity_error':identity_error,'physical_JSON_head_replay_error':0,'negative_core_checks':negative_checks,'incomplete_protocol_entrypoints_rejected':2,'official_task_data_or_labels_accessed':False,'GPU_or_remote_used':False,'headFIT232_fit_or_headEVAL201_prediction_executed':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(__file__).with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
