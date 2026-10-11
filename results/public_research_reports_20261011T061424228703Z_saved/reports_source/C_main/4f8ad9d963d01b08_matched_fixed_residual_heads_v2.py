"""Source-only refinement: explicit common design frozen before FIT labels."""
import numpy as np
from matched_fixed_residual_heads_v1 import HeadLabels,features,video_equal_weights

def design(pF,delta,videos):
 f,d=map(lambda x:np.asarray(x,dtype=np.float64),(pF,delta));v=np.asarray(videos)
 if any(x.ndim!=1 for x in (f,d,v)) or not f.shape==d.shape==v.shape or len(f)==0 or not np.isfinite(np.column_stack([f,d])).all():raise ValueError('COMMON_FINITE_DESIGN_ROWS')
 a=video_equal_weights(v);raw=np.column_stack([f,d]);mu=a@raw;var=a@((raw-mu)**2);std=np.sqrt(var)
 constant=np.all(raw==raw[0],axis=0);mu[constant]=raw[0,constant];std[constant]=0
 X=features(f,d,mu,std);rawW=4*d*d;c=float(a@rawW)
 if c==0:raise ValueError('EXACT_ZERO_DELTA_CANDIDATE_W_STOP_NO_EPSILON')
 qW=a*rawW/c
 return {'mean':mu,'std':std,'X':X,'normalized_U_weights':a,'normalized_W_weights':qW,'W_raw':rawW,'W_global_normalizer':c,'effective_features':np.r_[True,std>0],'W_exact_algebraic_duplicate':bool(np.all(np.abs(d)==abs(d[0])))}

def fit(pF,delta,labels,videos,common_design,ridge=.01):
 if ridge!=.01:raise ValueError('NO_LAMBDA_SEARCH')
 actual=design(pF,delta,videos)
 for name,val in actual.items():
  if not np.array_equal(val,common_design[name]):raise ValueError('COMMON_DESIGN_CHANGED_AFTER_LABEL_ACCESS:'+name)
 f,y=map(lambda a:np.asarray(a,dtype=np.float64),(pF,labels))
 if y.shape!=f.shape or not np.isfinite(y).all():raise ValueError('FINITE_SAME_FIT_LABEL_ROWS')
 r=y-f;X=actual['X'];active=actual['effective_features'];P=np.diag([0.,ridge,ridge]);m=dict(actual);diagnostics={}
 for name,w in [('U',actual['normalized_U_weights']),('W',actual['normalized_W_weights'])]:
  if name=='W' and actual['W_exact_algebraic_duplicate']:theta=m['U'].copy()
  else:
   A=X.T@(w[:,None]*X)+P;b=X.T@(w*r);theta=np.zeros(3);theta[active]=np.linalg.solve(A[np.ix_(active,active)],b[active])
  m[name]=theta;e=X@theta-r;data=float(w@(e*e));penalty=float(ridge*np.dot(theta[1:],theta[1:]))
  diagnostics[name]={'data_loss':data,'slope_penalty':penalty,'total_objective':data+penalty,'weighted_residual_mean':float(w@e),'normal_equation_max_residual':float(np.max(np.abs(X.T@(w*e)+P@theta))),
      'weighted_feature_mean':(w@X[:,1:]).tolist(),'weighted_target_mean':float(w@r),'intercept_weighted_normal_equation_error':float(w@e)}
 m.update(constant=float(actual['normalized_U_weights']@r),ridge=ridge,fit_objective_diagnostics=diagnostics,shared_standardization_weights='one pre-label FIT videoequal design shared; only loss weights differ')
 return m

def predict(model,pF,delta,pC):
 f,d,c=map(lambda x:np.asarray(x,dtype=np.float64),(pF,delta,pC))
 if not np.array_equal(c-f,d):raise ValueError('SAVED_REAL_PC_AND_DELTA_ALIGNMENT')
 X=features(f,d,model['mean'],model['std']);result={'F':f.copy(),'C':c.copy(),'constant_free':f+model['constant']}
 for name in ('U','W'):
  h=X@model[name];result[name+'_free']=f+h;result[name+'_interval']=f+np.clip(h,np.minimum(0,d),np.maximum(0,d));result[name+'_discrete']=np.where(d*d-2*d*h<0,c,f)
 if not all(np.isfinite(x).all() for x in result.values()):raise ValueError('ALL_READOUTS_FINITE')
 return result
