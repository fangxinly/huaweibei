"""Common FIT-only features and fixed U/W residual ridge objectives, NumPy only."""
import hashlib
from pathlib import Path
import numpy as np

def video_equal_weights(videos):
    v=np.asarray(videos)
    if v.ndim!=1 or len(v)==0:raise ValueError('NONEMPTY_VIDEO_IDS')
    unique,counts=np.unique(v,return_counts=True)
    return np.asarray([1/(len(unique)*counts[np.where(unique==x)[0][0]]) for x in v],dtype=np.float64)

def features(pF,delta,mean=None,std=None):
    x=np.column_stack([np.asarray(pF,dtype=np.float64),np.asarray(delta,dtype=np.float64)])
    if x.ndim!=2 or x.shape[1]!=2 or not np.isfinite(x).all():raise ValueError('TWO_FINITE_FEATURES')
    if mean is None or std is None:raise ValueError('FROZEN_FIT_STATISTICS_REQUIRED')
    active=np.asarray(std)>0
    z=np.zeros_like(x);z[:,active]=(x[:,active]-np.asarray(mean)[active])/np.asarray(std)[active]
    return np.column_stack([np.ones(len(x)),z])

def fit(pF,delta,labels,videos,ridge=.01):
    if ridge!=.01:raise ValueError('NO_LAMBDA_SEARCH')
    f,d,y=map(lambda a:np.asarray(a,dtype=np.float64),(pF,delta,labels))
    if any(a.ndim!=1 for a in (f,d,y)) or not (f.shape==d.shape==y.shape==np.asarray(videos).shape):raise ValueError('FIXED_ROW_ALIGNMENT')
    if not np.isfinite(np.column_stack([f,d,y])).all():raise ValueError('NO_NONFINITE_ROW_DROPPING')
    a=video_equal_weights(videos);raw=np.column_stack([f,d]);mean=a@raw
    variance=a@((raw-mean)**2);std=np.sqrt(variance)
    # Exact constant input must remain structurally zero even when a weighted
    # mean accumulates a rounding residual for a non-binary decimal constant.
    constant=np.all(raw==raw[0],axis=0);std[constant]=0;mean[constant]=raw[0,constant]
    X=features(f,d,mean,std);residual=y-f;active=np.r_[True,std>0]
    penalty=np.diag([0.,ridge,ridge]);solutions={};qraw=4*d*d;c=float(a@qraw)
    if c==0:raise ValueError('EXACT_ZERO_DELTA_CANDIDATE_W_STOP_NO_EPSILON')
    redundant=bool(np.all(np.abs(d)==abs(d[0])))
    for name,w in [('U',a),('W',a*qraw/c)]:
        if name=='W' and redundant:solutions[name]=solutions['U'].copy();continue
        A=X.T@(w[:,None]*X)+penalty;b=X.T@(w*residual);beta=np.zeros(3)
        beta[active]=np.linalg.solve(A[np.ix_(active,active)],b[active]);solutions[name]=beta
    return {'mean':mean,'std':std,'U':solutions['U'],'W':solutions['W'],'constant':float(a@residual),
        'W_global_normalizer':c,'W_exact_algebraic_duplicate':redundant,'effective_features':active,
        'shared_standardization_weights':'video equal U; frozen for both U/W','ridge':ridge,
        'normalized_U_weights':a,'normalized_W_weights':a*qraw/c}

def predict(model,pF,delta):
    f,d=map(lambda x:np.asarray(x,dtype=np.float64),(pF,delta));X=features(f,d,model['mean'],model['std'])
    result={'F':f.copy(),'C':f+d,'constant_free':f+model['constant']}
    for name in ('U','W'):
        h=X@model[name]
        result[name+'_free']=f+h
        result[name+'_interval']=f+np.clip(h,np.minimum(0,d),np.maximum(0,d))
        # This form avoids squaring two large terms; exact Q tie selects F.
        choose=(d*d-2*d*h)<0
        result[name+'_discrete']=f+np.where(choose,d,0.)
    if not all(np.isfinite(x).all() for x in result.values()):raise ValueError('NONFINITE_FROZEN_READOUT')
    return result

class HeadLabels:
    """Row-only label access; evaluation requires complete frozen prediction proof."""
    def __init__(self,examples,fit_rows,eval_rows):
        self.examples=examples;self.fit_rows=np.asarray(fit_rows);self.eval_rows=np.asarray(eval_rows);self.journal=[];self.eval_used=False
        if len(examples)!=1281 or len(self.fit_rows)!=232 or len(self.eval_rows)!=201 or set(self.fit_rows)&set(self.eval_rows):raise ValueError('FROZEN_HEAD_ROLES')
    def get_fit(self,rows):
        if not np.array_equal(rows,self.fit_rows):raise PermissionError('ONLY_EXACT_HEADFIT232_ORDER')
        y=np.asarray([np.asarray(self.examples[int(i)][1]).reshape(-1)[0] for i in rows],dtype=np.float64)
        self.journal.append({'role':'headFIT232','row_ids':rows.tolist(),'labels_read':True});return y
    def get_eval(self,rows,predictions_path,predictions_sha,head_state_path,head_state_sha):
        if self.eval_used or not np.array_equal(rows,self.eval_rows):raise PermissionError('ONE_EXACT_HEADEVAL201_ORDER')
        for p,h in [(predictions_path,predictions_sha),(head_state_path,head_state_sha)]:
            if len(h)!=64 or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h:raise PermissionError('ACTUAL_DISK_PREDICTION_AND_HEAD_SHA_REQUIRED')
        expected={'F','C','constant_free','U_free','U_interval','U_discrete','W_free','W_interval','W_discrete'}
        with np.load(predictions_path,allow_pickle=False) as z:
            if set(z.files)!=expected|{'row_ids','head_state_sha256'} or not np.array_equal(z['row_ids'],rows) or str(z['head_state_sha256'].item())!=head_state_sha:raise PermissionError('COMPLETE_FROZEN_NINE_READOUT_SCHEMA')
            if any(z[n].shape!=(201,) or not np.isfinite(z[n]).all() for n in expected):raise PermissionError('ALL201_FINITE_NO_ROW_DROPPING')
        # A separate source-pinned runner must enforce head-before-inference
        # chronology and evidence provenance. This helper alone cannot launch.
        self.eval_used=True;y=np.asarray([np.asarray(self.examples[int(i)][1]).reshape(-1)[0] for i in rows],dtype=np.float64)
        self.journal.append({'role':'headEVAL201','row_ids':rows.tolist(),'labels_read':True,'purpose':'one development evaluation; no model or readout selection'});return y
