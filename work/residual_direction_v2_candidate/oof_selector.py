"""Low-capacity residual estimator and exact-mask utility selection.

The lineage checks validate supplied source-qualified records, not their external
authenticity. Real operators must also link original SHA/capture/argv receipts.
No empirical no-harm guarantee follows from a nonnegative estimated utility.
"""
import numpy as np

PREDECESSORS = ('encoder','normalization','flow','proposal','feature_transform')

def video(row_id): return str(row_id).split('[')[0]

def check_lineage(ids, official_train_ids, calibration_ids, lineage):
    ids = list(map(str,ids));allowed = set(map(str,official_train_ids))
    cal = set(map(str,calibration_ids));cal_v = {video(s) for s in cal}
    if len(ids)!=len(set(ids)) or not set(ids)<=allowed or not cal<=allowed:
        raise ValueError('Unique official TRAIN rows required')
    if {video(s) for s in ids}&cal_v:
        raise ValueError('Residual training and calibration video overlap')
    seen=[]
    for fold in lineage:
        held=list(map(str,fold['held_ids']));held_v={video(s) for s in held}
        if not held or not set(held)<=set(ids):raise ValueError('Unknown or empty held rows')
        # All rows of a video in this packet must be held together.
        if set(held)!={s for s in ids if video(s) in held_v}:
            raise ValueError('A held video was split across roles')
        if set(fold['predecessors'])!=set(PREDECESSORS):
            raise ValueError('Every learned predecessor must have exclusion evidence')
        for name in PREDECESSORS:
            rec=fold['predecessors'][name]
            train=list(map(str,rec['train_ids']))
            if not set(train)<=allowed or {video(s) for s in train}&(held_v|cal_v):
                raise ValueError('Predecessor TRAIN/held/CAL video leakage: '+name)
            h=rec['state_SHA']
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError('Missing pinned predecessor state SHA')
        seen.extend(held)
    if sorted(seen)!=sorted(ids) or len(lineage)<2:
        raise ValueError('Every residual training row requires exactly one OOF prediction')

def weights(ids):
    videos=np.asarray([video(s) for s in ids]);unique,count=np.unique(videos,return_counts=True)
    population=dict(zip(unique,count))
    return np.asarray([1/(len(unique)*population[v]) for v in videos])

class ResidualSelector:
    RIDGE = 1.0   # Fixed normalized population objective; no lambda sweep.
    MAX_FEATURES = 24
    def __init__(self):self.slope=0.;self.fitted=False;self.calibrated=False

    def fit(self,x,y,p0,ids,official_train_ids,calibration_ids,lineage):
        check_lineage(ids,official_train_ids,calibration_ids,lineage)
        x=np.asarray(x,float);y=np.asarray(y,float);p0=np.asarray(p0,float)
        if x.ndim!=2 or x.shape[0]!=len(ids) or not 0<x.shape[1]<=self.MAX_FEATURES:
            raise ValueError('Explicit low-dimensional OOF features and scalar predictions required')
        if y.shape!=(len(ids),) or p0.shape!=(len(ids),) or not all(np.isfinite(v).all() for v in (x,y,p0)):
            raise ValueError('Finite same-row OOF arrays required')
        w=weights(ids);self.mean=w@x
        scale=np.sqrt(w@((x-self.mean)**2));self.scale=np.where(scale>0,scale,1.)
        z=(x-self.mean)/self.scale;r=y-p0;intercept=float(w@r)
        self.coef=np.linalg.solve(z.T@(w[:,None]*z)+self.RIDGE*np.eye(x.shape[1]),z.T@(w*(r-intercept)))
        self.intercept=intercept;self.fitted=True;self.training_videos={video(s) for s in ids}
        self.calibration_ids=list(map(str,calibration_ids));self.lineage=lineage;self.allowed=set(map(str,official_train_ids))
        return self

    def residual(self,x):
        if not self.fitted:raise ValueError('No genuine OOF fit supplied')
        x=np.asarray(x,float)
        if x.ndim!=2 or x.shape[1]!=len(self.coef) or not np.isfinite(x).all():
            raise ValueError('Finite fixed feature schema required')
        return ((x-self.mean)/self.scale)@self.coef+self.intercept

    def calibrate(self,x,y,p0,ids,final_predecessors):
        if list(map(str,ids))!=self.calibration_ids or {video(s) for s in ids}&self.training_videos:
            raise ValueError('Only predeclared separate TRAIN calibration videos allowed')
        cal_v={video(s) for s in ids}
        if set(final_predecessors)!=set(PREDECESSORS):raise ValueError('Final deployment lineage required')
        for rec in final_predecessors.values():
            if not set(map(str,rec['train_ids']))<=self.allowed or {video(s) for s in rec['train_ids']}&cal_v:
                raise ValueError('Final feature/flow model has seen calibration videos')
            h=rec['state_SHA']
            if len(h)!=64 or any(c not in '0123456789abcdef' for c in h):raise ValueError('Final state SHA required')
        rh=self.residual(x);rho=np.asarray(y,float)-np.asarray(p0,float);w=weights(ids)
        if rho.shape!=rh.shape or not np.isfinite(rho).all():raise ValueError('Calibration row mismatch')
        denominator=float(w@(rh*rh))
        self.slope=max(0.,float(w@(rh*rho))/denominator) if denominator>0 else 0.
        self.calibrated=True
        return self.slope

    def choose(self,x,delta):
        delta=np.asarray(delta,float)
        if delta.ndim!=2 or delta.shape!=(len(x),8) or not np.isfinite(delta).all() or not np.array_equal(delta[:,0],np.zeros(len(delta))):
            raise ValueError('Exactly OFF/six individual/ALL actual-flow deltas required')
        if self.fitted and self.calibrated:rh=self.slope*self.residual(x)
        else:rh=np.zeros(len(delta))
        utility=2*rh[:,None]*delta-delta*delta
        index=np.argmax(utility,axis=1) # OFF first makes all ties conservative.
        return dict(index=index,estimated_utility=utility[np.arange(len(delta)),index],
                    residual=rh,unfitted_or_uncalibrated_returns_OFF=not(self.fitted and self.calibrated),
                    actual_risk_guarantee=False)

