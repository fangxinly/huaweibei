"""TRAIN OOF calibration controls; evaluation only on official VAL/TEST.

Fixed five training-prediction quantile anchors, monotone linear spline q0,
and unconstrained delta regression on the entire SAME spline basis. This is
a finite low-capacity approximation to E[delta|p0], not an independence test.
No polynomial degree/knot/regularization sweep is enabled.
"""
import numpy as np
from scipy.optimize import nnls
from oof_selector import check_lineage, weights
from linear_controls import packet


def anchors(x, w):
    order=np.argsort(x,kind='stable')
    x,w=x[order],w[order]
    cumulative=np.cumsum(w)/w.sum()
    index=np.searchsorted(cumulative,np.array([0.,.25,.5,.75,1.]),side='left')
    return np.unique(x[np.minimum(index,len(x)-1)])


def basis(x, knots):
    x=np.asarray(x,float)
    if len(knots)==1:return np.ones((len(x),1))
    xc=np.clip(x,knots[0],knots[-1])
    j=np.clip(np.searchsorted(knots,xc,side='right')-1,0,len(knots)-2)
    t=(xc-knots[j])/(knots[j+1]-knots[j])
    B=np.zeros((len(x),len(knots)))
    B[np.arange(len(x)),j]=1-t
    B[np.arange(len(x)),j+1]=t
    return B


def monotone_fit(B,y,w):
    # Values at knots are v0, v0+s1, v0+s1+s2, ...; all increments >=0.
    if B.shape[1]==1:return np.array([float(w@y)])
    C=np.column_stack([B[:,j:].sum(1) for j in range(1,B.shape[1])])
    mean=w@C
    centered=C-mean
    slope,_=nnls(centered*np.sqrt(w)[:,None],(y-w@y)*np.sqrt(w))
    intercept=float(w@y-mean@slope)
    return intercept+np.r_[0.,np.cumsum(slope)]


class FlexibleControls:
    def fit(self,y,p0,p1,ids,official_train_ids,calibration_ids,lineage):
        check_lineage(ids,official_train_ids,calibration_ids,lineage)
        p0,p1=packet(p0,p1)
        y=np.asarray(y,float)
        if y.shape!=p0.shape or len(ids)!=len(p0) or not np.isfinite(y).all():
            raise ValueError('Finite genuine OOF training labels required')
        w=weights(ids)
        self.knots=anchors(p0,w)
        B=basis(p0,self.knots)
        self.values=monotone_fit(B,y,w)
        root=np.sqrt(w)
        delta=p1-p0
        self.delta_coef,_,self.rank,self.singular=np.linalg.lstsq(B*root[:,None],delta*root,rcond=None)
        i=delta-B@self.delta_coef
        q0=B@self.values
        _,_,rank_aug,s_aug=np.linalg.lstsq(np.column_stack((B,i))*root[:,None],y*root,rcond=None)
        self.rank_aug=int(rank_aug)
        self.augmented_singular=s_aug
        den=float(w@(i*i))
        self.gamma=float(w@((y-q0)*i)/den) if rank_aug>self.rank else 0.
        self.orthogonality_maxerror=float(abs(B.T@(w*i)).max())
        self.innovation_std=float(np.sqrt(den))
        self.effect_std=abs(self.gamma)*self.innovation_std
        self.source_ids=list(map(str,ids))
        self.calibration_ids=list(map(str,calibration_ids))
        self.lineage=lineage
        self.fold_stability=[]
        for n,fold in enumerate(lineage):
            lookup=set(map(str,fold['held_ids']))
            ix=np.array([str(s) in lookup for s in ids])
            ids_f=[str(s) for s in ids if str(s) in lookup]
            wf=weights(ids_f)
            A=np.column_stack((np.ones(ix.sum()),p0[ix]))
            coef=np.linalg.lstsq(A*np.sqrt(wf)[:,None],y[ix]*np.sqrt(wf),rcond=None)[0]
            kf=anchors(p0[ix],wf)
            vf=monotone_fit(basis(p0[ix],kf),y[ix],wf)
            self.fold_stability.append(dict(fold=n,rows=int(ix.sum()),affine_intercept_slope=coef.tolist(),
                                            monotone_knots=kf.tolist(),monotone_values=vf.tolist(),
                                            diagnostic_only_not_deployment_parameters=True))
        self.fitted=True
        return self

    def predict(self,p0,p1):
        if not getattr(self,'fitted',False):raise ValueError('Genuine TRAIN OOF fit required')
        p0,p1=packet(p0,p1)
        B=basis(p0,self.knots)
        q0=B@self.values
        i=p1-p0-B@self.delta_coef
        return dict(monotone_calibration_only=q0,
                    monotone_calibration_plus_orthogonal_delta=q0+self.gamma*i,
                    remaining_delta=i)

    def state(self):
        if not getattr(self,'fitted',False):raise ValueError('No fitted state')
        return dict(knots=self.knots.tolist(),monotone_values=self.values.tolist(),
                    delta_conditional_regression_coef=self.delta_coef.tolist(),gamma=self.gamma,
                    weighted_basis_orthogonality_maxerror=self.orthogonality_maxerror,
                    delta_regression_rank=int(self.rank),augmented_rank=self.rank_aug,
                    delta_singular=self.singular.tolist(),augmented_singular=self.augmented_singular.tolist(),
                    innovation_std=self.innovation_std,effect_std=self.effect_std,
                    fold_stability=self.fold_stability,fit_ids=self.source_ids,
                    calibration_ids=self.calibration_ids,extrapolation='clip input to TRAIN anchor support',
                    nonlinear_conditionally_independent_proved=False,risk_guarantee=False)


def official_five(y,p,ids,role):
    if role not in ('VAL','TEST'):raise ValueError('Performance report accepts only official VAL/TEST')
    y,p=np.asarray(y,float),np.asarray(p,float)
    if y.ndim!=1 or p.shape!=y.shape or len(ids)!=len(y) or not len(y):raise ValueError('Same official rows required')
    if len(set(map(str,ids)))!=len(ids) or not np.isfinite(y).all() or not np.isfinite(p).all():raise ValueError('Unique finite official arrays required')
    keep=y!=0
    if not keep.any():raise ValueError('Acc2/F1 require nonzero truth samples')
    yt,pt=y[keep]>=0,p[keep]>=0
    f1=0.
    for cls in (False,True):
        tp=np.sum((yt==cls)&(pt==cls))
        fp=np.sum((yt!=cls)&(pt==cls))
        fn=np.sum((yt==cls)&(pt!=cls))
        den=2*tp+fp+fn
        f1+=np.mean(yt==cls)*(2*tp/den if den else 0.)
    corr=float(np.corrcoef(y,p)[0,1]) if np.std(y)>0 and np.std(p)>0 else None
    return dict(role=role,rows=len(y),Acc7=float(np.mean(np.round(np.clip(y,-3,3))==np.round(np.clip(p,-3,3)))),
                Acc2=float(np.mean(yt==pt)),F1=float(f1),MAE=float(np.mean(abs(y-p))),Corr=corr,
                MSE=float(np.mean((y-p)**2)))
