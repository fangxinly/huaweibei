"""Only audited complete DEV arrays; this is not a GPU intervention runner."""
from pathlib import Path
import argparse,datetime,hashlib,io,json,zipfile
import numpy as np
PAIRS=((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))
NAMES=('T<-A','T<-V','A<-T','A<-V','V<-T','V<-A')
def digest(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def rank(x):
    order=np.argsort(x,kind='stable');v=x[order];r=np.empty(len(x),float)
    i=0
    while i<len(x):
        j=i+1
        while j<len(x) and v[j]==v[i]:j+=1
        r[order[i:j]]=(i+j-1)/2.;i=j
    return r
def corr(a,b):
    if np.var(a)==0 or np.var(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])
def smooth(a,b):
    d=np.abs(a-b);return float(np.where(d<1,.5*d*d,d-.5).mean())
def errors(pred,y):return {'MAE':float(np.abs(pred-y).mean()),'MSE':float(np.square(pred-y).mean())}
def calibration(u,q):
    t=np.tanh(q/.5);positive=q>1e-10;negative=q< -1e-10;nonzero=positive|negative
    pos=float((u[positive]>0).mean()) if positive.any() else None
    neg=float((u[negative]<0).mean()) if negative.any() else None
    return {'q_mean':float(q.mean()),'q_abs_mean':float(np.abs(q).mean()),'positive_fraction':float(positive.mean()),'negative_fraction':float(negative.mean()),
        'target_zero_count':int((~nonzero).sum()),'predicted_zero_count':int((np.abs(u)<=1e-10).sum()),
        'sign_accuracy_nonzero_q':float((np.sign(u[nonzero])==np.sign(q[nonzero])).mean()) if nonzero.any() else None,
        'positive_recall':pos,'negative_recall':neg,'balanced_sign_recall':(pos+neg)/2 if pos is not None and neg is not None else None,
        'smooth_l1':smooth(u,t),'zero_smooth_l1':smooth(np.zeros_like(u),t),'MSE_to_target':float(np.square(u-t).mean()),
        'zero_MSE_to_target':float(np.square(t).mean()),'spearman_u_q':corr(rank(u),rank(q)),'pearson_u_q':corr(u,q)}
def analyze(z):
    y=z['valid_y'].astype(float);on=z['valid_pred'].astype(float);off=z['condition_off_pred'].astype(float)
    own=z['own'].astype(float);pair=z['pair'].astype(float);u=z['utility'].astype(float);w=z['predicted_weights'].astype(float)
    assert y.shape==on.shape==off.shape==(229,) and own.shape==(229,3)
    assert pair.shape==u.shape==w.shape==(229,6)
    assert all(np.isfinite(a).all() for a in [y,on,off,own,pair,u,w])
    assert np.allclose(w,1/(1+np.exp(-4*u)),atol=1e-7,rtol=1e-7)
    q=np.stack([np.square(own[:,m]-y)-np.square(pair[:,i]-y) for i,(m,j) in enumerate(PAIRS)],1)
    directions=[]
    for i,name in enumerate(NAMES):
        bins=[]
        for b in range(5):
            lo,hi=b/5,(b+1)/5;mask=(w[:,i]>=lo)&((w[:,i]<hi) if b<4 else (w[:,i]<=hi))
            bins.append({'lower':lo,'upper':hi,'count':int(mask.sum()),'u_mean':float(u[mask,i].mean()) if mask.any() else None,
                'target_mean':float(np.tanh(q[mask,i]/.5).mean()) if mask.any() else None,'q_mean':float(q[mask,i].mean()) if mask.any() else None})
        directions.append({'direction':name,'pair_teacher':errors(pair[:,i],y),'calibration':calibration(u[:,i],q[:,i]),'fixed_weight_bins':bins})
    ordering=[]
    for m,name in enumerate(('T','A','V')):
        d=q[:,2*m]-q[:,2*m+1];v=u[:,2*m]-u[:,2*m+1];use=np.abs(d)>1e-10
        ordering.append({'receiver':name,'target_ties':int((~use).sum()),'prediction_ties':int((np.abs(v)<=1e-10).sum()),
            'agreement_excluding_target_ties':float((np.sign(d[use])==np.sign(v[use])).mean()) if use.any() else None})
    delta=on-off;e=off-y;benefit=np.square(off-y)-np.square(on-y);identity=-2*e*delta-delta*delta
    residual=float(np.max(np.abs(benefit-identity)));assert residual<1e-10
    return {'own_teachers':{name:errors(own[:,i],y) for i,name in enumerate(('T','A','V'))},'directions':directions,'donor_ordering':ordering,
        'main_default':errors(on,y),'main_all_off':errors(off,y),'default_retention_benefit_vs_all_off':{'mean':float(benefit.mean()),'positive_fraction':float((benefit>1e-10).mean()),
            'negative_fraction':float((benefit< -1e-10).mean()),'prediction_displacement_rms':float(np.sqrt(np.square(delta).mean())),'risk_identity_maxabs':residual}}
def main():
    a=argparse.ArgumentParser();a.add_argument('--audit',type=Path,required=True);a.add_argument('--directory',type=Path,required=True);a.add_argument('--out',type=Path,required=True);c=a.parse_args()
    audit=json.loads(c.audit.read_text(encoding='utf-8'))
    assert audit['status']=='UTILITY_SELECTED_NODES_COMPLETED_WHOLE_SHA_100_SELECTION_OFFICIAL_DEV_VERIFIED'
    assert not c.out.exists();rows=[]
    for row in audit['rows']:
        assert row['full_checkpoint_verified'] and row['selection']['epochs']==100
        cp=Path(row['checkpoint']);assert cp.stat().st_size==row['checkpoint_bytes'] and digest(cp)==row['checkpoint_sha256']
        archive=c.directory/row['node']/'snapshot.zip';assert digest(archive)==row['archive_sha256']
        with zipfile.ZipFile(archive) as z:
            name='inflow_utility_v4_deployment_20261005T0346Z/run_'+row['mode']+'/predictions.npz';raw=z.read(name)
            assert len(raw)==row['files']['predictions.npz']['bytes'] and hashlib.sha256(raw).hexdigest()==row['files']['predictions.npz']['sha256']
            with np.load(io.BytesIO(raw),allow_pickle=False) as saved:result=analyze(saved)
        rows.append({'node':row['node'],'mode':row['mode'],'checkpoint_sha256':row['checkpoint_sha256'],**result})
    report={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'AUDITED_COMPLETED_DEV_PROXY_CALIBRATION_ANALYZED','audit_sha256':digest(c.audit),'source_sha256':digest(__file__),'rows':rows,
        'limits':['No channel or donor intervention result is generated by this CPU analyzer.','Official DEV used for selection and diagnosis; no held-out confirmation.','Task-head proxy does not establish semantic truth or causal information decomposition.','Single seed with mode-node binding; all six directions retained.']}
    with c.out.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'status':report['status'],'nodes':[r['node'] for r in rows]}))
if __name__=='__main__':main()
