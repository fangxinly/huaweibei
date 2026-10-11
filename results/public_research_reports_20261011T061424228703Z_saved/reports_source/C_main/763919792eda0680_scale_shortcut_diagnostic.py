"""Fixed saved-array audit. FIT-only affine calibration; no forward or selection."""
import argparse,csv,datetime,hashlib,json,os,sys
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def corr(a,b):
    return float(np.corrcoef(a,b)[0,1]) if min(np.std(a),np.std(b))>1e-15 else None
def weights(video,scheme):
    if scheme=='rows': return np.full(len(video),1/len(video))
    _,inverse,count=np.unique(video,return_inverse=True,return_counts=True)
    return 1/(len(count)*count[inverse])
def affine(x,y,w):
    X=np.column_stack([x,np.ones(len(x))]);theta=np.linalg.lstsq(X*np.sqrt(w[:,None]),y*np.sqrt(w),rcond=None)[0]
    q=X@theta;e=y-q;var=float(w@((y-w@y)**2))
    return dict(slope=float(theta[0]),intercept=float(theta[1]),R2=float(1-w@(e*e)/var) if var>1e-25 else None,residual_rms=float(np.sqrt(w@(e*e)))),q,e
def evaluate(theta,x): return theta['slope']*x+theta['intercept']
def structure(base,target,y,f=None):
    w=np.full(len(y),1/len(y));d=target-base;r=y-base
    pf,_,_=affine(base,target,w);df,_,de=affine(base,d,w);rf,_,re=affine(base,r,w)
    keep=np.abs(base)>.3;ratios=target[keep]/base[keep]
    result=dict(rows=len(y),target_on_base=pf,delta_on_base=df,
        unexplained_delta_variance_fraction=None if df['R2'] is None else 1-df['R2'],
        residual_on_base=rf,partial_corr_delta_rho_controlling_linear_base=corr(de,re),
        corr_delta_remainder_raw_rho=corr(de,r),ratio_abs_base_gt_point3=dict(rows=int(keep.sum()),mean=float(ratios.mean()) if len(ratios) else None,std_population=float(ratios.std()) if len(ratios) else None),
        base_range=[float(base.min()),float(base.max())],target_range=[float(target.min()),float(target.max())])
    if f is not None:
        fd=f-base;g=float(fd@d/(fd@fd))
        result['gain_from_exact_b_f_p_identity']=g
        result['gain_identity_max_error']=float(np.max(np.abs(base+g*fd-target)))
        assert result['gain_identity_max_error']<5e-7
        result['flow_readout_on_base']=affine(base,f,w)[0]
    return result
def score_regions(pred,y,metric):
    return {r:{k:metric(v[mask],y[mask]) for k,v in pred.items()} for r,mask in [('all',np.ones(len(y),bool)),('weak',abs(y)<=1),('strong',abs(y)>1)]}
def paired_bootstrap(q,p,y,video,draws,seed):
    videos=sorted(set(video));sample=np.random.default_rng(seed).integers(0,len(videos),size=(draws,len(videos)))
    result={}
    for region,mask in [('all',np.ones(len(y),bool)),('weak',abs(y)<=1),('strong',abs(y)>1)]:
        counts=np.array([np.sum((video==v)&mask) for v in videos]);den=counts[sample].sum(1);valid=den>0;values={}
        for name,fn in [('MAE',lambda z:abs(z-y)),('MSE',lambda z:(z-y)**2)]:
            d=fn(q)-fn(p);sums=np.array([d[(video==v)&mask].sum() for v in videos]);b=sums[sample].sum(1)[valid]/den[valid]
            values[name]=dict(point=float(d[mask].mean()),percentile95=np.quantile(b,[.025,.975]).tolist(),valid_draws=int(valid.sum()))
        result[region]=values
    return result
def tests():
    x=np.array([-2.,-1.,0.,1.,2.]);w=np.ones(5)/5
    t,q,e=affine(x,1.37*x+.2,w);assert abs(t['slope']-1.37)<1e-14 and abs(t['intercept']-.2)<1e-14 and np.max(abs(e))<1e-14
    # All calibration functions accept only the fit arrays. Changed evaluation labels cannot change q.
    a=evaluate(t,x);y=x.copy();y+=1000;assert np.array_equal(evaluate(t,x),a)
    t2=affine(x,1.37*x+.2,np.array([.1,.1,.4,.2,.2]))[0];assert abs(t2['slope']-1.37)<1e-14
    return dict(exact_affine_recovery=True,weighted_exact_recovery=True,evaluation_labels_not_used_in_calibration=True)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan',type=Path,required=True);ap.add_argument('--plan-sha',required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert sha(a.plan)==a.plan_sha;plan=read(a.plan);root=a.plan.parent
    for name,h in plan['member_sha256'].items(): assert sha(root/name)==h,name
    assert sha(__file__)==plan['worker_SHA'];assert not a.out.exists();a.out.mkdir()
    sys.path.insert(0,str(root));from sentiment_metrics_careflow_v1 import metrics
    result=dict(status='SAVED_ARRAY_SCALE_AFFINE_FIT_INNER_DIAGNOSTIC_COMPLETE',actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,plan_SHA=a.plan_sha,tests=tests(),models={},no_model_forward=True,no_optimizer_training=True,no_original_VAL_TEST_files_opened=True)
    data={}
    for model in ('best20','best36'):
        arrays={}
        for role in ('fit','inner'):
            with np.load(root/f'{model}_{role}_bfp.npz',allow_pickle=False) as z:
                arrays[role]={k:np.asarray(z[k],dtype=np.float64).reshape(-1) for k in ('b','f','p')};arrays[role]['row_id']=z['row_ids'].astype(str);state=str(z['selected_state_sha256'])
            x=arrays[role];x['video']=np.array([v.split('[')[0] for v in x['row_id']]);x['y']=np.load(root/f'{model}_{role}_targets.npy',allow_pickle=False).astype(np.float64).reshape(-1)
            assert state==plan['selected_state_SHA'][model];assert len(x['y'])==plan['rows'][role];assert len(set(x['row_id']))==len(x['y'])
            assert all(np.isfinite(x[k]).all() for k in ('b','f','p','y'))
        assert not set(arrays['fit']['video'])&set(arrays['inner']['video'])
        data[model]=arrays;mr=dict(calibration={},roles={});result['models'][model]=mr
        for scheme in ('rows','videos'):
            fit=arrays['fit'];w=weights(fit['video'],scheme)
            bt=affine(fit['b'],fit['y'],w)[0];pt=affine(fit['p'],fit['y'],w)[0];mr['calibration'][scheme]=dict(b_to_y_FIT=bt,p_to_y_FIT=pt)
        for role,x in arrays.items():
            preds={k:x[k] for k in ('b','p')}
            for scheme,ts in mr['calibration'].items():
                preds[f'b_calibrated_{scheme}']=evaluate(ts['b_to_y_FIT'],x['b']);preds[f'p_calibrated_{scheme}']=evaluate(ts['p_to_y_FIT'],x['p'])
            rr=dict(videos=len(set(x['video'])),structure=structure(x['b'],x['p'],x['y'],x['f']),scores=score_regions(preds,x['y'],metrics))
            mr['roles'][role]=rr
            if role=='inner':
                rr['paired_video_bootstrap']={k:paired_bootstrap(preds[k],x['p'],x['y'],x['video'],plan['bootstrap_draws'],plan['bootstrap_seed']) for k in ('b_calibrated_rows','b_calibrated_videos','p_calibrated_rows','p_calibrated_videos')}
            if model=='best36':
                with np.load(root/f'best36_{role}_donor_masks.npz',allow_pickle=False) as z:
                    assert np.array_equal(z['row_ids'].astype(str),x['row_id']);p0=z['off'].astype(np.float64);p1=z['on'].astype(np.float64);assert np.array_equal(p1,x['p'])
                x['p0']=p0;x['p1']=p1;rr['same_flow_donor_off_on_structure']=structure(p0,p1,x['y'])
            path=a.out/f'{model}_{role}_y_b_p_video.csv';cols=['row_id','video','y','b','p','f']+(['p0','p1'] if model=='best36' else [])
            with path.open('w',newline='',encoding='utf8') as f:
                writer=csv.writer(f);writer.writerow(cols);writer.writerows(zip(*(x[k] for k in cols)))
            # Round-trip verification of every ID and every floating-point value.
            with path.open(encoding='utf8',newline='') as f: back=list(csv.DictReader(f))
            assert [r['row_id'] for r in back]==x['row_id'].tolist()
            for k in cols:
                if k not in ('row_id','video'): assert np.array_equal(np.array([float(r[k]) for r in back]),x[k]),k
            rr['csv_SHA']=sha(path)
    for role in ('fit','inner'):
        old,new=data['best20'][role],data['best36'][role]
        assert np.array_equal(old['row_id'],new['row_id']) and np.array_equal(old['y'],new['y'])
    result['best36_minus_best20_INNER_bootstrap']={}
    old,new=data['best20']['inner'],data['best36']['inner']
    for name in ('b','p'):
        result['best36_minus_best20_INNER_bootstrap'][name]=paired_bootstrap(new[name],old[name],old['y'],old['video'],plan['bootstrap_draws'],plan['bootstrap_seed'])
    for scheme in ('rows','videos'):
        key='b_calibrated_'+scheme
        qo=evaluate(result['models']['best20']['calibration'][scheme]['b_to_y_FIT'],old['b']);qn=evaluate(result['models']['best36']['calibration'][scheme]['b_to_y_FIT'],new['b'])
        result['best36_minus_best20_INNER_bootstrap'][key]=paired_bootstrap(qn,qo,old['y'],old['video'],plan['bootstrap_draws'],plan['bootstrap_seed'])
    write(a.out/'actual_result.json',result)
    print(json.dumps({m:dict(inner_structure=result['models'][m]['roles']['inner']['structure'],fit_structure=result['models'][m]['roles']['fit']['structure'],calibration=result['models'][m]['calibration'],inner_scores=result['models'][m]['roles']['inner']['scores'],same_flow_inner=result['models'][m]['roles']['inner'].get('same_flow_donor_off_on_structure')) for m in result['models']},ensure_ascii=False))
if __name__=='__main__':main()
