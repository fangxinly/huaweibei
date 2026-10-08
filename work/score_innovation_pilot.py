"""All frozen pilot arms scored once, no fitting, no selection or TEST score."""
import argparse,csv,datetime,hashlib,json,os,sys,zipfile
from pathlib import Path
import numpy as np
def sha(path):return hashlib.file_digest(Path(path).open('rb'),'sha256').hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def run(a):
    assert sha(a.plan)==a.plan_sha;p=read(a.plan)
    for name,h in p['source_SHA'].items():assert sha(a.plan.parent/name)==h
    sys.path.insert(0,str(a.plan.parent));from sentiment_metrics_careflow_v1 import metrics,SEMANTICS
    D=Path(p['D']);t=D/'A_train/extracted/out';audit=read(D/'B_train_audit/extracted/out/actual_cpu_audit.json')
    assert sha(D/'A_train/actual_D_verification.json')==p['train_D_verification_SHA'] and sha(D/'B_train_audit/actual_D_verification.json')==p['CPU_D_verification_SHA']
    assert audit['labels_read']==False and audit['actual_small_head_CPU_forward'] and max(audit['CPU_prediction_error'].values())<=audit['predeclared_CPU_tolerance']
    assert sha(t/'frozen_INNER_predictions.npz')==p['prediction_SHA'] and sha(t/'frozen_INNER_gate_details.npz')==p['gate_details_SHA']
    assert sha(p['labels_CSV'])==p['labels_CSV_SHA'];assert not a.out.exists();a.out.mkdir()
    token=D/'INNER_labels_scoring_once.token';assert not token.exists()
    with token.open('x',encoding='utf8') as f:f.write(a.plan_sha)
    q=np.load(t/'frozen_INNER_predictions.npz',allow_pickle=False);ids=q['row_ids'].astype(str);pred={k:q[k].astype(float) for k in p['methods']}
    rows=list(csv.DictReader(Path(p['labels_CSV']).open(encoding='utf8')));assert [r['row_id'] for r in rows]==ids.tolist();y=np.array([float(r['y']) for r in rows]);videos=np.array([r['video'] for r in rows]);uv=np.unique(videos)
    groups={'all':np.ones(len(y),bool),'weak':abs(y)<=1,'strong':abs(y)>1};base=pred['baseline_calibrated'];result={}
    for name,x in pred.items():
        result[name]={g:dict(metrics(x[m],y[m]),prediction_mean=float(x[m].mean()),prediction_std=float(x[m].std()),label_std=float(y[m].std()),bias=float((x[m]-y[m]).mean())) for g,m in groups.items()}
        result[name]['video_equal_MSE']=float(np.mean([np.mean((x[videos==v]-y[videos==v])**2) for v in uv]))
    # Paired whole-video resampling; five correlated metrics are not five independent confirmations.
    rng=np.random.default_rng(p['bootstrap_seed']);indices=[np.flatnonzero(videos==v) for v in uv];boots={k:{m:[] for m in ('Acc7','Acc2','F1','MAE','Corr')} for k in pred if k!='baseline_calibrated'}
    for _ in range(p['bootstrap_repeats']):
        idx=np.concatenate([indices[j] for j in rng.integers(len(uv),size=len(uv))]);bm=metrics(base[idx],y[idx])
        for name in boots:
            mm=metrics(pred[name][idx],y[idx])
            for m in boots[name]:
                if mm[m] is not None and bm[m] is not None:boots[name][m].append(mm[m]-bm[m])
    ci={name:{m:np.quantile(v,[.025,.975]).tolist() for m,v in vals.items()} for name,vals in boots.items()}
    gate=np.load(t/'frozen_INNER_gate_details.npz',allow_pickle=False);diagnostic={}
    for name in ('flow','regression'):
        delta=gate[name+'_delta'];rho=y-base;rh=gate[name+'_rhat'];g=gate[name+'_gate'];effect=g*delta;U=2*rho*effect-effect**2
        x=np.stack([base,np.ones(len(base))],1);dres=delta-x@np.linalg.lstsq(x,delta,rcond=None)[0];rres=rho-x@np.linalg.lstsq(x,rho,rcond=None)[0]
        def corr(x,y):return None if np.std(x)<1e-12 or np.std(y)<1e-12 else float(np.corrcoef(x,y)[0,1])
        slopes=read(t/'actual_train_receipt.json')['calibration']['slopes'];estimated=2*slopes[name]*rh*effect-effect**2;order=np.argsort(estimated,kind='stable')
        bins=[dict(n=len(ix),estimated_utility=float(estimated[ix].mean()),actual_utility=float(U[ix].mean())) for ix in np.array_split(order,5)]
        diagnostic[name]=dict(CAL_slope=slopes[name],gate_mean=float(g.mean()),gate_zero_fraction=float((g==0).mean()),gate_one_fraction=float((g==1).mean()),all_on_delta_std=float(delta.std()),delta_explained_by_baseline_linear_R2=None if np.var(delta)<1e-12 else float(1-np.var(dres)/np.var(delta)),partial_corr_after_baseline_linear=corr(dres,rres),actual_MSE_gain=float(U.mean()),positive_utility_fraction=float((U>0).mean()),estimated_utility_fixed_quintiles=bins,weak_actual_MSE_gain=float(U[groups['weak']].mean()),strong_actual_MSE_gain=float(U[groups['strong']].mean()))
    out=dict(status='FIXED_FROZEN_FEATURE_INNOVATION_PILOT_INNER_ALL_FIVE_ONCE_COMPLETE',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,plan_SHA=a.plan_sha,source_SHA=p['source_SHA'],prediction_SHA=p['prediction_SHA'],samples=len(y),videos=len(uv),weak_count=int(groups['weak'].sum()),strong_count=int(groups['strong'].sum()),all_methods=result,paired_video_bootstrap_delta_vs_calibrated_baseline=ci,bootstrap_repeats=p['bootstrap_repeats'],mechanism_diagnostics=diagnostic,metric_semantics=SEMANTICS,INNER_exploratory_previously_checkpoint_selected=True,formal_CaReFlow_comparison=False,original_TEST_not_a_holdout=True,full_fivefold_completed=False,no_fit_or_selection_in_scoring=True)
    write(a.out/'actual_scores.json',out)
    for name,r in result.items():print(name,{k:r['all'][k] for k in ('Acc7','Acc2','F1','MAE','Corr')})
    write(a.out/'natural_exit.json',dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),natural_exit=0,pid=os.getpid(),fullargv=[sys.executable]+sys.argv))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--out',type=Path,required=True);run(p.parse_args())
