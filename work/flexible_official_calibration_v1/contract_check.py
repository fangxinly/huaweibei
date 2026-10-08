"""Local synthetic checks; never reads real arrays or checkpoints."""
import datetime,hashlib,json,os,sys
from pathlib import Path
import numpy as np
from flexible_controls import FlexibleControls,official_five,basis,anchors,monotone_fit
from oof_selector import PREDECESSORS,weights

def reject(fn):
    try:fn()
    except ValueError:return
    raise AssertionError('Invalid source or role accepted')

def main():
    ids=[f'train_{v}[{j}]' for v in range(10) for j in range(10)]
    cal=[f'cal_{v}[0]' for v in range(2)]
    allowed=ids+cal
    lineage=[]
    for k in range(5):
        held=[s for s in ids if s.split('[')[0] in {f'train_{2*k}',f'train_{2*k+1}'}]
        train=[s for s in ids if s not in held]
        lineage.append(dict(held_ids=held,predecessors={n:dict(train_ids=train,state_SHA='a'*64) for n in PREDECESSORS}))
    x=np.linspace(-2,2,len(ids))
    w=weights(ids)
    k=anchors(x,w)
    B=basis(x,k)
    yy=B@np.array([-2.,-1.8,-.1,.7,2.])
    # A nonlinear delta within the full basis has no incremental direction.
    d=B@np.array([.8,-.4,.1,-.2,1.1])
    pure=FlexibleControls().fit(yy,x,x+d,ids,allowed,cal,lineage)
    assert pure.gamma==0.
    assert pure.rank_aug==pure.rank
    dense=np.linspace(-3,3,500)
    q=pure.predict(dense,dense)['monotone_calibration_only']
    assert np.all(np.diff(q)>=-1e-12)
    e=float(abs(pure.predict(x,x+d)['monotone_calibration_only']-yy).max())
    assert e<1e-12
    # Residual-only influence: exact projection removes all b-only spline d.
    rng=np.random.default_rng(128)
    noise=rng.normal(size=len(x))
    proj=np.linalg.lstsq(B*np.sqrt(w)[:,None],noise*np.sqrt(w),rcond=None)[0]
    i=noise-B@proj
    target=yy+.25*i
    model=FlexibleControls().fit(target,x,x+d+i,ids,allowed,cal,lineage)
    assert model.orthogonality_maxerror<1e-12
    assert abs(model.gamma-.25)<1e-12
    error=float(abs(model.predict(x,x+d+i)['monotone_calibration_plus_orthogonal_delta']-target).max())
    assert error<1e-12
    # Independently check convex optimum KKT conditions in knot coordinates.
    noisy_y=yy+rng.normal(size=len(x))
    got=monotone_fit(B,noisy_y,w)
    err=B@got-noisy_y
    node_grad=B.T@(w*err)
    intercept_gradient=float(node_grad.sum())
    increment_grad=np.array([node_grad[j:].sum() for j in range(1,len(got))])
    increments=np.diff(got)
    kkt=max(abs(intercept_gradient),max(0.,float(-increment_grad.min())),
            float(abs(increments*increment_grad).max()),max(0.,float(-increments.min())))
    assert kkt<1e-10
    bad=json.loads(json.dumps(lineage))
    bad[0]['predecessors']['encoder']['train_ids'].append(bad[0]['held_ids'][0])
    reject(lambda:FlexibleControls().fit(yy,x,x+d,ids,allowed,cal,bad))
    reject(lambda:FlexibleControls().fit(yy,x,x+d,ids,allowed,ids[:1],lineage))
    reject(lambda:official_five(yy,yy,ids,'INNER'))
    # Metric convention: remove truth=0; prediction=0 is positive.
    m=official_five(np.array([-1.,0.,1.,2.]),np.array([0.,0.,1.,2.]),['a','b','c','d'],'VAL')
    assert m['Acc2']==2/3 and m['Acc7']==.75 and m['MAE']==.25
    assert abs(m['F1']-8/15)<1e-12
    const=official_five(yy,np.zeros_like(yy),ids,'TEST')
    assert const['Corr'] is None
    result=dict(status='LOCAL_SYNTHETIC_FLEXIBLE_OFFICIAL_CALIBRATION_COMPLETE',
                actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),
                fullargv=[sys.executable,*sys.argv],synthetic_only=True,
                source_SHA={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest()
                            for n in ('flexible_controls.py','contract_check.py','oof_selector.py','linear_controls.py')},
                numpy=np.__version__,new_dependencies_installed=False,
                monotone_spline_exact_error=e,remaining_delta_exact_error=error,
                independent_convex_KKT_maxerror=kkt,
                nonlinear_p0_only_delta_exact_zero_gamma=True,video_CAL_leakage_rejected=True,
                official_metric_convention_passed=True,performance_role_INNER_rejected=True,
                true_pair_TRAIN_OOF_complete=False,new_VAL_TEST_scores=False,original_server_validation=False)
    Path(sys.argv[1]).write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
