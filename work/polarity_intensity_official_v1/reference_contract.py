"""NumPy mathematical reference only; never claims PyTorch flow verification."""
import ast,datetime,hashlib,json,math,os,sys
from pathlib import Path
import numpy as np

def readout(q,ds,da):
    anchor=np.arctanh(np.clip(q/3.,-1+1e-6,1-1e-6))
    s=anchor+ds;a=math.log(math.expm1(3.))+da
    magnitude=np.logaddexp(0.,a)
    return np.tanh(s)*magnitude,s,magnitude

def run():
    folder=Path(__file__).resolve().parent
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.glob('*.py')}
    for p in folder.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    q=np.array([-2.9,-1.,-.01,0.,.01,1.,2.9])
    out,s,m=readout(q,np.zeros_like(q),np.zeros_like(q))
    identity=float(np.max(np.abs(out-q)));assert identity<2e-14
    _,_,small=readout(q,np.zeros_like(q),np.ones_like(q)*-30)
    assert (small>0).all() and (small<1e-10).all()
    sign_fixed=readout(np.array([-.5]),np.zeros(1),np.array([3.]))[0]
    sign_corrected=readout(np.array([-.5]),np.array([1.]),np.zeros(1))[0]
    assert sign_fixed[0]<0 and sign_corrected[0]>0
    product,logit,mag=readout(q,np.ones_like(q)*.2,np.ones_like(q)*-.1)
    eps=1e-6
    ds=(readout(q,np.ones_like(q)*(.2+eps),np.ones_like(q)*-.1)[0]-readout(q,np.ones_like(q)*(.2-eps),np.ones_like(q)*-.1)[0])/(2*eps)
    expected_ds=(1-np.tanh(logit)**2)*mag
    gradient_error=float(np.max(np.abs(ds-expected_ds)));assert gradient_error<1e-8
    y=np.array([1.,-3.]);cov=float(np.mean(y)-np.mean(np.sign(y))*np.mean(np.abs(y)))
    assert cov==-1.
    clipped=readout(np.array([-4.,4.]),np.zeros(2),np.zeros(2))[0]
    assert np.allclose(clipped,[-3*(1-1e-6),3*(1-1e-6)])
    weights=np.minimum(abs(np.array([-1.,0.,.1])),1.);assert weights[1]==0
    return dict(status='NUMPY_MATH_AND_SOURCE_SYNTAX_ONLY_COMPLETE',
        actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,
        source_SHA=sources,anchor_identity_error=identity,polarity_gradient_error=gradient_error,
        positive_magnitude_can_approach_zero=True,all_modalities_may_correct_sign=True,
        conditional_mean_product_covariance_counterexample=True,out_of_range_anchor_clipping_explicit=True,
        local_PyTorch_available=False,native_contract_executed=False,original_server_verified=False,
        actual_new_training=False,real_dataset_or_weights_used=False,new_VAL_TEST_scores=False)

if __name__=='__main__':
    if len(sys.argv)!=2:raise ValueError('Explicit result JSON path required')
    result=run();Path(sys.argv[1]).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)
