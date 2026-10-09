"""Synthetic both-arm saved-tail and independent five-metric qualification."""
import argparse, os, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score,f1_score
from scipy.stats import pearsonr
from common import sha,write,utc
from candidate_audit_utils import make_tail,tail_model_key,load_verified_tail
from sentiment_metrics_careflow_v1 import metrics
from fixed_flow_components_candidate import tensor_sha

def run(out):
    torch.set_num_threads(2); torch.manual_seed(128); modes={}
    for mode in ('factorized_aux','regression_aux'):
        tail=make_tail(mode)
        with torch.no_grad():
            tail.flow.core.gain.fill_(.6)
            for head in tail.flow.core.message.heads: head[-1].weight.normal_(std=.08)
            for head in (tail.flow.polarity_delta,tail.flow.magnitude_delta): head.weight.normal_(std=.008)
        x=torch.randn(7,3,9,100); mask=torch.ones(7,9,dtype=torch.bool); mask[0,1:]=False
        x=x*mask[:,None,:,None]; selected={tail_model_key(k):v.clone() for k,v in tail.state_dict().items()}
        restored=load_verified_tail(dict(mode=mode,state=tail.state_dict()),mode,selected)
        state=tensor_sha(restored.state_dict()); rng=torch.get_rng_state().clone()
        with torch.no_grad():
            on,off=tail(x,mask); on2,off2=restored(x,mask)
            assert torch.equal(on,on2) and torch.equal(off,off2)
            whole=restored.flow(x,mask,lambda s:restored.predictor(restored.fusion(s)),torch.zeros(7))[0]
            other=restored.flow(x,mask,lambda s:restored.predictor(restored.fusion(s)),torch.full((7,),7.))[0]
            assert torch.equal(whole,on2) and torch.equal(whole,other)
            assert float((on-off).abs().max())>1e-7
        assert state==tensor_sha(restored.state_dict()) and torch.equal(rng,torch.get_rng_state())
        modes[mode]=dict(saved_tail_exact=True,same_flow_ON_OFF_nonzero=True,dummy_labels0vs7_exact=True,parameters_buffers_RNG_unchanged=True)
    y=np.array([-3.,-1.5,-.5,0.,.5,1.5,3.]); pred=np.array([-4.,-.5,0.,-1.,0.,2.5,4.]); nz=y!=0
    independent=dict(Acc7=float(accuracy_score(np.round(np.clip(y,-3,3)),np.round(np.clip(pred,-3,3)))),Acc2=float(accuracy_score(y[nz]>=0,pred[nz]>=0)),F1=float(f1_score(y[nz]>=0,pred[nz]>=0,average='weighted')),MAE=float(np.abs(pred-y).mean()),Corr=float(pearsonr(pred,y).statistic))
    result=metrics(pred,y); error=max(abs(result[k]-v) for k,v in independent.items()); assert error<1e-14
    write(out,dict(status='CANDIDATE_BOTH_MODES_SAVED_TAIL_AND_INDEPENDENT_FIVE_CPU_PASSED',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,source_SHA=sha(__file__),modes=modes,independent_five_maxerror=error,real_data_or_labels_indexed=False,CPU_encoder_forward=False,CPU_only=True))
    print(out.read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',type=Path,required=True); run(p.parse_args().out)
