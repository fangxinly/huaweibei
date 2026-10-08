"""CPU synthetic qualification of selected flow ON/OFF and author five metrics.

No real TRAIN/VAL/TEST data, full encoder or performance evidence.
"""
import argparse, os, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr
from common import sha, write, utc
from official_upgrade import OfficialUpgrade
from incremental_message import OriginalTail
from sentiment_metrics_careflow_v1 import metrics
from fixed_flow_components_candidate import tensor_sha

def run(a):
    torch.set_num_threads(2)
    torch.manual_seed(128)
    flow = OfficialUpgrade()
    fusion = nn.Sequential(nn.Linear(300,150), nn.ReLU(), nn.Linear(150,100))
    predictor = nn.Sequential(nn.Linear(100,150), nn.ReLU(), nn.Linear(150,1))
    # Nonzero trained-like weights avoid merely checking zero-init identity.
    with torch.no_grad():
        flow.gain.fill_(.6)
        for head in flow.message.heads:
            head[-1].weight.normal_(std=.08)
        for field in flow.forward_fields:
            for param in field.parameters():
                param.add_(torch.randn_like(param)*.03)
    flow.eval(); fusion.eval(); predictor.eval()
    tail = OriginalTail(SimpleNamespace(own_flow=flow, fusion=fusion, predictor=predictor))
    x = torch.randn(5,3,7,100)
    mask = torch.ones(5,7,dtype=torch.bool); mask[0,1:]=False
    x = x*mask[:,None,:,None]
    before=tensor_sha(flow.state_dict()); rng=torch.get_rng_state().clone()
    with torch.no_grad():
        whole=flow(x,mask,lambda s:predictor(fusion(s)),torch.zeros(5))[0]
        base=tail.predictor(tail.fusion((x.sum(2)/mask.sum(1)[:,None,None]).flatten(1))).view(-1)
        zero=x.new_zeros((len(x),3,100))
        first=(x+.5*torch.stack([tail.fields[j](x[:,j],0.,zero[:,j]) for j in range(3)],1))*mask[:,None,:,None]
        slots=tail.reader(first,mask)['slots'].mean(2)
        explicit=tail(first,mask,base,flow.message(slots))
        off=tail(first,mask,base,zero)
        error=float((whole-explicit).abs().max()); assert error<1e-6
        assert float((whole-off).abs().max())>1e-7
        assert torch.equal(whole,flow(x,mask,lambda s:predictor(fusion(s)),torch.ones(5)*7)[0])
    assert before==tensor_sha(flow.state_dict()) and torch.equal(rng,torch.get_rng_state())
    y=np.array([-3.,-1.5,-.5,0.,.5,1.5,3.],dtype=float)
    p=np.array([-4.,-.5,0.,-1.,0.,2.5,4.],dtype=float)
    r=metrics(p,y); nz=y!=0
    independent=dict(Acc7=float(accuracy_score(np.round(np.clip(y,-3,3)),np.round(np.clip(p,-3,3)))),Acc2=float(accuracy_score(y[nz]>=0,p[nz]>=0)),F1=float(f1_score(y[nz]>=0,p[nz]>=0,average='weighted')),MAE=float(np.abs(p-y).mean()),Corr=float(pearsonr(p,y).statistic))
    assert max(abs(r[k]-independent[k]) for k in independent)<1e-14
    write(a.out,dict(status='CPU_SYNTHETIC_POSTTRAIN_FLOW_ON_OFF_AND_INDEPENDENT_FIVE_PASSED',actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,source_SHA=sha(__file__),CPU_only=True,real_data_or_labels_indexed=False,CPU_encoder_forward=False,nonzero_message_ON_equals_full_flow_maxerror=error,nonzero_functional_message_difference=True,dummy_label_replay_state_RNG_passed=True,author_semantics_sklearn_scipy_maxerror=max(abs(r[k]-independent[k]) for k in independent)))
    print(a.out.read_text(),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);run(p.parse_args())
