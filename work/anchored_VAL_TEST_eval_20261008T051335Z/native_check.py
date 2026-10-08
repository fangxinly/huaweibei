import argparse,os,sys
from pathlib import Path
import torch
from torch import nn
from types import SimpleNamespace
from incremental_message import OriginalTail,IncrementalMessage,synthetic,objective
from anchored_flow import AnchoredFlow
from fixed_flow_components_candidate import tensor_sha
from common import write,utc
def check(device):
    torch.manual_seed(128)
    core=SimpleNamespace(own_flow=AnchoredFlow().to(device),fusion=nn.Sequential(nn.Linear(300,150),nn.ReLU(),nn.Linear(150,100)).to(device),predictor=nn.Sequential(nn.Linear(100,150),nn.ReLU(),nn.Linear(150,1)).to(device))
    with torch.no_grad():core.own_flow.gain.fill_(-1.1);core.own_flow.role_head[-1].weight.normal_(0,.01)
    core.own_flow.eval();tail=OriginalTail(core).to(device);msg=IncrementalMessage().to(device)
    x=torch.randn(6,3,9,100,device=device);mask=torch.ones(6,9,dtype=torch.bool,device=device);mask[0,1:]=False;mask[1,5:]=False
    x=x*mask[:,None,:,None];zero=torch.zeros(6,3,100,device=device)
    with torch.no_grad():
        base=core.predictor(core.fusion((x.sum(2)/mask.sum(1)[:,None,None]).flatten(1))).view(-1)
        first=(x+.5*torch.stack([core.own_flow.forward_fields[j](x[:,j],0.,zero[:,j]) for j in range(3)],1))*mask[:,None,:,None]
        slots=core.own_flow.reader(first,mask)['slots'].mean(2)
        # Synthetic donor modules are exactly zero; direct original model is true off.
        original=core.own_flow(x,mask,lambda z:core.predictor(core.fusion(z)))[0]
        off=tail(first,mask,base,zero);identity_error=float((original-off).abs().max());assert identity_error<=2e-6
        assert torch.equal(off,tail(first,mask,base,msg(slots)))
        changed=first.clone();changed[~mask[:,None,:,None].expand_as(first)]=777.
        assert torch.equal(off,tail(changed,mask,base,zero))
    before=tensor_sha(tail.state_dict());opt=torch.optim.AdamW([p for p in msg.parameters() if p.requires_grad],lr=.001)
    y=torch.tensor([-2.,-1.,0.,.5,1.,2.],device=device)
    for _ in range(3):
        opt.zero_grad();c=msg(slots);pred=tail(first,mask,base,c);loss,_=objective(pred,y,c);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in msg.parameters() if p.requires_grad)
        assert all(p.grad is None for p in tail.parameters());opt.step()
    assert before==tensor_sha(tail.state_dict());assert not torch.equal(msg(slots),zero)
    return dict(device=device,direct_original_off_equivalence_maxerror=identity_error,zero_initial_exact_off=True,masked_padding_exact=True,singleton_and_adapter_gradients_finite=True,frozen_tail_unchanged=True,adapter_parameters=sum(p.numel() for p in msg.parameters() if p.requires_grad))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--gpu',action='store_true');a=p.parse_args();a.out.mkdir()
    torch.set_num_threads(2);r=dict(actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,synthetic=synthetic(),checks=[check('cpu')])
    if a.gpu:r['checks'].append(check('cuda'))
    write(a.out/'native_result.json',r);print(r)
