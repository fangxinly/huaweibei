"""One new source-supervised residual flow; coefficients fixed before CV execution.

No self-derived consensus/cycle targets. A zero-initial scalar starts the final
prediction exactly at its source prediction. This is not a risk bound.
"""
import torch
from torch import nn
from torch.nn import functional as F
from legacy_flow_model import Velocity,RelationReader
from finite_single_token_reader_v1 import install_for_flow

PAIRS=((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))


class AnchoredFlow(nn.Module):
    def __init__(self):
        super().__init__();d=100
        self.forward_fields=nn.ModuleList([Velocity(d) for _ in range(3)])
        self.reader=RelationReader(d,8);install_for_flow(self)
        self.role_head=nn.Sequential(nn.Linear(7*d,150),nn.GELU(),nn.Linear(150,1))
        self.donor_feedback=nn.ModuleList([
            nn.Sequential(nn.LayerNorm(4*d),nn.Linear(4*d,d),nn.GELU(),nn.Linear(d,d)) for _ in PAIRS])
        for module in self.donor_feedback:
            nn.init.zeros_(module[-1].weight);nn.init.zeros_(module[-1].bias)
        nn.init.zeros_(self.role_head[-1].weight);nn.init.zeros_(self.role_head[-1].bias)
        self.gain=nn.Parameter(torch.zeros(()))

    def forward(self, source, mask, decoder, labels=None):
        if source.ndim!=4 or source.shape[1]!=3 or source.shape[-1]!=100 or not mask.any(1).all():
            raise ValueError('Explicit valid multimodal source required')
        valid=mask[:,None,:,None].to(source.dtype)
        source=source*valid
        def pool(x):return (x*valid).sum(2)/mask.sum(1)[:,None,None]
        base=decoder(pool(source).flatten(1)).view(-1)
        states=source;context=source.new_zeros((len(source),3,100))
        for step in range(2):
            velocity=torch.stack([self.forward_fields[m](states[:,m],step*.5,context[:,m]) for m in range(3)],1)
            states=(states+.5*velocity)*valid
            relation=self.reader(states,mask)
            if step==0:
                slots=relation['slots'].mean(2)
                feedback=[]
                for i,(own,donor) in enumerate(PAIRS):
                    a,b=slots[:,own],slots[:,donor]
                    features=torch.cat([a,b,.5*(a-b),.5*(a+b)],-1).detach()
                    feedback.append(self.donor_feedback[i](features))
                # Same fixed donor scale as the old flow; no new weight sweep.
                context=.125*torch.stack(feedback,1).reshape(len(source),3,2,100).sum(2)
        flow=decoder(pool(states).flatten(1)).view(-1)
        flow=flow+.1*self.role_head(relation['groups'].mean(2).flatten(1)).view(-1)
        gain=self.gain.tanh()
        prediction=base+gain*(flow-base)
        self.last_base=base
        self.last_flow=flow
        self.last_gain=gain
        # First argument remains compatible with the copied feature adapter.
        trace={'source_prediction':base.detach(),'flow_prediction':flow.detach(),'gain':gain.detach(),
               'euler_steps':2,'forward_passes':1}
        return prediction,base,{},trace


def objective(prediction, labels, base):
    p=prediction.view(-1);y=labels.view(-1);b=base.view(-1)
    nz=y!=0
    sign=F.softplus(-y[nz].sign()*p[nz]).mean() if nz.any() else p.sum()*0
    parts={'final_mse':(p-y).square().mean(),'final_mae':(p-y).abs().mean(),
           'source_mse':(b-y).square().mean(),'nonzero_sign':sign}
    weak=y.abs()<=1
    # Only FIT training labels identify weak samples; inference is unchanged.
    # Detach the comparator so this term cannot reduce itself by a direct
    # gradient that makes the base prediction worse.
    excess=(p-y).square()-(b.detach()-y).square()
    retention=F.relu(excess[weak]).mean() if weak.any() else p.sum()*0
    parts['weak_harm_excess']=retention
    total=parts['final_mse']+.25*parts['final_mae']+.5*parts['source_mse']+.1*parts['nonzero_sign']+retention
    return total,parts


def synthetic_check():
    torch.set_num_threads(2);torch.manual_seed(128)
    x=torch.randn(6,3,8,100);mask=torch.ones(6,8,dtype=torch.bool)
    mask[0,1:]=False;mask[1,5:]=False
    y=torch.tensor([-2.,-1.,0.,0.,1.,2.]);decoder=nn.Linear(300,1)
    model=AnchoredFlow();opt=torch.optim.AdamW(list(model.parameters())+list(decoder.parameters()),lr=1e-3)
    p,base,_,_=model(x,mask,decoder,y)
    assert torch.equal(p,base) and model.gain.item()==0
    records=[]
    for step in range(3):
        opt.zero_grad();p,base,_,_=model(x,mask,decoder,y)
        loss,parts=objective(p,y,base);loss.backward()
        assert torch.isfinite(loss)
        assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in model.parameters())
        records.append({'step':step+1,'loss':float(loss.detach()),'gain_gradient':float(model.gain.grad),
                        'finite_gradient_tensors':len(list(model.parameters()))})
        opt.step()
    model.eval()
    with torch.no_grad():
        p=model(x,mask,decoder,y)[0]
        p0=model(x,mask,decoder,torch.zeros_like(y))[0]
        p7=model(x,mask,decoder,torch.full_like(y,7))[0]
        assert torch.equal(p0,p7)
        padded=x.clone();padded[~mask[:,None,:,None].expand_as(x)]=999
        assert torch.equal(model(padded,mask,decoder)[0],p0)
        perm=torch.tensor([5,3,1,0,4,2]);pp=model(x[perm],mask[perm],decoder)[0]
        assert torch.allclose(pp,p0[perm],atol=2e-6,rtol=2e-6)
        role=model.reader(x*mask[:,None,:,None],mask)
        assert role['attention'][~mask[:,None,None,:].expand_as(role['attention'])].abs().max()==0
    return {'status':'SYNTHETIC_TORCH_FLOW_CONTRACT_PASSED','records':records,
            'initial_source_identity':True,'dummy_label_invariance':True,'masked_padding_invariance':True,
            'singleton_gradients_finite':True,'real_data_or_pretrained_model_forward':False}


if __name__=='__main__':
    import json
    print(json.dumps(synthetic_check()))
