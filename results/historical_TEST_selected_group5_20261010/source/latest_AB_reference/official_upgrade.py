"""Clean official-TRAIN version of the original AnchoredFlow donor contrast."""
import torch
from torch.nn import functional as F
from anchored_flow import AnchoredFlow
from incremental_message import IncrementalMessage

class OfficialUpgrade(AnchoredFlow):
    def __init__(self):
        super().__init__()
        del self.donor_feedback
        self.message=IncrementalMessage()

    def forward(self,source,mask,decoder,labels=None):
        if source.ndim!=4 or source.shape[1:]!=(3,source.shape[2],100) or not mask.any(1).all():
            raise ValueError('Explicit original 3x100 source and valid mask required')
        valid=mask[:,None,:,None].to(source.dtype);states=source*valid
        def pool(z):return (z*valid).sum(2)/mask.sum(1)[:,None,None]
        base=decoder(pool(states).flatten(1)).view(-1);context=source.new_zeros((len(source),3,100))
        for step in range(2):
            velocity=torch.stack([self.forward_fields[m](states[:,m],step*.5,context[:,m]) for m in range(3)],1)
            states=(states+.5*velocity)*valid;relation=self.reader(states,mask)
            if step==0:context=self.message(relation['slots'].mean(2).detach())
        flow=decoder(pool(states).flatten(1)).view(-1)+.1*self.role_head(relation['groups'].mean(2).flatten(1)).view(-1)
        prediction=base+self.gain.tanh()*(flow-base)
        self.last_base=base;self.last_flow=flow;self.last_context=context
        return prediction,base,{},dict(euler_steps=2,forward_passes=1,source_prediction=base.detach())

def objective(model,p,y):
    # Exact current adapter objective, now jointly trained from a clean public
    # initialization under the shared 4000-update official budget.
    return F.huber_loss(p.view(-1),y.view(-1),delta=1.)+.01*model.dberta.own_flow.last_context.square().mean()

def synthetic():
    from torch import nn
    torch.set_num_threads(2);torch.manual_seed(128)
    m=OfficialUpgrade();d=nn.Linear(300,1);x=torch.randn(6,3,8,100);mask=torch.ones(6,8,dtype=torch.bool);mask[0,1:]=False
    p,b,_,_=m(x,mask,d);assert torch.equal(p,b);assert torch.equal(m.last_context,torch.zeros_like(m.last_context))
    m.gain.data.fill_(.1);opt=torch.optim.AdamW([p for p in list(m.parameters())+list(d.parameters()) if p.requires_grad],lr=1e-3)
    for _ in range(3):
        opt.zero_grad();p=m(x,mask,d)[0];loss=F.huber_loss(p,torch.linspace(-2,2,6))+.01*m.last_context.square().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
        opt.step()
    m.eval();before={n:v.clone() for n,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
    with torch.no_grad():
        p=m(x,mask,d,torch.zeros(6))[0];assert torch.equal(p,m(x,mask,d,torch.ones(6)*7)[0]);padded=x.clone();padded[~mask[:,None,:,None].expand_as(x)]=999
        assert torch.equal(p,m(padded,mask,d)[0]);perm=torch.arange(5,-1,-1);assert torch.allclose(p[perm],m(x[perm],mask[perm],d)[0],atol=2e-6,rtol=2e-6)
    assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,m.state_dict()[n]) for n,v in before.items())
    from incremental_message import synthetic as message_contract
    contrast=message_contract()
    return dict(zero_initial_source_identity=True,actual_native_CPU_synthetic_only=True,finite_gradients=True,message_contrast=contrast,dummy_padding_permutation_state_RNG=True,trainable_flow_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad))

if __name__=='__main__':
    import json
    print(json.dumps(synthetic()))
