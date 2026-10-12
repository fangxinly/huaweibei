"""Compatible incremental donor branch on the original selected AnchoredFlow."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from anchored_flow import PAIRS,AnchoredFlow
from types import SimpleNamespace

class OriginalTail(nn.Module):
    def __init__(self,core=None):
        super().__init__()
        if core is None:
            core=SimpleNamespace(own_flow=AnchoredFlow(),fusion=nn.Sequential(nn.Linear(300,150),nn.ReLU(),nn.Linear(150,100)),predictor=nn.Sequential(nn.Linear(100,150),nn.ReLU(),nn.Linear(150,1)))
        self.fields=copy.deepcopy(core.own_flow.forward_fields).cpu()
        self.reader=copy.deepcopy(core.own_flow.reader).cpu()
        self.role_head=copy.deepcopy(core.own_flow.role_head).cpu()
        self.fusion=copy.deepcopy(core.fusion).cpu();self.predictor=copy.deepcopy(core.predictor).cpu()
        self.register_buffer('original_gain',core.own_flow.gain.detach().cpu().tanh().clone())
        self.eval().requires_grad_(False)
    def forward(self,first,mask,base,context):
        valid=mask[:,None,:,None].to(first.dtype)
        velocity=torch.stack([self.fields[m](first[:,m],.5,context[:,m]) for m in range(3)],1)
        final=(first+.5*velocity)*valid
        relation=self.reader(final,mask)
        pooled=(final*valid).sum(2)/mask.sum(1)[:,None,None]
        flow=self.predictor(self.fusion(pooled.flatten(1))).view(-1)
        flow=flow+.1*self.role_head(relation['groups'].mean(2).flatten(1)).view(-1)
        return base+self.original_gain*(flow-base)

class IncrementalMessage(nn.Module):
    def __init__(self):
        super().__init__()
        self.heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(400),nn.Linear(400,8),nn.GELU(),nn.Linear(8,100)) for _ in PAIRS])
        for h in self.heads:nn.init.zeros_(h[-1].weight);nn.init.zeros_(h[-1].bias)
        # Last-layer bias cancels identically in the donor-minus-zero contrast.
        for h in self.heads:h[-1].bias.requires_grad_(False)
    def forward(self,slots):
        feedback=[]
        for i,(own,donor) in enumerate(PAIRS):
            feedback.append(self.channel(i,slots[:,own],slots[:,donor]))
        return .125*torch.stack(feedback,1).reshape(len(slots),3,2,100).sum(2)
    def channel(self,i,a,b):
        h=self.heads[i];zero=torch.zeros_like(b)
        def phi(x):return torch.cat([a,x,.5*(a-x),.5*(a+x)],-1)
        return h(phi(b))-h(phi(zero))

def objective(prediction,y,context):
    # Same original sentiment target; bounded Huber derivative reduces outlier
    # dominance. No real-label strong/weak switch at inference, no OOF claim.
    huber=F.huber_loss(prediction,y,delta=1.)
    return huber+.01*context.square().mean(),dict(huber=huber.detach(),context_square=context.square().mean().detach())

def synthetic():
    torch.set_num_threads(2);torch.manual_seed(128);m=IncrementalMessage();s=torch.randn(7,3,100)
    assert torch.equal(m(s),torch.zeros(7,3,100))
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=1e-3)
    for _ in range(3):
        opt.zero_grad();c=m(s);loss=(c-torch.ones_like(c)*.1).square().mean();loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad);opt.step()
    # Every individual direction with zero donor has exact zero feedback,
    # even after training; aggregate all-zero slots also returns exact zero.
    for i,(own,donor) in enumerate(PAIRS):
        a=s[:,own];assert torch.equal(m.channel(i,a,torch.zeros_like(a)),torch.zeros_like(a))
    assert torch.equal(m(torch.zeros_like(s)),torch.zeros(7,3,100))
    before={k:v.clone() for k,v in m.state_dict().items()};rng=torch.get_rng_state().clone();m.eval();p=m(s);perm=torch.arange(6,-1,-1)
    assert torch.allclose(m(s[perm]),p[perm],atol=1e-6);assert torch.equal(rng,torch.get_rng_state());assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    return dict(zero_initial_same_flow_identity=True,zero_donor_contrast_exact=True,finite_trainable_gradients=True,eval_batch_order_state_rng_invariance=True,small_trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad))
