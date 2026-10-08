"""Conditional flow decomposition and small task heads; no encoder ownership."""
import torch
from torch import nn
from contract import PAIRS

def mlp(din,dout,hidden=64):return nn.Sequential(nn.Linear(din,hidden),nn.SiLU(),nn.Linear(hidden,dout))
class Baseline(nn.Module):
    def __init__(self,d=32,input_dim=None):super().__init__();self.head=mlp(3*d if input_dim is None else input_dim,1)
    def forward(self,z):return self.head(z.flatten(1)).view(-1)
class ConditionalFlow(nn.Module):
    def __init__(self,d=32,draws=8,steps=8):
        super().__init__();assert draws%2==0;self.fields=nn.ModuleList([mlp(2*d+1,d) for _ in PAIRS]);self.steps=steps
        gen=torch.Generator().manual_seed(128);e=torch.randn(draws//2,d,generator=gen)
        self.register_buffer('common_noise',torch.cat([e,-e],0))
    def objective(self,z,generator):
        losses=[]
        for field,(m,n) in zip(self.fields,PAIRS):
            eps=torch.randn(z[:,n].shape,device=z.device,generator=generator);t=torch.rand(len(z),1,device=z.device,generator=generator)
            donor=z[:,n];zt=(1-t)*eps+t*donor;velocity=field(torch.cat([zt,z[:,m],t],-1))
            losses.append((velocity-(donor-eps)).square().mean())
        return torch.stack(losses).mean()
    def means(self,z):
        out=[];K=len(self.common_noise)
        for field,(m,n) in zip(self.fields,PAIRS):
            state=self.common_noise[None].expand(len(z),-1,-1).reshape(-1,z.shape[-1]).clone()
            receiver=z[:,m,None].expand(-1,K,-1).reshape_as(state)
            for step in range(self.steps):
                t=state.new_full((len(state),1),step/self.steps)
                state=state+field(torch.cat([state,receiver,t],-1))/self.steps
            out.append(state.reshape(len(z),K,-1).mean(1))
        return torch.stack(out,1)
class ConditionalRegression(nn.Module):
    def __init__(self,d=32):
        super().__init__();self.fields=nn.ModuleList([mlp(d,d,96) for _ in PAIRS])
    def means(self,z):return torch.stack([f(z[:,m]) for f,(m,n) in zip(self.fields,PAIRS)],1)
    def objective(self,z,generator=None):
        target=torch.stack([z[:,n] for m,n in PAIRS],1);return (self.means(z)-target).square().mean()
class Proposal(nn.Module):
    def __init__(self,d=32):super().__init__();self.heads=nn.ModuleList([mlp(2*d,1,32) for _ in PAIRS])
    def forward(self,z,messages):
        delta=torch.stack([(h(torch.cat([z[:,m],messages[:,c]],-1))-h(torch.cat([z[:,m],torch.zeros_like(messages[:,c])],-1))).view(-1) for c,(h,(m,n)) in enumerate(zip(self.heads,PAIRS))],1)
        return delta.mean(1),delta
class GateHead(nn.Module):
    def __init__(self,d=32):super().__init__();self.head=mlp(3*d+14,1)
    def forward(self,x):return self.head(x).view(-1)
def gate_features(z,p0,delta,channels,messages):
    energy=messages.square().mean(-1)
    return torch.cat([z.flatten(1),p0[:,None],delta[:,None],channels,energy],1)
