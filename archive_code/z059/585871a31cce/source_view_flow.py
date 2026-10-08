"""Shared coupled field, seven explicit encoded-source availability paths.

No common-response estimator or compression control is installed. All receivers
may receive available evidence; receiver positions do not identify its source.
"""
import torch
from torch import nn
from native_flow import JointVelocity,pool

class JointControlledFlow(nn.Module):
 def __init__(self,dimension=100,variant='none',rho=0,task_guard=True):
  super().__init__();assert variant=='none' and rho==0
  self.field=JointVelocity(dimension,100)
 def routes(self,source,valid,decoder):
  b,_,length,dim=source.shape
  masks=source.new_tensor([[bool(code&(1<<m)) for m in range(3)] for code in range(1,8)])
  states=(source[:,None]*masks[None,:,:,None,None]).reshape(b*7,3,length,dim)
  repeated=valid[:,None].expand(b,7,length).reshape(b*7,length)
  weight=repeated[:,None,:,None].to(source.dtype);states=states*weight
  context=states.new_zeros(b*7,3,dim);first=None
  for step in range(4):
   states=(states+.25*self.field(states,step/4,context))*weight
   if step==1:first=decoder(pool(states,repeated).flatten(1)).reshape(b,7)
  final=decoder(pool(states,repeated).flatten(1)).reshape(b,7)
  return final,first
 def forward(self,source,valid,decoder,labels=None):
  final,first=self.routes(source,valid,decoder)
  self.last_view_predictions=final.detach()
  loss={} if labels is None else {'source_view_task_loss':((final-labels.reshape(-1,1)).square().mean()+.1*(first-labels.reshape(-1,1)).square().mean())/1.1}
  zero=source.new_zeros(len(source));trace={k:zero for k in ('common_rank','control_norm','context_norm','safe_control_work','measured_rank','unclassified_fraction')}
  trace.update(relation_measured=False,control_installed=False,source_view_count=7)
  return final[:,6],first[:,6],loss,trace

def diagnostic():
 torch.manual_seed(128);f=JointControlledFlow(dimension=6).double();decoder=nn.Linear(18,1).double()
 x=torch.randn(2,3,4,6,dtype=torch.float64,requires_grad=True);valid=torch.tensor([[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
 final,first=f.routes(x,valid,decoder)
 changed=x.detach().clone();changed[:,1:]+=100
 altered=f.routes(changed,valid,decoder)[0];assert torch.equal(final[:,0],altered[:,0])
 padded=x.detach().clone().masked_fill(~valid[:,None,:,None],1000)
 assert torch.equal(final,f.routes(padded,valid,decoder)[0])
 states=x;w=valid[:,None,:,None];context=x.new_zeros(2,3,6)
 for step in range(4):states=(states+.25*f.field(states,step/4,context))*w
 assert torch.allclose(final[:,6],decoder(pool(states,valid).flatten(1)).reshape(-1),atol=1e-12,rtol=1e-12)
 (final.square().mean()+.1*first.square().mean()).backward()
 assert torch.isfinite(x.grad).all() and all(x.grad[:,m].abs().sum()>0 for m in range(3))
 assert f.field.net[-1].weight.grad.abs().sum()>0
 return {'status':'SOURCE_AVAILABILITY_PADDING_FULL_ROUTE_AND_GRADIENT_DIAGNOSTICS_PASSED','no_semantic_or_benefit_claim':True}

if __name__=='__main__':
 import json
 print(json.dumps(diagnostic()))
