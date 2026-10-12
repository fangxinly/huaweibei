"""Prospective VAL-only variants: input-adaptive messages and bounded residual readout."""
import torch
from torch import nn
from torch.nn import functional as F
from polarity_intensity_flow import PolarityIntensityFlow
from official_upgrade import OfficialUpgrade
from candidate_adapter import optimizer_groups as original_groups

class AdaptiveFlow(PolarityIntensityFlow):
 def __init__(self,residual):
  super().__init__(OfficialUpgrade(),'regression_aux');self.residual=residual
  with torch.random.fork_rng(devices=[]):
   self.direction_gate=nn.Sequential(nn.Linear(300,64),nn.GELU(),nn.Linear(64,6))
   self.residual_head=nn.Linear(1000,1) if residual else None
  nn.init.zeros_(self.direction_gate[-1].weight);nn.init.zeros_(self.direction_gate[-1].bias)
  if residual:nn.init.zeros_(self.residual_head.weight);nn.init.zeros_(self.residual_head.bias)
 def decode(self,regression,features):
  d=super().decode(regression,features)
  if self.residual:d['prediction']=regression.view(-1)+.5*torch.tanh(self.residual_head(features).view(-1))
  return d
 def forward(self,source,mask,decoder,labels=None,gates=None):
  prefix=self.prefix(source,mask,decoder)
  pooled=(prefix['first']*prefix['valid']).sum(2)/mask.sum(1)[:,None,None]
  learned=torch.sigmoid(self.direction_gate(pooled.flatten(1)))
  if self.training:learned=learned*(torch.rand_like(learned)>=.1).to(learned.dtype)
  if gates is not None:learned=learned*torch.as_tensor(gates,device=source.device,dtype=source.dtype)
  off=self.finish_detail(prefix,decoder,torch.zeros_like(learned));on=self.finish_detail(prefix,decoder,learned)
  self.last=on;self.last_context=on['context'];self.last_off=off;self.last_gates=learned
  return on['prediction'],off['prediction'],{},dict(same_flow_p0=off['prediction'].detach(),original_pooled_base=prefix['base'].detach(),direction_gates=learned.detach(),input_only_gate=True,euler_steps=2)

def make_candidate(method):
 if method not in ['adaptive_gated_aux','adaptive_gated_residual']:raise ValueError('Unfrozen iteration')
 return AdaptiveFlow(method=='adaptive_gated_residual')

def optimizer_groups(model):
 groups,named=original_groups(model);flow=model.dberta.own_flow
 added=list(flow.direction_gate.parameters())+(list(flow.residual_head.parameters()) if flow.residual_head is not None else [])
 ids={id(v) for v in added}
 for g in groups:g['params']=[v for v in g['params'] if id(v) not in ids]
 groups.append(dict(params=added,lr=.001,weight_decay=.01))
 owned=[id(v) for g in groups for v in g['params']]
 assert len(owned)==len(set(owned)) and set(owned)=={id(v) for _,v in named}
 return groups,named

def candidate_objective(model,prediction,labels):
 flow=model.dberta.own_flow;d=flow.last;y=labels.view(-1)
 assert flow.training and isinstance(flow,AdaptiveFlow) and y.shape==prediction.shape and torch.isfinite(y).all()
 weight=y.abs().clamp(max=1.)
 sign=(F.binary_cross_entropy_with_logits(d['polarity_logit'],(y>0).to(y.dtype),reduction='none')*weight).sum()/weight.sum().clamp_min(1e-12)
 return F.mse_loss(prediction,y)+.1*F.huber_loss(d['magnitude'],y.abs(),delta=1.)+.1*sign+.01*d['context'].square().mean()

def synthetic():
 torch.set_num_threads(2);results=[]
 for residual in [False,True]:
  torch.manual_seed(19);m=AdaptiveFlow(residual);decoder=nn.Linear(300,1);source=torch.randn(6,3,5,100);mask=torch.ones(6,5,dtype=torch.bool);mask[0,2:]=False
  class Holder:pass
  holder=Holder();holder.dberta=Holder();holder.dberta.own_flow=m
  opt=torch.optim.AdamW(list(m.parameters())+list(decoder.parameters()),lr=.001)
  y=torch.linspace(-2,2,6)
  for _ in range(3):
   opt.zero_grad();prediction=m(source,mask,decoder)[0];loss=candidate_objective(holder,prediction,y);assert torch.isfinite(loss);loss.backward()
   assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters() if v.requires_grad);opt.step()
  m.eval();state={n:v.clone() for n,v in m.state_dict().items()};rng=torch.get_rng_state().clone()
  with torch.no_grad():
   p=m(source,mask,decoder,torch.zeros(6))[0];assert torch.equal(p,m(source,mask,decoder,torch.ones(6)*7)[0])
   padded=source.clone();padded[~mask[:,None,:,None].expand_as(source)]=999;assert torch.equal(p,m(padded,mask,decoder)[0])
   assert ((m.last_gates>=0)&(m.last_gates<=1)).all()
  assert torch.equal(rng,torch.get_rng_state()) and all(torch.equal(v,m.state_dict()[n]) for n,v in state.items())
  results.append(dict(residual=residual,finite_all_gradients=True,dummy_padding_state_RNG_invariance=True,gate_range_passed=True))
 return dict(scope='NEW_SMALL_SYNTHETIC_CPU_ONLY_NO_TASK_ARRAYS_OR_PRETRAINED_MODEL',variants=results)
if __name__=='__main__':
 import json
 print(json.dumps(synthetic()))
