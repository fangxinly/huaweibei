"""Analytic single-valid-token geometry; frozen parameters and old source intact.

With one token, every query slot equals that token. Centered slots are zero,
cosines are zero and q=sigmoid(-relation_center/relation_temperature). Avoid
constructing the undefined zero-norm double backward for this exact branch.
Other examples retain the original cosine implementation.
"""
from types import MethodType
import torch
from torch.nn import functional as F

def install_single_token_reader(reader,config,memberships):
 def forward(self,states,mask):
  valid=mask[:,None,:,None].to(states.dtype)
  mean=(states*valid).sum(2,keepdim=True)/valid.sum(2,keepdim=True).clamp(min=1)
  keys=F.normalize(self.key_norm(states-mean),dim=-1);queries=F.normalize(self.queries,dim=-1)
  logits=torch.einsum('bmtd,sd->bmst',keys,queries)/config['query_temperature']
  attention=logits.masked_fill(~mask[:,None,None,:],-1e4).softmax(-1)
  slots=torch.einsum('bmst,bmtd->bmsd',attention,states);centered=slots-slots.mean(2,keepdim=True)
  singleton=mask.sum(1)==1;ids=(~singleton).nonzero().flatten()
  similarities=slots.new_zeros((len(states),3,slots.shape[2]))
  if len(ids):
   selected=centered[ids]
   similarities=similarities.index_copy(0,ids,torch.stack([F.cosine_similarity(selected[:,i],selected[:,j],dim=-1) for i,j in [(0,1),(0,2),(1,2)]],1))
  q=torch.sigmoid((similarities-config['relation_center'])/config['relation_temperature']);mass=memberships(q)
  global_slot=(slots*mass[...,0,None]).sum(1)/3
  pairs=[(slots[:,i]*mass[:,i,:,ri,None]+slots[:,j]*mass[:,j,:,rj,None])/2 for i,j,ri,rj in [(0,1,1,1),(0,2,2,1),(1,2,2,2)]]
  groups=torch.stack([global_slot]+pairs+[slots[:,m]*mass[:,m,:,3,None] for m in range(3)],1);pooled=groups.mean(2)
  feedback=[torch.cat([pooled[:,0],pooled[:,pi[0]],pooled[:,pi[1]],pooled[:,4+m]],-1) for m,pi in [(0,(1,2)),(1,(1,3)),(2,(2,3))]]
  return dict(slots=slots,attention=attention,mass=mass,q=q,groups=groups,feedback=torch.stack(feedback,1))
 reader.forward=MethodType(forward,reader)

def install_for_flow(flow):
 import sys
 module=sys.modules[type(flow.reader).__module__]
 install_single_token_reader(flow.reader,module.CONFIG,module.memberships)
