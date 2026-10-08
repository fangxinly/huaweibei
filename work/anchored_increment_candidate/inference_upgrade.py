"""Drop-in original flow replacement for the preserved full encoder/decoder model.

Reconstruction: load exact original selected full state, replace own_flow with
this wrapper, then load the small saved adapter. No new pretrained weights.
"""
import copy
import torch
from torch import nn
from incremental_message import IncrementalMessage

class AnchoredMessageUpgrade(nn.Module):
    def __init__(self,original_flow,adapter_state):
        super().__init__()
        self.original=copy.deepcopy(original_flow).eval().requires_grad_(False)
        self.adapter=IncrementalMessage()
        self.adapter.load_state_dict(adapter_state,strict=True)
        self.eval()
    def forward(self,source,mask,decoder,labels=None):
        if source.ndim!=4 or source.shape[1]!=3 or source.shape[-1]!=100 or not mask.any(1).all():raise ValueError('Original masked100-dimensional source required')
        valid=mask[:,None,:,None].to(source.dtype);source=source*valid
        def pool(x):return (x*valid).sum(2)/mask.sum(1)[:,None,None]
        base=decoder(pool(source).flatten(1)).view(-1);zero=source.new_zeros((len(source),3,100))
        first=(source+.5*torch.stack([self.original.forward_fields[j](source[:,j],0.,zero[:,j]) for j in range(3)],1))*valid
        slots=self.original.reader(first,mask)['slots'].mean(2)
        context=self.adapter(slots.detach())
        final=(first+.5*torch.stack([self.original.forward_fields[j](first[:,j],.5,context[:,j]) for j in range(3)],1))*valid
        relation=self.original.reader(final,mask)
        flow=decoder(pool(final).flatten(1)).view(-1)+.1*self.original.role_head(relation['groups'].mean(2).flatten(1)).view(-1)
        gain=self.original.gain.tanh();prediction=base+gain*(flow-base)
        self.last_base=base;self.last_flow=flow;self.last_gain=gain
        return prediction,base,{},dict(source_prediction=base.detach(),flow_prediction=flow.detach(),gain=gain.detach(),euler_steps=2,forward_passes=1,frozen_original_bone=True,only_donor_branch_changed=True)
