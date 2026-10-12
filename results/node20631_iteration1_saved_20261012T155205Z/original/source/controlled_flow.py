"""Exact directional interventions on the existing two-step AnchoredFlow.

Receiver first-step states and the final decoder are common across interventions.
Eight fixed masks: OFF, six individual directions, ALL. Nonlinear readouts are
computed separately, never formed by summing individual prediction changes.
"""
import math
import torch
from torch import nn
from anchored_flow import PAIRS

class ControlledFlow(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.core = core

    def prefix(self, source, mask, decoder):
        if source.ndim != 4 or source.shape[1] != 3 or source.shape[-1] != 100:
            raise ValueError('Expected original Bx3xTx100 source')
        if mask.shape != (len(source),source.shape[2]) or not mask.any(1).all():
            raise ValueError('Explicit valid original sequence mask required')
        valid = mask[:,None,:,None].to(source.dtype)
        source = source*valid
        base = decoder(((source*valid).sum(2)/mask.sum(1)[:,None,None]).flatten(1)).view(-1)
        zero = source.new_zeros((len(source),3,100))
        velocity = torch.stack([self.core.forward_fields[m](source[:,m],0.,zero[:,m]) for m in range(3)],1)
        first = (source+.5*velocity)*valid
        slots = self.core.reader(first,mask)['slots'].mean(2).detach()
        channels = torch.stack([self.core.message.channel(i,slots[:,own],slots[:,donor])
                                for i,(own,donor) in enumerate(PAIRS)],1)
        return dict(first=first,slots=slots,channels=channels,base=base,mask=mask,valid=valid)

    def finish(self, prefix, decoder, gates):
        first, mask, valid = prefix['first'],prefix['mask'],prefix['valid']
        gates = torch.as_tensor(gates,device=first.device,dtype=first.dtype)
        if gates.shape == (6,): gates = gates[None].expand(len(first),-1)
        if gates.shape != (len(first),6) or not torch.isfinite(gates).all() or (gates<0).any() or (gates>1).any():
            raise ValueError('One finite [0,1] gate per sample and direction required')
        context = .125*(prefix['channels']*gates[:,:,None]).reshape(len(first),3,2,100).sum(2)
        velocity = torch.stack([self.core.forward_fields[m](first[:,m],.5,context[:,m]) for m in range(3)],1)
        final = (first+.5*velocity)*valid
        relation = self.core.reader(final,mask)
        pooled = (final*valid).sum(2)/mask.sum(1)[:,None,None]
        flow = decoder(pooled.flatten(1)).view(-1)+.1*self.core.role_head(relation['groups'].mean(2).flatten(1)).view(-1)
        return prefix['base']+self.core.gain.tanh()*(flow-prefix['base'])

    def proposals(self,source,mask,decoder):
        prefix = self.prefix(source,mask,decoder)
        masks = torch.cat([torch.zeros(1,6,device=source.device),torch.eye(6,device=source.device),torch.ones(1,6,device=source.device)],0)
        values = torch.stack([self.finish(prefix,decoder,g) for g in masks],1)
        # Fixed public random projection; no data, label, normalization or seed selection.
        generator = torch.Generator(device='cpu').manual_seed(128)
        projection = torch.randn(100,4,generator=generator).to(source.device,source.dtype)/math.sqrt(100)
        delta = values-values[:,0,None]
        features = torch.cat([(prefix['slots']@projection).flatten(1),values[:,0,None],delta[:,1:]],1)
        return dict(predictions=values,delta=delta,masks=masks,features=features,
                    p0_is_same_flow_messages_off=True,time_control_claim=False)

    def forward(self,source,mask,decoder,labels=None,gates=None):
        # labels is deliberately ignored; p0 denotes the same flow with messages OFF.
        prefix=self.prefix(source,mask,decoder)
        p0=self.finish(prefix,decoder,torch.zeros(6,device=source.device))
        p=self.finish(prefix,decoder,torch.ones(6,device=source.device) if gates is None else gates)
        return p,p0,{},dict(original_pooled_base=prefix['base'].detach(),same_flow_p0=p0.detach(),euler_steps=2)

    def controlled(self,source,mask,decoder,selector):
        if self.training or self.core.training:
            raise ValueError('Fixed-prediction selection requires evaluation mode')
        proposals = self.proposals(source,mask,decoder)
        choice = selector.choose(proposals['features'].detach().cpu().numpy(),proposals['delta'].detach().cpu().numpy())
        index = torch.as_tensor(choice['index'],device=source.device)
        output = proposals['predictions'][torch.arange(len(source),device=source.device),index]
        return output,choice,proposals

