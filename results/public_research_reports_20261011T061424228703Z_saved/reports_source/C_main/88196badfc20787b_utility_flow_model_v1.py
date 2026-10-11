"""Directed task-head utility is a supervised proxy, not a semantic truth."""
import torch
from torch import nn
import torch.nn.functional as F
from legacy_flow_model import CONFIG as LEGACY_CONFIG, WholeStateFlow as LegacyWholeStateFlow
CONFIG=dict(LEGACY_CONFIG,pair_weight=.025,utility_weight=.01,utility_target_scale=.5,utility_gate_sharpness=4.)
PAIRS=((0,1),(0,2),(1,0),(1,2),(2,0),(2,1))

class WholeStateFlow(LegacyWholeStateFlow):
    def __init__(self,mode):
        assert mode in ('none','fixed','predicted')
        super().__init__('mid')
        self.mode=mode;self.context_override=None
        # Legacy geometric feedback is replaced; reader and role hypotheses remain.
        del self.feedback
        d=CONFIG['dimension']
        self.pair_heads=nn.ModuleList([nn.Sequential(nn.Linear(2*d,d),nn.GELU(),nn.Linear(d,1)) for _ in PAIRS])
        self.utility_heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(2*d+3),nn.Linear(2*d+3,d),nn.GELU(),nn.Linear(d,1)) for _ in PAIRS])
        self.donor_feedback=nn.ModuleList([nn.Sequential(nn.LayerNorm(4*d),nn.Linear(4*d,d),nn.GELU(),nn.Linear(d,d)) for _ in PAIRS])
        for net in list(self.utility_heads)+list(self.donor_feedback):
            nn.init.zeros_(net[-1].weight);nn.init.zeros_(net[-1].bias)
        self._stage1=None;self._calls=0

    def update_context(self,old,relation):
        self._calls+=1
        if self._calls>1:return old
        pooled=relation['slots'].mean(2)
        own=torch.stack([self.unimodal_heads[m](pooled[:,m]).view(-1) for m in range(3)],1)
        pairs=[];utilities=[];feedback=[]
        for i,(m,j) in enumerate(PAIRS):
            pair=self.pair_heads[i](torch.cat([pooled[:,m],pooled[:,j]],-1)).view(-1)
            ui=torch.cat([pooled[:,m],pooled[:,j],torch.tanh(torch.stack([own[:,m],own[:,j],pair],-1)/3.)],-1).detach()
            utilities.append(torch.tanh(self.utility_heads[i](ui).view(-1)))
            fi=torch.cat([pooled[:,m],pooled[:,j],.5*(pooled[:,m]-pooled[:,j]),.5*(pooled[:,m]+pooled[:,j])],-1).detach()
            feedback.append(self.donor_feedback[i](fi));pairs.append(pair)
        pairs=torch.stack(pairs,1);utilities=torch.stack(utilities,1);feedback=torch.stack(feedback,1)
        predicted_weights=torch.sigmoid(CONFIG['utility_gate_sharpness']*utilities)
        mode=self.mode if self.context_override is None else self.context_override
        assert mode in ('none','fixed','predicted')
        weights={'none':torch.zeros_like(predicted_weights),'fixed':torch.full_like(predicted_weights,.5),'predicted':predicted_weights}[mode]
        # Exactly two directed donors per receiver; fixed averaging bounds scale.
        context=.5*(weights[:,:,None]*feedback).reshape(len(old),3,2,-1).sum(2)
        self._stage1={'own':own,'pair':pairs,'utility':utilities,'weights':weights,'predicted_weights':predicted_weights}
        return .5*old+.5*context

    def forward(self,source,mask,decoder,labels=None):
        self._calls=0;self._stage1=None
        prediction,first,losses,trace=super().forward(source,mask,decoder,labels)
        assert self._calls==2 and self._stage1 is not None
        s=self._stage1
        if self.training:
            y=labels.view(-1,1)
            own_error=(s['own']-y).square()
            pair_error=(s['pair']-y).square()
            receiver_error=torch.stack([own_error[:,m] for m,j in PAIRS],1)
            q=(receiver_error-pair_error).detach()
            target=torch.tanh(q/CONFIG['utility_target_scale'])
            assert not target.requires_grad
            losses['source_unimodal_sentiment']=losses['unimodal_sentiment']
            losses['stage1_observer_sentiment']=own_error.mean()
            losses['unimodal_sentiment']=.5*(losses['source_unimodal_sentiment']+losses['stage1_observer_sentiment'])
            losses['pair_sentiment']=pair_error.mean()
            losses['utility_calibration']=F.smooth_l1_loss(s['utility'],target)
            s['target']=target;s['raw_utility']=q
        trace['utility_weights']=s['weights'].detach();trace['utility_prediction']=s['utility'].detach()
        trace['condition_mode']=self.mode
        return prediction,first,losses,trace
