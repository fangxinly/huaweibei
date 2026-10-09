"""Small polarity/intensity readout on the existing directional two-step flow.

Both branches see all modalities. Ordinary supervised candidate, not an OOF
utility controller or a claim that audio/vision uniquely encode intensity.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from controlled_flow import ControlledFlow


class PolarityIntensityFlow(ControlledFlow):
    MODES = ('regression_aux', 'factorized_aux')
    FEATURES = 1000
    SCALE = 3.0
    EPS = 1e-6

    def __init__(self, core, mode):
        if mode not in self.MODES:
            raise ValueError('Fixed regression_aux or factorized_aux arm required')
        super().__init__(core)
        self.mode = mode
        # Extra heads cannot change RNG used for common encoder/decoder init.
        with torch.random.fork_rng(devices=[]):
            self.polarity_delta = nn.Linear(self.FEATURES, 1)
            self.magnitude_delta = nn.Linear(self.FEATURES, 1)
        for head in (self.polarity_delta, self.magnitude_delta):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def decode(self, regression, features):
        if features.shape != (len(regression), self.FEATURES):
            raise ValueError('Original 300 pooled + 700 relation features required')
        q = regression.view(-1)
        anchor = torch.atanh((q / self.SCALE).clamp(-1+self.EPS, 1-self.EPS))
        s = anchor + self.polarity_delta(features).view(-1)
        a = math.log(math.expm1(self.SCALE)) + self.magnitude_delta(features).view(-1)
        magnitude = F.softplus(a)
        product = torch.tanh(s) * magnitude
        return dict(prediction=q if self.mode == 'regression_aux' else product,
                    regression=q, product=product, polarity_logit=2*s,
                    soft_polarity=torch.tanh(s), magnitude=magnitude,
                    anchor_clipped=(q.abs() > self.SCALE*(1-self.EPS)))

    def finish_detail(self, prefix, decoder, gates):
        first, mask, valid = prefix['first'], prefix['mask'], prefix['valid']
        gates = torch.as_tensor(gates, device=first.device, dtype=first.dtype)
        if gates.shape == (6,): gates = gates[None].expand(len(first), -1)
        if (gates.shape != (len(first), 6) or not torch.isfinite(gates).all()
                or (gates < 0).any() or (gates > 1).any()):
            raise ValueError('Finite [0,1] sample/direction gates required')
        context = .125*(prefix['channels']*gates[:,:,None]).reshape(len(first),3,2,100).sum(2)
        velocity = torch.stack([self.core.forward_fields[m](first[:,m],.5,context[:,m]) for m in range(3)],1)
        final = (first+.5*velocity)*valid
        relation = self.core.reader(final,mask)
        pooled = (final*valid).sum(2)/mask.sum(1)[:,None,None]
        groups = relation['groups'].mean(2).flatten(1)
        flow = decoder(pooled.flatten(1)).view(-1)+.1*self.core.role_head(groups).view(-1)
        q = prefix['base']+self.core.gain.tanh()*(flow-prefix['base'])
        detail = self.decode(q,torch.cat([pooled.flatten(1),groups],1))
        detail['context'] = context
        return detail

    def forward(self, source, mask, decoder, labels=None, gates=None):
        # Labels never enter predictions, features, or gates.
        prefix = self.prefix(source, mask, decoder)
        off = self.finish_detail(prefix, decoder, torch.zeros(6, device=source.device))
        on = self.finish_detail(prefix, decoder, torch.ones(6, device=source.device) if gates is None else gates)
        self.last = on
        self.last_context = on['context']
        self.last_off = off
        return on['prediction'], off['prediction'], {}, dict(
            same_flow_p0=off['prediction'].detach(), original_pooled_base=prefix['base'].detach(),
            original_regression=on['regression'].detach(), magnitude=on['magnitude'].detach(),
            soft_polarity=on['soft_polarity'].detach(), anchor_clipped=on['anchor_clipped'].detach(),
            mode=self.mode, euler_steps=2, source_statistics_changed=False)

    def proposals(self, source, mask, decoder):
        prefix = self.prefix(source,mask,decoder)
        masks = torch.cat([torch.zeros(1,6,device=source.device),torch.eye(6,device=source.device),torch.ones(1,6,device=source.device)],0)
        detail = [self.finish_detail(prefix,decoder,g) for g in masks]
        values = torch.stack([d['prediction'] for d in detail],1)
        return dict(predictions=values,delta=values-values[:,0,None],masks=masks,
                    p0_is_same_flow_messages_off=True,
                    individual_channel_effects_must_not_be_summed=True,
                    magnitude=torch.stack([d['magnitude'] for d in detail],1),
                    soft_polarity=torch.stack([d['soft_polarity'] for d in detail],1))


def train_objective(flow, labels, role):
    if role != 'TRAIN':
        raise ValueError('Only official TRAIN labels may update this candidate')
    if not isinstance(flow,PolarityIntensityFlow) or not flow.training:
        raise ValueError('Training candidate required')
    y = labels.view(-1)
    d = flow.last
    if y.shape != d['prediction'].shape or not torch.isfinite(y).all():
        raise ValueError('Finite aligned TRAIN targets required')
    regression = F.huber_loss(d['prediction'],y,delta=1.)
    magnitude = F.huber_loss(d['magnitude'],y.abs(),delta=1.)
    # Zero truth has no binary sign target; near-neutral sign has less weight.
    weight = y.abs().clamp(max=1.)
    sign = (F.binary_cross_entropy_with_logits(d['polarity_logit'],(y>0).to(y.dtype),reduction='none')*weight).sum()/weight.sum().clamp_min(1e-12)
    context = d['context'].square().mean()
    parts = dict(regression_huber=regression,magnitude_huber=magnitude,
                 weighted_sign_bce=sign,context_square=context)
    # One fixed definition for both arms. No label- or TEST-selected coefficients.
    return regression+.1*magnitude+.1*sign+.01*context, parts
