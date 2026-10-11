"""Label-free finite-risk scalar control and differentiable vector feedback.

Candidate implementation, not a trained model or a guarantee of true risk.
Only the TRAIN-fitted residual estimator is supervised; the controller gets no y.
"""
import torch
from torch import nn
from task_gradient_vector_candidate_v3 import TaskGradientVectorFeedback


class FiniteTaskRiskFeedback(TaskGradientVectorFeedback):
    def __init__(self, mode, *, inner_steps=3, step_size=.25, trust_fraction=.25):
        super().__init__('fixed', relative_norm2=.1)
        assert mode in ('fixed','finite_scalar','finite_vector')
        self.mode=mode; self.inner_steps=inner_steps; self.step_size=step_size
        self.trust_fraction=trust_fraction
        self.register_buffer('train_residual_rms',torch.ones(()))
        self.register_buffer('train_message_rms',torch.ones(6))
        self.register_buffer('finite_beta',torch.ones(()))
        self.register_buffer('finite_scale_fitted',torch.tensor(False))

    def set_train_scales(self, residual_rms, message_rms, beta):
        assert residual_rms>1e-6 and beta>0
        v=torch.as_tensor(message_rms,device=self.train_message_rms.device,
                          dtype=self.train_message_rms.dtype)
        assert v.shape==(6,) and (v>1e-6).all() and torch.isfinite(v).all()
        with torch.no_grad():
            self.train_residual_rms.fill_(residual_rms)
            self.train_message_rms.copy_(v);self.finite_beta.fill_(beta)
            self.finite_scale_fitted.fill_(True)

    def estimate_residual(self, old, pool, p0):
        features=torch.cat([pool.flatten(1),old.flatten(1),torch.tanh(p0[:,None]/3)],1).detach()
        # Same 214506 matched head parameters. Shared scalar conditional residual.
        normalized=torch.stack([head(features) for head in self.gradient_heads],1).mean((1,2))
        return normalized, normalized*self.train_residual_rms

    def residual_loss(self, normalized, train_residual, fit_mask):
        target=(train_residual.detach()/self.train_residual_rms).detach()
        if not fit_mask.any():return normalized.sum()*0
        return (normalized[fit_mask]-target[fit_mask]).square().mean()

    @staticmethod
    def context(old, messages):
        # messages are raw F: fixed baseline amplitude is .5 and mixing is .25.
        return .5*old+.125*messages.reshape(len(old),3,2,-1).sum(2)

    @staticmethod
    def risk_change(delta, rho):return 2*rho*delta+delta.square()

    def control(self, old, messages, p0, rho, terminal, mode=None):
        assert self.finite_scale_fitted.item()
        mode=self.mode if mode is None else mode
        assert mode in ('fixed','finite_scalar','finite_vector')
        rho=rho.detach();p0=p0.detach();old=old.detach()
        default=self.context(old,messages)
        if mode=='fixed':
            return default,{'actual_weights':messages.new_full(messages.shape[:2],.5),
                            'estimated_residual':rho}
        if mode=='finite_scalar':
            with_delta=terminal(default)-p0
            with_risk=self.risk_change(with_delta,rho)
            utilities=[]
            for i in range(6):
                keep=messages.new_ones((1,6,1));keep[:,i]=0
                without=terminal(self.context(old,messages*keep))-p0
                utilities.append(self.risk_change(without,rho)-with_risk)
            utility=torch.stack(utilities,1)
            # TRAIN residual squared scale; no DEV tuning or inference labels.
            weights=torch.sigmoid(utility/(.05*self.train_residual_rms.square()))
            ctx=.5*old+.25*(weights[:,:,None]*messages).reshape(len(old),3,2,-1).sum(2)
            return ctx,{'actual_weights':weights,'estimated_finite_utility':utility,
                        'estimated_residual':rho,'default_delta':with_delta}
        outer_grad=torch.is_grad_enabled()
        scale=self.train_message_rms[None,:,None]
        with torch.enable_grad():
            base=messages/scale
            if not outer_grad:base=base.detach()
            value=base if base.requires_grad else base.detach().requires_grad_()
            radius=self.trust_fraction*base.norm(dim=(1,2),keepdim=True)
            losses=[]
            for _ in range(self.inner_steps):
                delta=terminal(self.context(old,value*scale))-p0
                prox=.5*(value-base).square().sum((1,2))
                objective=prox+self.finite_beta*self.risk_change(delta,rho)/self.train_residual_rms.square()
                derivative=torch.autograd.grad(objective.sum(),value,
                                               create_graph=outer_grad)[0]
                assert torch.isfinite(derivative).all()
                proposed=value-self.step_size*derivative
                shift=proposed-base
                multiplier=(radius/shift.norm(dim=(1,2),keepdim=True).clamp(min=1e-12)).clamp(max=1)
                value=base+multiplier*shift
                losses.append(objective.detach())
                if not outer_grad:value=value.detach().requires_grad_()
            controlled=value*scale
            ctx=self.context(old,controlled)
            relative=(value-base).norm(dim=(1,2))/base.norm(dim=(1,2)).clamp(min=1e-12)
        if not outer_grad:ctx=ctx.detach();controlled=controlled.detach()
        return ctx,{'actual_weights':messages.new_full(messages.shape[:2],.5),
                    'estimated_residual':rho,'controlled_message':controlled,
                    'trust_relative_change':relative.detach(),
                    'inner_objectives':torch.stack(losses,1)}

    def forward(self,old,pool,p0,messages,terminal,mode=None):
        normalized,rho=self.estimate_residual(old,pool,p0)
        context,obs=self.control(old,messages,p0,rho,terminal,mode)
        obs['message']=messages.detach();obs['context']=context.detach()
        return context,normalized,obs
