"""Unintegrated candidate module. No GPU mechanism check or training performed."""
import torch
from torch import nn

def estimated_halfspace_projection(message,estimated_gradient,epsilon=1e-12):
    assert message.shape==estimated_gradient.shape
    norm2=estimated_gradient.square().sum(-1,keepdim=True)
    dot=(message*estimated_gradient).sum(-1,keepdim=True)
    active=(norm2>epsilon).to(message.dtype)
    correction=active*torch.relu(dot)/norm2.clamp_min(epsilon)
    return message-correction*estimated_gradient

class TaskGradientVectorFeedback(nn.Module):
    def __init__(self,mode,dimension=100,epsilon=1e-12,risk_scale=.05):
        super().__init__()
        assert mode in ('fixed','scalar','projected') and dimension>0 and epsilon>0 and risk_scale>0
        self.mode=mode;self.dimension=dimension;self.epsilon=epsilon;self.risk_scale=risk_scale
        k=6*dimension+1
        self.gradient_heads=nn.ModuleList([nn.Sequential(nn.LayerNorm(k),nn.Linear(k,dimension),nn.GELU(),nn.Linear(dimension,dimension)) for _ in range(3)])
        for head in self.gradient_heads:
            nn.init.zeros_(head[-1].weight);nn.init.zeros_(head[-1].bias)
        self.register_buffer('train_gradient_rms',torch.ones(3))

    def set_train_fitted_gradient_rms(self,scale):
        """Caller must fit these three scales from designated TRAIN teacher rows only."""
        scale=torch.as_tensor(scale,device=self.train_gradient_rms.device,dtype=self.train_gradient_rms.dtype)
        assert scale.shape==(3,) and torch.isfinite(scale).all() and (scale>=1e-8).all()
        with torch.no_grad():self.train_gradient_rms.copy_(scale)

    def forward(self,old_context,pooled_state,reference_prediction,donor_messages):
        b,d=old_context.shape[0],self.dimension
        assert old_context.shape==pooled_state.shape==(b,3,d)
        assert reference_prediction.shape==(b,) and donor_messages.shape==(b,6,d)
        # No labels or teacher gradients are accepted by the inference interface.
        features=torch.cat([pooled_state.flatten(1),old_context.flatten(1),torch.tanh(reference_prediction[:,None]/3)],1).detach()
        normalized_prediction=torch.stack([head(features) for head in self.gradient_heads],1)
        predicted_gradient=normalized_prediction*self.train_gradient_rms[None,:,None]
        # Task loss trains messages through the transformation, but cannot distort
        # the gradient predictor away from its separate supervised objective.
        used_gradient=predicted_gradient.detach().repeat_interleave(2,dim=1)
        dot=(donor_messages*used_gradient).sum(-1)
        scalar_weights=torch.sigmoid(-.125*dot/self.risk_scale)
        if self.mode=='fixed':
            transformed=.5*donor_messages;actual_weights=torch.full_like(dot,.5)
        elif self.mode=='scalar':
            transformed=scalar_weights[:,:,None]*donor_messages;actual_weights=scalar_weights
        else:
            transformed=.5*estimated_halfspace_projection(donor_messages,used_gradient,self.epsilon)
            actual_weights=torch.full_like(dot,.5)
        # Matches the v5 context map: .5 old + .25 sum(weight_i * F_i).
        context=.5*old_context+.25*transformed.reshape(b,3,2,d).sum(2)
        observations={'predicted_gradient':predicted_gradient.detach(),'scalar_weights':scalar_weights.detach(),
            'actual_weights':actual_weights.detach(),'estimated_raw_message_dot':dot.detach(),
            'estimated_transformed_dot':(transformed*used_gradient).sum(-1).detach(),
            'small_gradient_identity_mask':(used_gradient.square().sum(-1)<=self.epsilon).detach()}
        return context,normalized_prediction,observations

    def gradient_regression_loss(self,normalized_prediction,train_teacher_gradient):
        # Targets supplied only by the TRAIN collection/training loop, separately
        # from forward. Collector and full-model integration still need GPU checks.
        assert normalized_prediction.shape==train_teacher_gradient.shape
        target=(train_teacher_gradient.detach()/self.train_gradient_rms[None,:,None]).detach()
        return (normalized_prediction-target).square().mean()
