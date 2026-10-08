"""Same-capacity heads constrained to the label-free reference sensitivity."""
import torch
from task_gradient_vector_candidate_v3 import TaskGradientVectorFeedback,soft_halfspace_proximal

class JacobianAlignedTaskGradientFeedback(TaskGradientVectorFeedback):
    def forward(self,old_context,pooled_state,reference_prediction,donor_messages,reference_jacobian):
        assert self.train_scale_fitted.item()
        b,d=old_context.shape[0],self.dimension
        assert old_context.shape==pooled_state.shape==reference_jacobian.shape==(b,3,d)
        assert reference_prediction.shape==(b,) and donor_messages.shape==(b,6,d)
        assert all(x.dtype==torch.float32 and torch.isfinite(x).all() for x in [old_context,pooled_state,reference_prediction,donor_messages,reference_jacobian])
        features=torch.cat([pooled_state.flatten(1),old_context.flatten(1),torch.tanh(reference_prediction[:,None]/3)],1).detach()
        unconstrained=torch.stack([head(features) for head in self.gradient_heads],1)
        direction=(reference_jacobian.detach()/self.train_gradient_rms[None,:,None]).detach()
        norm2=direction.square().sum((1,2),keepdim=True)
        coefficient=(unconstrained*direction).sum((1,2),keepdim=True)/norm2.clamp(min=1e-20)
        normalized=coefficient*direction
        predicted=normalized*self.train_gradient_rms[None,:,None]
        used=predicted.detach().repeat_interleave(2,1)
        lam=self.projection_norm2_regularization.repeat_interleave(2)[None,:,None]
        dot=(donor_messages*used).sum(-1);weights=torch.sigmoid(-.125*dot/self.risk_scale)
        if self.mode=='fixed':transformed=.5*donor_messages;actual=torch.full_like(dot,.5)
        elif self.mode=='scalar':transformed=weights[:,:,None]*donor_messages;actual=weights
        else:transformed=.5*soft_halfspace_proximal(donor_messages,used,lam);actual=torch.full_like(dot,.5)
        context=.5*old_context+.25*transformed.reshape(b,3,2,d).sum(2)
        obs={'predicted_gradient':predicted.detach(),'scalar_weights':weights.detach(),'actual_weights':actual.detach(),'estimated_raw_message_dot':dot.detach(),'estimated_transformed_dot':(transformed*used).sum(-1).detach(),'projection_norm2_regularization':lam.detach(),'positive_parallel_retention_ratio':(lam/(used.square().sum(-1,keepdim=True)+lam)).detach(),'estimated_residual':(.5*coefficient.view(-1)).detach(),'reference_jacobian':reference_jacobian.detach(),'unconstrained_normalized_gradient':unconstrained.detach()}
        return context,normalized,obs
