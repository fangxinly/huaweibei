"""Unintegrated continuous proximal candidate; no Torch/GPU run or training yet."""
import math
import torch
from torch import nn


def soft_halfspace_proximal(message, estimated_gradient, regularization_norm2):
    """Proximal squared-hinge penalty; positive residual dots are intentional."""
    assert message.shape == estimated_gradient.shape
    assert message.dtype in (torch.float32, torch.float64)
    assert estimated_gradient.dtype == message.dtype
    lam = torch.as_tensor(regularization_norm2, dtype=message.dtype, device=message.device)
    assert torch.isfinite(lam).all() and (lam > 0).all()
    norm2 = estimated_gradient.square().sum(-1, keepdim=True)
    dot = (message * estimated_gradient).sum(-1, keepdim=True)
    assert torch.isfinite(norm2).all() and torch.isfinite(dot).all()
    denominator = norm2 + lam
    assert denominator.shape == norm2.shape and torch.isfinite(denominator).all()
    return message - torch.relu(dot) / denominator * estimated_gradient


class TaskGradientVectorFeedback(nn.Module):
    def __init__(self, mode, *, relative_norm2, dimension=100, risk_scale=.05):
        super().__init__()
        assert mode in ('fixed', 'scalar', 'soft_projected')
        assert dimension > 0 and risk_scale > 0
        assert math.isfinite(relative_norm2) and relative_norm2 > 0
        self.mode = mode
        self.dimension = dimension
        self.relative_norm2 = float(relative_norm2)
        self.risk_scale = risk_scale
        k = 6 * dimension + 1
        self.gradient_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(k), nn.Linear(k, dimension), nn.GELU(),
                          nn.Linear(dimension, dimension)) for _ in range(3)])
        for head in self.gradient_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
        self.register_buffer('train_gradient_rms', torch.ones(3))
        self.register_buffer('projection_norm2_regularization', torch.ones(3))
        self.register_buffer('train_scale_fitted', torch.tensor(False))

    def set_train_fitted_gradient_rms(self, scale):
        """s_m^2=mean designated TRAIN rows of ||a_m||^2/d; no DEV fit."""
        scale = torch.as_tensor(scale, device=self.train_gradient_rms.device,
                                dtype=self.train_gradient_rms.dtype)
        assert scale.shape == (3,) and torch.isfinite(scale).all() and (scale >= 1e-8).all()
        lam = self.relative_norm2 * self.dimension * scale.square()
        assert torch.isfinite(lam).all() and (lam > 0).all()
        with torch.no_grad():
            self.train_gradient_rms.copy_(scale)
            self.projection_norm2_regularization.copy_(lam)
            self.train_scale_fitted.fill_(True)

    def forward(self, old_context, pooled_state, reference_prediction, donor_messages):
        assert self.train_scale_fitted.item(), 'Fit designated TRAIN gradient scales first'
        b, d = old_context.shape[0], self.dimension
        assert old_context.shape == pooled_state.shape == (b, 3, d)
        assert reference_prediction.shape == (b,) and donor_messages.shape == (b, 6, d)
        assert old_context.dtype == pooled_state.dtype == reference_prediction.dtype == donor_messages.dtype == torch.float32
        features = torch.cat([pooled_state.flatten(1), old_context.flatten(1),
                              torch.tanh(reference_prediction[:, None] / 3)], 1).detach()
        normalized_prediction = torch.stack([head(features) for head in self.gradient_heads], 1)
        predicted_gradient = normalized_prediction * self.train_gradient_rms[None, :, None]
        used_gradient = predicted_gradient.detach().repeat_interleave(2, dim=1)
        lam = self.projection_norm2_regularization.repeat_interleave(2)[None, :, None]
        dot = (donor_messages * used_gradient).sum(-1)
        scalar_weights = torch.sigmoid(-.125 * dot / self.risk_scale)
        if self.mode == 'fixed':
            transformed = .5 * donor_messages
            actual_weights = torch.full_like(dot, .5)
        elif self.mode == 'scalar':
            transformed = scalar_weights[:, :, None] * donor_messages
            actual_weights = scalar_weights
        else:
            transformed = .5 * soft_halfspace_proximal(donor_messages, used_gradient, lam)
            actual_weights = torch.full_like(dot, .5)
        context = .5 * old_context + .25 * transformed.reshape(b, 3, 2, d).sum(2)
        norm2 = used_gradient.square().sum(-1, keepdim=True)
        observations = {
            'predicted_gradient': predicted_gradient.detach(),
            'scalar_weights': scalar_weights.detach(),
            # For soft_projected this is outer amplitude, not a scalar account
            # of the complete vector transform.
            'actual_weights': actual_weights.detach(),
            'estimated_raw_message_dot': dot.detach(),
            'estimated_transformed_dot': (transformed * used_gradient).sum(-1).detach(),
            'projection_norm2_regularization': lam.detach(),
            'positive_parallel_retention_ratio': (lam / (norm2 + lam)).detach(),
        }
        return context, normalized_prediction, observations

    def gradient_regression_loss(self, normalized_prediction, train_teacher_gradient):
        assert self.train_scale_fitted.item()
        assert normalized_prediction.shape == train_teacher_gradient.shape
        target = (train_teacher_gradient.detach() / self.train_gradient_rms[None, :, None]).detach()
        return (normalized_prediction - target).square().mean()
