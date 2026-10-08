"""Zero-initialized signed scalar gates per receiver, trained jointly from scratch.

This is an optimization control, not a calibrated semantic/conflict decision.
"""
import torch
from legacy_flow_model import CONFIG, WholeStateFlow as LegacyWholeStateFlow


class WholeStateFlow(LegacyWholeStateFlow):
    def __init__(self, mode):
        assert mode in ('none', 'state', 'task')
        super().__init__('mid')
        self.mode = mode
        self.context_override = None
        self.feedback_gate = torch.nn.Parameter(torch.zeros(3))
        self._observer_predictions = []

    def update_context(self, old, relation):
        # Each slot here reads only its own receiver. The first invocation
        # precedes all cross-modal velocity conditioning in this model.
        pooled = relation['slots'].mean(dim=2)
        prediction = torch.stack([
            self.unimodal_heads[m](pooled[:, m]).view(-1) for m in range(3)
        ], dim=1)
        self._observer_predictions.append(prediction)
        scaled = torch.tanh(prediction / 3.0)
        state_features, task_features = [], []
        for m, j, k in ((0, 1, 2), (1, 0, 2), (2, 0, 1)):
            state_features.append(torch.cat([
                pooled[:, m], pooled[:, j], pooled[:, k],
                .5 * (pooled[:, j] - pooled[:, k])
            ], dim=-1))
            signals = torch.stack([
                scaled[:, m], scaled[:, j], scaled[:, k],
                .5 * (scaled[:, j] - scaled[:, k])
            ], dim=-1)
            task_features.append(signals.repeat_interleave(CONFIG['dimension'], dim=-1))
        state_features = torch.stack(state_features, dim=1).detach()
        task_features = torch.stack(task_features, dim=1).detach()
        # Both candidate computations run in every mode. These deterministic
        # networks have no dropout or mode-dependent sampling.
        state_update = torch.stack([
            self.feedback[m](state_features[:, m]) for m in range(3)
        ], dim=1)
        task_update = torch.stack([
            self.feedback[m](task_features[:, m]) for m in range(3)
        ], dim=1)
        if len(self._observer_predictions) > 1:
            return old  # Only the first Euler stage supplies the condition.
        mode = self.mode if self.context_override is None else self.context_override
        assert mode in ('none', 'state', 'task')
        selected = {'none': torch.zeros_like(state_update),
                    'state': state_update, 'task': task_update}[mode]
        selected = selected * torch.tanh(self.feedback_gate)[None, :, None]
        return .5 * old + .5 * selected

    def forward(self, source, mask, decoder, labels=None):
        self._observer_predictions = []
        prediction, first, losses, trace = super().forward(source, mask, decoder, labels)
        assert len(self._observer_predictions) == 2
        if self.training:
            observer_loss = (self._observer_predictions[0] - labels.view(-1, 1)).square().mean()
            # Hold the overall auxiliary coefficient at the original .05;
            # split its objective equally between original source and stage1.
            losses['source_unimodal_sentiment'] = losses['unimodal_sentiment']
            losses['stage1_observer_sentiment'] = observer_loss
            losses['unimodal_sentiment'] = .5 * (losses['unimodal_sentiment'] + observer_loss)
        trace['stage1_task_prediction'] = self._observer_predictions[0].detach()
        trace['condition_mode'] = self.mode
        return prediction, first, losses, trace

