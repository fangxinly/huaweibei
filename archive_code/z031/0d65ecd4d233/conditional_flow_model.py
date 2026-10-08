"""One-pass V6 conditions: zero, ordinary state, or task predictions.

All modes construct and execute the same observer and context projections.
Task signals are predictions, not identified shared/complementary factors.
Labels enter auxiliary training losses only. No teacher or hard risk gate.
"""
import torch
from legacy_flow_model import CONFIG, WholeStateFlow as LegacyWholeStateFlow


class WholeStateFlow(LegacyWholeStateFlow):
    def __init__(self, mode):
        assert mode in ('none', 'state', 'task')
        super().__init__('mid')
        self.mode = mode
        self.context_override = None
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


def check_synthetic_cpu():
    """Meaningful checks for matched modes and actual condition use, not MSA."""
    torch.set_num_threads(2)
    torch.manual_seed(91812)
    source = torch.randn(5, 3, 11, CONFIG['dimension'])
    mask = torch.ones(5, 11, dtype=torch.bool)
    mask[:, -3:] = False
    decoder = torch.nn.Linear(3 * CONFIG['dimension'], 1)
    labels = torch.tensor([-2., -1., 0., 1., 2.])
    models = {}
    for mode in ('none', 'state', 'task'):
        torch.manual_seed(91811)
        models[mode] = WholeStateFlow(mode)
    states = [model.state_dict() for model in models.values()]
    assert all(set(state) == set(states[0]) for state in states)
    assert all(torch.equal(states[0][key], state[key]) for state in states[1:] for key in states[0])
    rows = []
    predictions = {}
    for mode, model in models.items():
        model.train()
        torch.manual_seed(91813)
        prediction, first, losses, trace = model(source, mask, decoder, labels)
        objective = ((prediction - labels)**2).mean()
        for name, coefficient in (('flow_matching', .02), ('cycle_reconstruction', .01),
                                  ('unimodal_sentiment', .05), ('variance_floor', .01)):
            objective = objective + coefficient * losses[name]
        objective.backward()
        for parameter in (model.forward_fields[0].net[-1].weight,
                          model.backward_fields[0].net[-1].weight,
                          model.unimodal_heads[0].weight, model.reader.queries):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
        feedback = model.feedback[0][0].weight.grad
        if mode == 'none':
            assert feedback is None or torch.count_nonzero(feedback) == 0
        else:
            assert feedback is not None and torch.isfinite(feedback).all() and feedback.abs().sum() > 0
        model.eval()
        with torch.no_grad():
            own = model(source, mask, decoder)[0]
            altered = source.clone()
            altered[:, :, -3:] = 1000
            assert torch.allclose(own, model(altered, mask, decoder)[0], atol=1e-6)
            perm = torch.tensor([4, 2, 0, 3, 1])
            assert torch.allclose(own[perm], model(source[perm], mask[perm], decoder)[0], atol=1e-6)
            model.context_override = 'none'
            off = model(source, mask, decoder)[0]
            model.context_override = None
        predictions[mode] = own
        if mode == 'none':
            assert torch.equal(own, off)
        else:
            assert (own - off).abs().max() > 1e-9
        rows.append({'mode': mode, 'parameters': sum(p.numel() for p in model.parameters()),
                     'objective': float(objective.detach()),
                     'context_norm': float(trace['context_norm']),
                     'frozen_condition_off_maxabs': float((own-off).abs().max()),
                     'feedback_gradient_nonzero': mode != 'none'})
    assert (predictions['state'] - predictions['task']).abs().max() > 1e-9
    return {'status': 'SYNTHETIC_STRUCTURE_CHECKS_PASSED', 'same_initial_parameters': True,
            'padding_and_batch_permutation_verified': True, 'rows': rows,
            'scope': 'New small flow modules only; no MOSI, old checkpoints or encoder execution.'}


if __name__ == '__main__':
    import json
    print(json.dumps(check_synthetic_cpu()), flush=True)
