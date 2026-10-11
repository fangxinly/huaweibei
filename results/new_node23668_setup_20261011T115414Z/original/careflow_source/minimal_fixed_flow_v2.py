"""Local candidate for the predeclared minimal fixed reference, not GPU certified.

No task-trained weight may initialize this new reference. Constructor pruning
changes task RNG consumption; a new clean initial state is required. The copied
legacy full-state dynamics and analytic singleton patch remain source-pinned.
"""
import torch
from torch import nn
from legacy_flow_model import CONFIG as LEGACY_CONFIG
from legacy_flow_model import WholeStateFlow as LegacyFlow, Velocity, RelationReader
from finite_single_token_reader_v1 import install_for_flow

CONFIG = dict(LEGACY_CONFIG, passes=1, steps=2)
PAIRS = ((0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1))
EXPECTED_AUXILIARY = frozenset(("flow_matching", "cycle_reconstruction",
                                "unimodal_sentiment", "variance_floor"))

class MinimalFixedFlow(LegacyFlow):
    def __init__(self):
        # Construct retained modules directly: no unused legacy feedback, pair,
        # utility, predicted gate, or deterministic counterfactual decoder branch.
        nn.Module.__init__(self)
        self.variant = "mid"
        d = CONFIG["dimension"]
        self.forward_fields = nn.ModuleList([Velocity(d) for _ in range(3)])
        self.backward_fields = nn.ModuleList([Velocity(d) for _ in range(3)])
        self.reader = RelationReader(d, CONFIG["slots"])
        install_for_flow(self)
        self.role_head = nn.Sequential(nn.Linear(7*d, 150), nn.GELU(), nn.Linear(150, 1))
        self.unimodal_heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(3)])
        self.donor_feedback = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(4*d), nn.Linear(4*d, d), nn.GELU(), nn.Linear(d, d))
            for _ in PAIRS])
        for donor in self.donor_feedback:
            nn.init.zeros_(donor[-1].weight)
            nn.init.zeros_(donor[-1].bias)
        self._context_updates = 0
        self._observer_prediction = None

    @staticmethod
    def fixed_context(old, feedback):
        # Same old .5 weights, .5 selected factor, then .5 context factor.
        # From zero context each directed donor contributes exactly .125.
        weights = feedback.new_full((len(old), 6), .5)
        selected = .5*(weights[:, :, None]*feedback).reshape(len(old), 3, 2, -1).sum(2)
        return .5*old + .5*selected

    def update_context(self, old, relation):
        self._context_updates += 1
        if self._context_updates > 1:
            return old
        pooled = relation["slots"].mean(2)
        self._observer_prediction = torch.stack([
            self.unimodal_heads[m](pooled[:, m]).view(-1) for m in range(3)], 1)
        feedback = []
        for index, (own, donor) in enumerate(PAIRS):
            features = torch.cat([pooled[:, own], pooled[:, donor],
                                  .5*(pooled[:, own]-pooled[:, donor]),
                                  .5*(pooled[:, own]+pooled[:, donor])], -1).detach()
            feedback.append(self.donor_feedback[index](features))
        return self.fixed_context(old, torch.stack(feedback, 1))

    def forward(self, source, mask, decoder, labels=None):
        self._context_updates = 0
        self._observer_prediction = None
        prediction, first, auxiliary, trace = LegacyFlow.forward(self, source, mask, decoder, labels)
        assert self._context_updates == 2 and self._observer_prediction is not None
        if self.training:
            assert labels is not None and set(auxiliary) == EXPECTED_AUXILIARY
            source_unimodal = auxiliary["unimodal_sentiment"]
            observer_unimodal = (self._observer_prediction-labels.view(-1, 1)).square().mean()
            auxiliary["unimodal_sentiment"] = .5*source_unimodal + .5*observer_unimodal
            trace["source_unimodal_mse"] = source_unimodal.detach()
            trace["stage1_observer_mse"] = observer_unimodal.detach()
        trace["fixed_weight"] = prediction.new_tensor(.5)
        trace["euler_steps"] = 2
        trace["forward_passes"] = 1
        # Legacy cycle terminal/context and second FM context stay attached.
        # Only the original source/consensus targets and donor input are detached.
        return prediction, first, auxiliary, trace

def objective(prediction, fit_labels, auxiliary):
    assert set(auxiliary) == EXPECTED_AUXILIARY
    return ((prediction.view(-1)-fit_labels.view(-1)).square().mean()
            + .02*auxiliary["flow_matching"]
            + .01*auxiliary["cycle_reconstruction"]
            + .05*auxiliary["unimodal_sentiment"]
            + .01*auxiliary["variance_floor"])
