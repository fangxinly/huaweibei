"""V6: masked whole-sequence flows with the V5 relation/feedback mechanism.

No semantic coordinate split before the flow. Shared query slots read full
states after an Euler update. Geometric memberships are hypotheses, not an
identified information decomposition or calibrated probabilities.
"""
import math
import torch
from torch import nn
import torch.nn.functional as F

CONFIG = {"dimension": 100, "slots": 8, "steps": 2, "passes": 2,
          "query_temperature": .5, "relation_center": .3, "relation_temperature": .2,
          "velocity_bound": .5, "consensus_shift": .25,
          "role_prediction_scale": .1, "first_pass_weight": .1,
          "flow_matching_weight": .02, "cycle_weight": .01,
          "unimodal_weight": .05, "variance_weight": .01, "variance_floor": .1}


def memberships(q):
    """q order: TA, TV, AV; output B,M,S,4 = G,P(m,n),P(m,p),U."""
    qa, qv, qav = q.unbind(dim=1)
    rows = []
    for a, b, d in ((qa, qv, qav), (qa, qav, qv), (qv, qav, qa)):
        global_mass = a * b * d
        pair_first = a * (1 - b) + .5 * a * b * (1 - d)
        pair_second = b * (1 - a) + .5 * a * b * (1 - d)
        unique = (1 - a) * (1 - b)
        rows.append(torch.stack([global_mass, pair_first, pair_second, unique], dim=-1))
    return torch.stack(rows, dim=1)


class Velocity(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.d = d
        self.net = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))
        nn.init.normal_(self.net[-1].weight, std=.001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x, t, context):
        b, length, d = x.shape
        if not torch.is_tensor(t):
            t = x.new_full((b,), t)
        t = t.reshape(-1)
        if len(t) == 1:
            t = t.expand(b)
        frequencies = torch.exp(torch.linspace(0, math.log(10000), d // 2, device=x.device))
        angles = t[:, None] * 1000 / frequencies[None, :]
        time = torch.cat([angles.sin(), angles.cos()], dim=-1)
        inputs = torch.cat([x, time[:, None].expand(-1, length, -1),
                            context[:, None].expand(-1, length, -1)], dim=-1)
        return CONFIG["velocity_bound"] * torch.tanh(self.net(inputs))


class RelationReader(nn.Module):
    def __init__(self, d, slots):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(slots, d) * .1)
        self.key_norm = nn.LayerNorm(d)

    def forward(self, states, mask):
        # states B,3,T,D. Remove within-sequence offset only for geometric
        # keys; values keep the original full state, including the offset.
        valid = mask[:, None, :, None].to(states.dtype)
        mean = (states * valid).sum(2, keepdim=True) / valid.sum(2, keepdim=True).clamp(min=1)
        keys = F.normalize(self.key_norm(states - mean), dim=-1)
        queries = F.normalize(self.queries, dim=-1)
        logits = torch.einsum("bmtd,sd->bmst", keys, queries) / CONFIG["query_temperature"]
        logits = logits.masked_fill(~mask[:, None, None, :], -1e4)
        attention = logits.softmax(dim=-1)
        slots = torch.einsum("bmst,bmtd->bmsd", attention, states)
        centered = slots - slots.mean(dim=2, keepdim=True)
        similarities = torch.stack([F.cosine_similarity(centered[:, i], centered[:, j], dim=-1)
                                    for i, j in ((0, 1), (0, 2), (1, 2))], dim=1)
        q = torch.sigmoid((similarities - CONFIG["relation_center"]) / CONFIG["relation_temperature"])
        mass = memberships(q)
        global_slot = (slots * mass[..., 0, None]).sum(1) / 3
        pair_slots = []
        for i, j, ri, rj in ((0, 1, 1, 1), (0, 2, 2, 1), (1, 2, 2, 2)):
            pair_slots.append((slots[:, i] * mass[:, i, :, ri, None] +
                               slots[:, j] * mass[:, j, :, rj, None]) / 2)
        unique_slots = [slots[:, m] * mass[:, m, :, 3, None] for m in range(3)]
        groups = torch.stack([global_slot] + pair_slots + unique_slots, dim=1)
        # Role values have mass included; retain absence/strength information.
        pooled_groups = groups.mean(dim=2)
        feedback = []
        for m, pair_indices in ((0, (1, 2)), (1, (1, 3)), (2, (2, 3))):
            feedback.append(torch.cat([pooled_groups[:, 0], pooled_groups[:, pair_indices[0]],
                                       pooled_groups[:, pair_indices[1]], pooled_groups[:, 4 + m]], dim=-1))
        return {"slots": slots, "attention": attention, "mass": mass, "q": q,
                "groups": groups, "feedback": torch.stack(feedback, dim=1)}


class WholeStateFlow(nn.Module):
    def __init__(self, variant="cycle"):
        super().__init__()
        assert variant in ("plain", "single", "mid", "cycle")
        self.variant = variant
        d = CONFIG["dimension"]
        self.forward_fields = nn.ModuleList([Velocity(d) for _ in range(3)])
        self.backward_fields = nn.ModuleList([Velocity(d) for _ in range(3)])
        self.reader = RelationReader(d, CONFIG["slots"])
        self.feedback = nn.ModuleList([nn.Sequential(nn.Linear(4 * d, d), nn.GELU(),
                                                   nn.Linear(d, d), nn.Tanh()) for _ in range(3)])
        self.role_head = nn.Sequential(nn.Linear(7 * d, 150), nn.GELU(), nn.Linear(150, 1))
        self.unimodal_heads = nn.ModuleList([nn.Linear(d, 1) for _ in range(3)])

    def update_context(self, old, relation):
        detached = relation["feedback"].detach()
        update = torch.stack([self.feedback[m](detached[:, m]) for m in range(3)], dim=1)
        return .5 * old + .5 * update

    def read_prediction(self, states, relation, decoder):
        base = decoder(self.pool(states).flatten(start_dim=1)).view(-1)
        residual = self.role_head(relation["groups"].mean(dim=2).flatten(start_dim=1)).view(-1)
        return base + CONFIG["role_prediction_scale"] * residual

    def pool(self, states):
        return (states * self.valid[:, None, :, None]).sum(2) / self.valid.sum(1)[:, None, None]

    def state_mse(self, value):
        valid = self.valid[:, None, :, None]
        return (value.square() * valid).sum() / (valid.sum() * value.shape[1] * value.shape[-1])

    @staticmethod
    def consensus_target(source, relation):
        # Targets are formed after the flow has run and read its relations.
        # This target operates on full coordinates, with no pre-flow split.
        slots, mass, attention = relation["slots"], relation["mass"], relation["attention"]
        g = slots.mean(dim=1)
        pairs = [(slots[:, 0] + slots[:, 1]) / 2,
                 (slots[:, 0] + slots[:, 2]) / 2,
                 (slots[:, 1] + slots[:, 2]) / 2]
        shifts = []
        for m, indices in ((0, (0, 1)), (1, (0, 2)), (2, (1, 2))):
            target_slots = (mass[:, m, :, 0, None] * g +
                            mass[:, m, :, 1, None] * pairs[indices[0]] +
                            mass[:, m, :, 2, None] * pairs[indices[1]] +
                            mass[:, m, :, 3, None] * slots[:, m])
            reverse_attention = attention[:, m].transpose(1, 2)
            reverse_attention = reverse_attention / reverse_attention.sum(-1, keepdim=True).clamp(min=1e-8)
            shifts.append(torch.einsum("bts,bsd->btd", reverse_attention, target_slots - slots[:, m]))
        return (source + CONFIG["consensus_shift"] * torch.stack(shifts, dim=1)).detach()

    def forward(self, source, mask, decoder, labels=None):
        assert source.shape[1] == 3 and source.shape[-1] == CONFIG["dimension"]
        assert mask.any(dim=1).all()
        self.valid = mask.to(source.dtype)
        valid_state = self.valid[:, None, :, None]
        source = source * valid_state
        if self.variant == "plain":
            pooled = self.pool(source)
            prediction = decoder(pooled.flatten(start_dim=1)).view(-1)
            losses = {}
            if self.training:
                unimodal = torch.stack([self.unimodal_heads[m](pooled[:, m]).view(-1) for m in range(3)], 1)
                losses = {"flow_matching": source.new_zeros(()), "cycle_reconstruction": source.new_zeros(()),
                          "unimodal_sentiment": (unimodal - labels.view(-1, 1)).square().mean(),
                          "variance_floor": F.relu(CONFIG["variance_floor"] - pooled.std(0, unbiased=False)).mean()}
            return prediction, prediction, losses, {"context_norm": source.new_zeros(())}
        context = source.new_zeros((len(source), 3, source.shape[-1]))
        passes = 2 if self.variant == "cycle" else 1
        records, predictions = [], []
        for iteration in range(passes):
            states = source
            step_contexts = []
            for step in range(2):
                step_contexts.append(context)
                velocity = torch.stack([self.forward_fields[m](states[:, m], step * .5, context[:, m])
                                        for m in range(3)], dim=1)
                states = (states + .5 * velocity) * valid_state
                relation = self.reader(states, mask)
                relation["stage_states"] = states
                if self.variant != "single":
                    context = self.update_context(context, relation)
            prediction = self.read_prediction(states, relation, decoder)
            predictions.append(prediction)
            records.append((states, relation, step_contexts, context))
        losses = {}
        if self.training:
            assert labels is not None
            labels = labels.view(-1)
            fm, cyc = [], []
            for terminal, relation, contexts, final_context in records:
                target = self.consensus_target(source.detach(), relation)
                for step, stage_context in enumerate(contexts):
                    t = source.new_empty((len(source),)).uniform_(step * .5, (step + 1) * .5)
                    mixed = (1 - t[:, None, None, None]) * source.detach() + t[:, None, None, None] * target
                    velocity = torch.stack([self.forward_fields[m](mixed[:, m], t, stage_context[:, m])
                                            for m in range(3)], dim=1)
                    fm.append(self.state_mse(velocity - (target - source.detach())))
                reconstructed = terminal
                for step in range(2):
                    backward = torch.stack([self.backward_fields[m](reconstructed[:, m], step * .5,
                                                                    final_context[:, m]) for m in range(3)], dim=1)
                    reconstructed = (reconstructed + .5 * backward) * valid_state
                cyc.append(self.state_mse(reconstructed - source.detach()))
            pooled = self.pool(source)
            unimodal = torch.stack([self.unimodal_heads[m](pooled[:, m]).view(-1) for m in range(3)], dim=1)
            variance = pooled.std(dim=0, unbiased=False)
            losses = {"flow_matching": torch.stack(fm).mean(), "cycle_reconstruction": torch.stack(cyc).mean(),
                      "unimodal_sentiment": (unimodal - labels[:, None]).square().mean(),
                      "variance_floor": F.relu(CONFIG["variance_floor"] - variance).mean()}
        final, relation, _, context = records[-1]
        trace = {"mass": relation["mass"].detach().mean(dim=(0, 2)),
                 "q": relation["q"].detach().mean(dim=(0, 2)),
                 "context_norm": context.detach().norm(dim=-1).mean(),
                 "source_variance": self.pool(source.detach()).var(dim=0, unbiased=False).mean(-1),
                 "terminal_variance": self.pool(final.detach()).var(dim=0, unbiased=False).mean(-1),
                 "first_prediction": predictions[0].detach(), "final_prediction": predictions[-1].detach()}
        return predictions[-1], predictions[0], losses, trace


def v5_forward(self, input_ids, visual, acoustic, label_ids=None, input_mask=None):
    mask = input_mask.bool() if input_mask is not None else torch.ones_like(input_ids, dtype=torch.bool)
    # Match the released encoder settings: no text attention mask unless opted in.
    hidden = self.model(input_ids, attention_mask=input_mask if self.use_attention_mask else None)[0]
    text = self.LayerNorm_l(self.proj_l(hidden))
    audio = self.proj_a(acoustic.transpose(1, 2)).permute(2, 0, 1)
    vision = self.proj_v(visual.transpose(1, 2)).permute(2, 0, 1)
    audio = self.LayerNorm_a(self.transa(audio).permute(1, 0, 2))
    vision = self.LayerNorm_v(self.transv(vision).permute(1, 0, 2))
    source = torch.stack([text, audio, vision], dim=1)
    prediction, first, losses, trace = self.own_flow(source, mask,
                                                   lambda x: self.predictor(self.fusion(x)), label_ids)
    self.last_first_prediction = first
    self.last_losses = losses
    self.last_trace = trace
    return prediction[:, None], prediction.new_zeros(()), prediction.new_zeros(())


def check_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(128)
    q = torch.rand(4, 3, 8)
    mass = memberships(q)
    assert torch.allclose(mass.sum(-1), torch.ones_like(mass[..., 0]), atol=1e-6)
    assert (mass >= 0).all()
    for values, expected in ((1, 0), (0, 3)):
        m = memberships(torch.full((2, 3, 8), float(values)))
        assert torch.equal(m[..., expected], torch.ones_like(m[..., expected]))
    source = torch.randn(4, 3, 10, 100, requires_grad=True)
    mask = torch.ones(4, 10, dtype=torch.bool)
    mask[:, -2:] = False
    decoder = nn.Linear(300, 1)
    model = WholeStateFlow("cycle")
    prediction, first, losses, trace = model(source, mask, decoder, torch.randn(4))
    loss = prediction.square().mean() + sum(losses.values()) + first.square().mean()
    loss.backward()
    for name, parameter in {"source": source, "queries": model.reader.queries,
                            "velocity": model.forward_fields[0].net[-1].weight,
                            "feedback": model.feedback[0][0].weight,
                            "backward": model.backward_fields[0].net[-1].weight}.items():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0, name
    model.eval()
    with torch.no_grad():
        own = model(source.detach(), mask, decoder)[0]
        permutation = torch.randperm(4)
        replay = model(source.detach()[permutation], mask[permutation], decoder)[0]
        assert torch.allclose(replay, own[permutation], atol=1e-5, rtol=1e-5)
        relation = model.reader(source.detach(), mask)
        assert relation["attention"][..., -2:].abs().max() == 0
    print("V5_CPU_CHECK_COMPLETE", {"loss": float(loss), "first_final_delta": float((own - first.detach()).abs().mean())})


if __name__ == "__main__":
    check_cpu()
