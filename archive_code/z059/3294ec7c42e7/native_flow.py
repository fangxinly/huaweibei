"""Experimental coupled representation flow with measured motion responses.

Driver overlap is a local dynamical relation, NOT identified shared semantics.
No semantic branches precede the field. This is not rectified flow matching.
"""
import math
import torch
from torch import nn

PAIRS = ((0, 1), (0, 2), (1, 2))


def pool(states, valid):
    weight = valid[:, None, :, None].to(states.dtype)
    return (states * weight).sum(2) / weight.sum(2).clamp(min=1)


def joint_basis(projectors, sweeps=2):
    """Approximate simultaneous diagonalization, with explicit residuals."""
    batch, _, rank, _ = projectors.shape
    codes = projectors.new_tensor([1, 2, 4])[None, :, None, None]
    _, basis = torch.linalg.eigh((projectors * codes).sum(1))
    for _ in range(sweeps):
        for p in range(rank):
            for q in range(p + 1, rank):
                two = basis[:, :, [p, q]]
                blocks = torch.einsum("bdk,bmde,bel->bmkl", two, projectors, two)
                x = 2 * blocks[:, :, 0, 1]
                y = blocks[:, :, 1, 1] - blocks[:, :, 0, 0]
                gram = torch.stack([torch.stack([(x*x).sum(1), (x*y).sum(1)], -1),
                                    torch.stack([(x*y).sum(1), (y*y).sum(1)], -1)], -2)
                _, eigen = torch.linalg.eigh(gram)
                direction = eigen[:, :, 0]
                direction = direction * torch.where(direction[:, :1] < 0, -1, 1)
                angle = .5 * torch.atan2(direction[:, 1], direction[:, 0])
                angle = torch.where(gram.square().sum((1, 2)) > 1e-20, angle, torch.zeros_like(angle))
                c, s = angle.cos(), angle.sin()
                rotation = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)
                basis[:, :, [p, q]] = two @ rotation
    rotated = torch.einsum("bdi,bmde,bej->bmij", basis, projectors, basis)
    scores = rotated.diagonal(dim1=-2, dim2=-1).clamp(0, 1)
    offdiag = rotated - torch.diag_embed(scores)
    residual = offdiag.square().sum((1, 2))
    return basis, scores, residual


def classify_operators(projectors, rho=.5, sweeps=2, high=.9, low=.1, residual_limit=.01):
    """Select only directions with clear joint support; allow abstention."""
    basis, scores, residual = joint_basis(projectors, sweeps)
    definite = ((scores >= high) | (scores <= low)).all(1) & (residual <= residual_limit)
    active = scores >= high
    bits = (active.to(torch.long) * active.new_tensor([1, 2, 4], dtype=torch.long)[None, :, None]).sum(1)
    bits = torch.where(definite, bits, torch.zeros_like(bits))
    groups, selected, counts, eligible_counts = {}, [], [], []
    for code in range(1, 8):
        flag = bits == code
        groups[code] = (basis * flag[:, None, :]) @ basis.transpose(1, 2)
    for i, j in PAIRS:
        eligible = definite & active[:, i] & active[:, j]
        strength = torch.minimum(scores[:, i], scores[:, j]) * (1 - residual.clamp(max=1))
        order = strength.masked_fill(~eligible, -1).argsort(dim=1, descending=True)
        inverse = order.argsort(dim=1)
        budget = torch.floor(rho * eligible.sum(1)).to(torch.long)
        use = eligible & (inverse < budget[:, None])
        selected.append((basis * use[:, None, :]) @ basis.transpose(1, 2))
        counts.append(use.sum(1))
        eligible_counts.append(eligible.sum(1))
    return {"groups": groups, "pairs": torch.stack(selected, 1), "bits": bits,
            "scores": scores, "residual": residual, "selected_counts": torch.stack(counts, 1),
            "eligible_counts": torch.stack(eligible_counts, 1),
            "unclassified_fraction": (bits == 0).to(projectors.dtype).mean(1)}


class JointVelocity(nn.Module):
    def __init__(self, dimension, hidden=100, bound=.5):
        super().__init__()
        self.dimension, self.bound = dimension, bound
        self.net = nn.Sequential(nn.Linear(7*dimension, hidden), nn.GELU(), nn.Linear(hidden, 3*dimension))
        nn.init.normal_(self.net[-1].weight, std=.001)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, states, t, context):
        batch, _, length, dimension = states.shape
        time = states.new_tensor(t) * 1000 / torch.exp(torch.linspace(0, math.log(10000), dimension, device=states.device, dtype=states.dtype))
        time = time.sin().view(1, 1, dimension).expand(batch, length, -1)
        whole = states.permute(0, 2, 1, 3).flatten(2)
        memory = context.flatten(1)[:, None].expand(-1, length, -1)
        value = self.bound * torch.tanh(self.net(torch.cat([whole, time, memory], -1)))
        return value.reshape(batch, length, 3, dimension).permute(0, 2, 1, 3)


@torch.no_grad()
def measure_responses(field, states, t, context, valid, probes, epsilon=.02,
                      ridge_absolute=1e-8, ridge_relative=1e-4, max_basis=16, sweeps=2, rho=.5):
    """Use every receiver block; averaging velocities would permit cancellation."""
    responses = []
    for driver in range(3):
        columns = []
        for direction in probes.T:
            perturbation = torch.zeros_like(states)
            perturbation[:, driver] = valid[..., None] * direction
            delta = (field(states + epsilon * perturbation, t, context) -
                     field(states - epsilon * perturbation, t, context)) / (2 * epsilon)
            # Concatenate receiver responses as columns in the common feature coordinates.
            columns.append(pool(delta, valid).transpose(1, 2))
        responses.append(torch.cat(columns, -1))
    responses = torch.stack(responses, 1)
    covariance = responses @ responses.transpose(-1, -2)
    _, eigenvectors = torch.linalg.eigh(covariance.sum(1))
    size = min(max_basis, states.shape[-1], responses.shape[-1] * 3)
    candidate = eigenvectors[:, :, -size:]
    reduced = torch.einsum("bdi,bmde,bej->bmij", candidate, covariance, candidate)
    # One common scale plus an absolute floor; never normalize each weak modality to unit energy.
    scale = reduced.diagonal(dim1=-2, dim2=-1).sum((1, 2)) / max(1, 3 * size)
    ridge = ridge_absolute + ridge_relative * scale
    identity = torch.eye(size, device=states.device, dtype=states.dtype)
    projectors = torch.linalg.solve(reduced + ridge[:, None, None, None] * identity, reduced)
    projectors = .5 * (projectors + projectors.transpose(-1, -2))
    relation = classify_operators(projectors, rho, sweeps)
    relation["pairs"] = torch.einsum("bdi,bmij,bej->bmde", candidate, relation["pairs"], candidate)
    relation["groups"] = {code: candidate @ value @ candidate.transpose(1, 2)
                           for code, value in relation["groups"].items()}
    relation["operators"] = torch.einsum("bdi,bmij,bej->bmde", candidate, projectors, candidate)
    relation["response_energy"] = covariance.diagonal(dim1=-2, dim2=-1).sum(-1)
    return relation


def contraction(states, projections, alpha):
    force = torch.zeros_like(states)
    for index, (i, j) in enumerate(PAIRS):
        gap = states[:, i] - states[:, j]
        shift = alpha * torch.einsum("bij,btj->bti", projections[:, index], gap)
        force[:, i] -= shift
        force[:, j] += shift
    return force


def task_guarded_contraction(states, projections, alpha, decoder, valid):
    """Preserve the current scalar prediction to first order within the pair-control range.

    If A is the selected pair graph operator, c=-alpha*A*z and
    c_safe=c-(g.c)/(g.A.g)*A*g. A*g keeps the correction inside selected
    directions and preserves zero modality-sum. The measurement gradient is
    detached; the ordinary force still differentiates through the live states.
    This protects a model prediction, not true semantics or all task information.
    """
    # Evaluation can use no_grad or inference_mode. Clone after disabling inference
    # so the local decoder measurement can still obtain a state gradient.
    with torch.inference_mode(False), torch.enable_grad():
        measured = states.detach().clone().requires_grad_(True)
        measured_valid = valid.clone()
        prediction = decoder(pool(measured, measured_valid).flatten(1)).reshape(-1)
        gradient = torch.autograd.grad(prediction.sum(), measured, create_graph=False)[0].detach()
    raw = contraction(states, projections, alpha)
    reachable_gradient = -contraction(gradient, projections, 1.)
    axes = (1, 2, 3)
    numerator = (gradient * raw).sum(axes)
    denominator = (gradient * reachable_gradient).sum(axes)
    floor = torch.finfo(states.dtype).eps * gradient.square().sum(axes).clamp_min(torch.finfo(states.dtype).tiny)
    active = denominator > floor
    coefficient = torch.where(active, numerator / denominator.clamp_min(floor), torch.zeros_like(numerator))
    safe = raw - coefficient[:, None, None, None] * reachable_gradient
    diagnostics = {
        "raw_first_order_drift": numerator.detach(),
        "safe_first_order_drift": (gradient * safe).sum(axes).detach(),
        "raw_control_work": (states * raw).sum(axes).detach(),
        "safe_control_work": (states * safe).sum(axes).detach(),
        "guard_active": active.detach(),
        "guard_correction_norm": (safe - raw).flatten(1).norm(dim=1).detach(),
    }
    return safe, diagnostics


class NativeFlow(nn.Module):
    def __init__(self, dimension=100, variant="memory", rho=.5, probes=4, max_basis=16,
                 hidden=100, alpha=.1, sweeps=2, task_guard=True):
        super().__init__()
        assert variant in ("instant", "memory") and 0 <= rho <= 1
        self.variant, self.rho, self.alpha = variant, rho, alpha
        self.max_basis, self.sweeps = max_basis, sweeps
        self.task_guard = task_guard
        self.field = JointVelocity(dimension, hidden)
        generator = torch.Generator().manual_seed(9128)
        directions = torch.randn(dimension, min(probes, dimension), generator=generator)
        directions = torch.linalg.qr(directions, mode="reduced")[0]
        self.register_buffer("probes", directions)

    def forward(self, source, valid, decoder, labels=None):
        assert source.ndim == 4 and source.shape[1] == 3 and valid.any(1).all()
        weight = valid[:, None, :, None].to(source.dtype)
        states = source * weight
        context = states.new_zeros((len(states), 3, states.shape[-1]))
        records, controls, first_operators, first_prediction = [], [], [], None
        for step in range(4):
            t = step / 4
            velocity = self.field(states, t, context)
            relation = measure_responses(self.field, states.detach(), t, context.detach(), valid,
                self.probes, max_basis=self.max_basis, sweeps=self.sweeps, rho=self.rho)
            base = (states + .25 * velocity) * weight
            if self.task_guard:
                force, control = task_guarded_contraction(base, relation["pairs"], self.alpha, decoder, valid)
            else:
                force = contraction(base, relation["pairs"], self.alpha)
                control = {"safe_control_work": (base * force).sum((1, 2, 3)).detach()}
            states = (base + .25 * force) * weight
            controls.append(control)
            if step < 2:
                first_operators.append(relation["operators"])
            operators = (torch.stack(first_operators).mean(0) if self.variant == "memory" and step >= 1
                         else relation["operators"])
            # Feedback from measured continuous responses; only reliable directions receive forced alignment.
            context = torch.einsum("bmij,bmj->bmi", operators, pool(states, valid))
            records.append(relation)
            if step == 1:
                first_prediction = decoder(pool(states, valid).flatten(1)).reshape(-1)
        prediction = decoder(pool(states, valid).flatten(1)).reshape(-1)
        trace = {"selected_counts": torch.stack([r["selected_counts"] for r in records]).to(source.dtype).mean(0),
                 "eligible_counts": torch.stack([r["eligible_counts"] for r in records]).to(source.dtype).mean(0),
                 "unclassified_fraction": torch.stack([r["unclassified_fraction"] for r in records]).mean(0),
                 "response_energy": records[-1]["response_energy"],
                 "joint_residual": records[-1]["residual"].mean(1),
                 "context_norm": context.detach().norm(dim=-1).mean(),
                 "continued_state": True, "steps": 4, "variant": self.variant,
                 "task_guard": self.task_guard,
                 "safe_control_work": torch.stack([c["safe_control_work"] for c in controls]).mean(0)}
        if self.task_guard:
            trace.update({name: torch.stack([c[name] for c in controls]).abs().max(0).values
                          for name in ("raw_first_order_drift", "safe_first_order_drift", "guard_correction_norm")})
        return prediction, first_prediction, {}, trace
