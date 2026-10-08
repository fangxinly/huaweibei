"""TRAIN-only fixed C2 feasibility diagnostic. No fitting or deployable oracle.

GPU execution is gated by pinned inputs, empty compute inventory, lease reserve,
and a frozen plan. Original scientific modules are imported unchanged.
"""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import argparse, datetime, hashlib, json, shutil, subprocess, sys, time
import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def tensor_sha(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode())
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def solve(feedback, old, messages, p0, rho, terminal, y):
    """Same three updates as frozen controller; save all four feasible candidates."""
    scale = feedback.train_message_rms[None, :, None]
    base = (messages / scale).detach()
    value = base.requires_grad_()
    radius = feedback.trust_fraction * base.norm(dim=(1, 2), keepdim=True)
    predictions, objectives, proxes, relatives, distances, angles = [], [], [], [], [], []
    best = base.detach().clone()
    best_objective = None
    best_index = torch.zeros(len(base), dtype=torch.int64, device=base.device)
    rms2 = feedback.train_residual_rms.double().square()
    beta = feedback.finite_beta.double()
    for step in range(feedback.inner_steps + 1):
        with torch.enable_grad():
            prediction = terminal(feedback.context(old, value * scale))
            delta = prediction - p0
            prox = .5 * (value - base).square().sum((1, 2))
            objective = prox + feedback.finite_beta * feedback.risk_change(delta, rho) / feedback.train_residual_rms.square()
            if step < feedback.inner_steps:
                derivative = torch.autograd.grad(objective.sum(), value)[0]
                assert torch.isfinite(derivative).all()
        # Select oracle candidates in double from unchanged model predictions.
        true_objective = prox.double() + beta * (prediction.double() - y.double()).square() / rms2
        choose = torch.ones(len(base), dtype=torch.bool, device=base.device) if best_objective is None else true_objective < best_objective
        best[choose] = value.detach()[choose]
        best_index[choose] = step
        best_objective = true_objective.detach() if best_objective is None else torch.minimum(best_objective, true_objective.detach())
        shift = value - base
        relative = shift.norm(dim=(1, 2)) / base.norm(dim=(1, 2)).clamp(min=1e-12)
        channel_distance = ((value - base) * scale).norm(dim=2)
        current = value * scale
        cosine = (current * messages).sum(2) / (current.norm(dim=2) * messages.norm(dim=2)).clamp(min=1e-12)
        predictions.append(prediction.detach()); objectives.append(objective.detach())
        proxes.append(prox.detach()); relatives.append(relative.detach())
        distances.append(channel_distance.detach()); angles.append(cosine.clamp(-1, 1).detach())
        assert torch.isfinite(prediction).all() and torch.isfinite(objective).all()
        assert (relative <= feedback.trust_fraction + 1e-5).all()
        if step < feedback.inner_steps:
            proposed = value.detach() - feedback.step_size * derivative.detach()
            shift = proposed - base
            multiplier = (radius / shift.norm(dim=(1, 2), keepdim=True).clamp(min=1e-12)).clamp(max=1)
            value = (base + multiplier * shift).detach().requires_grad_()
    with torch.no_grad():
        best_prediction = terminal(feedback.context(old, best * scale))
    return dict(prediction_path=torch.stack(predictions, 1), objective_path=torch.stack(objectives, 1),
                prox_path=torch.stack(proxes, 1), relative_path=torch.stack(relatives, 1),
                message_distance_path=torch.stack(distances, 1), message_cosine_path=torch.stack(angles, 1),
                best_true_prediction=best_prediction, best_true_step=best_index)


def check_equivalence(feedback, old, messages, p0, y, terminal):
    base = (messages / feedback.train_message_rms[None, :, None]).detach().requires_grad_()
    pred = terminal(feedback.context(old, base * feedback.train_message_rms[None, :, None])).double()
    delta = pred - p0.double(); rho = p0.double() - y.double()
    weight = feedback.finite_beta.double() / feedback.train_residual_rms.double().square()
    direct = weight * (pred - y.double()).square()
    expanded = weight * (2 * rho * delta + delta.square())
    constant = weight * rho.square()
    error = float(((direct - expanded - constant).abs() / (1 + direct.abs() + expanded.abs() + constant.abs())).max())
    gd = torch.autograd.grad(direct.sum(), base, create_graph=True, retain_graph=True)[0]
    ge = torch.autograd.grad(expanded.sum(), base, create_graph=True, retain_graph=True)[0]
    direction = torch.ones_like(base) / np.sqrt(base.numel())
    hd = torch.autograd.grad((gd * direction).sum(), base, retain_graph=True)[0]
    he = torch.autograd.grad((ge * direction).sum(), base)[0]
    grad_error = float((gd - ge).norm() / (1 + gd.norm()))
    hvp_error = float((hd - he).norm() / (1 + hd.norm()))
    assert error < 1e-12 and grad_error < 1e-5 and hvp_error < 1e-5
    assert torch.isfinite(hd).all() and torch.isfinite(he).all()
    return dict(float64_objective_relative_error=error, input_gradient_relative_error=grad_error,
                input_hvp_relative_error=hvp_error, model_dtype='float32, risk algebra float64')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    plan = json.loads(a.plan.read_text())
    assert plan['status'] == 'FROZEN_TRAIN_ORACLE_DIAGNOSTIC_NOT_EXECUTED'
    assert sha(__file__) == plan['diagnostic_source_sha256']
    root = Path(plan['base_root']); formal = Path(plan['formal_root']); run = formal / 'run'
    for path, expected in plan['pinned_files'].items():
        assert sha(path) == expected, path
    assert not a.out.exists(); a.out.mkdir(parents=True)
    q = lambda args: subprocess.check_output(args, text=True).strip()
    gpu = q(['nvidia-smi', '--query-gpu=uuid,name,memory.total', '--format=csv,noheader,nounits'])
    compute = q(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
    assert gpu.split(',')[0] == plan['expected_uuid'] and not compute
    now = datetime.datetime.now(datetime.timezone.utc)
    lease = datetime.datetime.fromisoformat(plan['estimated_lease_end_utc'])
    assert (lease - now).total_seconds() > plan['maximum_seconds'] + plan['preservation_reserve_seconds']
    assert shutil.disk_usage(a.out).free >= plan['minimum_remote_free_bytes']
    sys.path[:0] = [str(formal), str(root)]
    from finite_task_risk_runtime_v2 import FrozenCoordinateLearner, PAIRS
    torch.set_num_threads(2); torch.manual_seed(91817)
    torch.use_deterministic_algorithms(True)
    selection = json.loads((run / 'selection.json').read_text())
    assert selection['epochs'] == 100 and selection['best_epoch'] == 37 and selection['effective_mode'] == 'finite_vector'
    scales = json.loads((formal / 'train_scales.json').read_text())
    learner = FrozenCoordinateLearner(root, 'finite_vector', scales).cuda()
    learner.restore_addon(torch.load(run / 'best_addon.pt', map_location='cpu'))
    learner.eval(); learner.requires_grad_(False)
    before = tensor_sha(learner.state_dict())
    cache_path = root / 'teacher_cache_v1/train_cache.npz'
    cache = np.load(cache_path, allow_pickle=False)
    # No DEV/TEST cache or original data container is opened by this diagnostic.
    assert len(cache['y']) == plan['rows'] == 1281
    mapping = json.loads(Path(plan['mapping_path']).read_text())
    assert len(mapping) == 1281 and [r['row'] for r in mapping] == list(range(1281))
    videos = np.asarray([r['video_id'] for r in mapping])
    assert len(np.unique(videos)) == 52
    permutation = np.roll(np.arange(1281), 1)
    chunks = {}; replay_error = 0.; label_error = 0.; p0_error = 0.
    started = time.monotonic(); started_utc = now.isoformat(); equivalence = None
    def add(name, value):
        array = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value
        chunks.setdefault(name, []).append(array)
    for first in range(0, 1281, plan['batch_size']):
        assert time.monotonic() - started < plan['maximum_seconds'], 'Diagnostic budget exhausted; not COMPLETE'
        ids = np.arange(first, min(first + plan['batch_size'], 1281))
        b = {key: torch.as_tensor(cache[key][ids], device='cuda') for key in
             ['state', 'mask', 'old_context', 'pooled_state', 'reference_prediction', 'y']}
        old = b['old_context']; pool = b['pooled_state']; p0 = b['reference_prediction']; y = b['y']
        terminal = lambda cx: learner.terminal(b['state'], b['mask'], cx)
        with torch.no_grad():
            original, _, obs = learner(b, 'finite_vector')
            replacement = learner(dict(b, y=y.flip(0) + 20), 'finite_vector')[0]
            label_error = max(label_error, float((original - replacement).abs().max()))
            messages = torch.stack([learner.donor[i](torch.cat([pool[:,r], pool[:,d], .5 * (pool[:,r]-pool[:,d]), .5 * (pool[:,r]+pool[:,d])], -1)) for i, (r,d) in enumerate(PAIRS)], 1)
            raw = terminal(learner.feedback.context(old, messages))
            p0_error = max(p0_error, float((terminal(.5 * old) - p0).abs().max()))
        if equivalence is None:
            equivalence = check_equivalence(learner.feedback, old, messages, p0, y, terminal)
        rhos = dict(learned=obs['estimated_residual'], oracle=p0-y, zero=torch.zeros_like(p0),
                    mismatched=p0-torch.as_tensor(cache['y'][permutation[ids]], device='cuda'))
        add('row', ids); add('video', videos[ids]); add('y', y); add('p0', p0); add('raw_prediction', raw)
        add('original_prediction', original); add('estimated_residual', rhos['learned'])
        add('mismatched_label_row', permutation[ids])
        for name, rho in rhos.items():
            result = solve(learner.feedback, old, messages, p0, rho.detach(), terminal, y)
            # Only oracle best-candidate output is exposed; controls use final step.
            if name != 'oracle':
                result.pop('best_true_prediction'); result.pop('best_true_step')
            if name == 'learned':
                replay_error = max(replay_error, float((result['prediction_path'][:,-1]-original).abs().max()))
            add(name + '_residual', rho)
            for key, value in result.items():
                add(name + '_' + key, value)
        torch.cuda.synchronize()
        assert torch.cuda.max_memory_allocated() <= plan['maximum_peak_allocated_bytes']
        progress = dict(rows_finished=int(ids[-1])+1, elapsed_seconds=time.monotonic()-started,
                        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), completed=False)
        (a.out/'progress.json').write_text(json.dumps(progress, indent=2))
        print('TRAIN_ORACLE_PROGRESS', progress['rows_finished'], flush=True)
    assert replay_error <= 2e-6 and label_error == 0 and p0_error < 2e-5
    assert tensor_sha(learner.state_dict()) == before and all(v.grad is None for v in learner.parameters())
    arrays = {key: np.concatenate(value) for key, value in chunks.items()}
    np.savez(a.out/'diagnostics.npz', **arrays)
    receipt = dict(status='TRAIN_ORACLE_DIAGNOSTIC_COMPLETE_NOT_TRAINING', started_utc=started_utc,
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), gpu=gpu, initial_compute=compute,
        argv=sys.argv, pid=os.getpid(), rows=1281, videos=52, elapsed_seconds=time.monotonic()-started,
        parameter_sha_before=before, parameter_sha_after=before, parameter_gradients_none=True,
        cached_learned_replay_error=replay_error, label_replacement_error=label_error, p0_replay_error=p0_error,
        equivalence=equivalence, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        plan_sha256=sha(a.plan), source_sha256=sha(__file__), train_cache_sha256=sha(cache_path),
        diagnostics_sha256=sha(a.out/'diagnostics.npz'), pinned_files=plan['pinned_files'],
        dev_requested=False, test_requested=False, optimizer_created=False, fitted=False,
        scope=plan['scope'])
    (a.out/'receipt.json').write_text(json.dumps(receipt, indent=2))
    print('TRAIN_ORACLE_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
