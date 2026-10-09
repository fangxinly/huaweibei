"""Task/regularizer gradients and actual updates versus rounded pure decay.

No forward, random draw, parameter mutation, or backward accumulation occurs
in component_gradients. Autograd work and CPU transfers add diagnostic cost.
Only flow tensors are fully sampled; other tensors use fixed <=512 entries.
Non-decay update is a numerical comparison, not evidence of generalization.
"""
import torch

def fixed_sample_indices(numel, device, full=False):
    # Integer arithmetic avoids float32 endpoint rounding for large encoders.
    count = numel if full else min(numel, 512)
    if count == 0: return torch.empty(0, dtype=torch.long, device=device)
    if full: return torch.arange(numel, dtype=torch.long, device=device)
    return torch.arange(count, dtype=torch.long, device=device) * (numel-1) // max(count-1, 1)

from training_health_v1 import group_name


def component_gradients(model, task_loss, context_penalty):
    named = [(n, p) for n, p in model.named_parameters()
             if p.requires_grad and '.own_flow.' in n]
    result = {}
    for label, loss in [('task', task_loss), ('context_penalty', context_penalty)]:
        grads = torch.autograd.grad(loss, [p for _, p in named],
                                    retain_graph=True, allow_unused=True)
        groups = {}
        for (name, p), grad in zip(named, grads):
            g = groups.setdefault(group_name(name),
                                  dict(numel=0, missing=0, nonzero_tensors=0, square_sum=0.))
            g['numel'] += p.numel()
            if grad is None:
                g['missing'] += 1
                continue
            assert torch.isfinite(grad).all(), name
            squared = float(grad.detach().double().square().sum())
            g['square_sum'] += squared
            g['nonzero_tensors'] += int(squared != 0)
        for g in groups.values():
            g['gradient_RMS'] = (g['square_sum'] / g['numel']) ** .5
        result[label] = groups
    return result


def before_optimizer_step(model, optimizer):
    owners = {id(p): group for group in optimizer.param_groups for p in group['params']}
    result = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        group = owners[id(p)]
        full = '.own_flow.' in name
        indices = fixed_sample_indices(p.numel(), p.device, full)
        result[name] = dict(indices=indices, values=p.detach().flatten()[indices].clone(),
                            lr=float(group['lr']), weight_decay=float(group['weight_decay']),
                            has_gradient=p.grad is not None, full_tensor=full)
    return result


def after_optimizer_step(model, before):
    result = {}
    for name, p in model.named_parameters():
        if name not in before:
            continue
        old = before[name]
        now = p.detach().flatten()[old['indices']]
        pure_decay = old['values'] * (1-old['lr']*old['weight_decay']) if old['has_gradient'] else old['values']
        actual = now.double() - old['values'].double()
        beyond_decay = now.double() - pure_decay.double()
        g = result.setdefault(group_name(name), dict(sample_count=0, full_tensors=0,
                             sampled_tensors=0, changed_values=0, beyond_decay_values=0,
                             actual_square_sum=0., beyond_decay_square_sum=0.))
        g['sample_count'] += now.numel()
        g['full_tensors'] += int(old['full_tensor'])
        g['sampled_tensors'] += int(not old['full_tensor'])
        g['changed_values'] += int((actual != 0).sum())
        g['beyond_decay_values'] += int((beyond_decay != 0).sum())
        g['actual_square_sum'] += float(actual.square().sum())
        g['beyond_decay_square_sum'] += float(beyond_decay.square().sum())
    for g in result.values():
        g['actual_update_RMS'] = (g['actual_square_sum']/g['sample_count'])**.5
        g['versus_rounded_pure_decay_RMS'] = (g['beyond_decay_square_sum']/g['sample_count'])**.5
        g['quantization_sensitive'] = True
    return result
