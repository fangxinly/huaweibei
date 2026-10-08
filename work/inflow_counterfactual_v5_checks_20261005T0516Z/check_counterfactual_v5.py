"""Matched full-sequence conditional flow pilot. TRAIN/dev only; no TEST access."""
from pathlib import Path
from types import MethodType
import argparse
import importlib
import json
import os
import subprocess
import sys
import time
import numpy as np
import torch
from counterfactual_flow_model import CONFIG, WholeStateFlow
import hashlib
from encoder_adapter import fit_statistics, install, forward_v6, forward_batch


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()

def objective(prediction, labels, auxiliary):
    return ((prediction.view(-1) - labels.view(-1)).square().mean()
            + CONFIG["flow_matching_weight"] * auxiliary["flow_matching"]
            + CONFIG["cycle_weight"] * auxiliary["cycle_reconstruction"]
            + CONFIG["unimodal_weight"] * auxiliary["unimodal_sentiment"]
            + CONFIG["variance_weight"] * auxiliary["variance_floor"]
            + CONFIG["pair_weight"] * auxiliary["pair_sentiment"]
            + CONFIG["utility_weight"] * auxiliary["utility_calibration"])

def train_epoch(author, model, loader, optimizer, scheduler):
    model.train()
    totals = {}
    for batch in loader:
        batch = tuple(x.to(author.DEVICE) for x in batch)
        prediction, _, _ = author._forward_eval(model, batch)
        labels = batch[3].view(-1)
        final_task = (prediction.view(-1) - labels).square().mean()
        first_task = (model.dberta.last_first_prediction - labels).square().mean()
        cycle = model.dberta.own_flow.variant == "cycle"
        weight = CONFIG["first_pass_weight"] if cycle else 0.0
        task = (final_task + weight * first_task) / (1 + weight)
        auxiliary = model.dberta.last_losses
        loss = (task + CONFIG["flow_matching_weight"] * auxiliary["flow_matching"]
                + CONFIG["cycle_weight"] * auxiliary["cycle_reconstruction"]
                + CONFIG["unimodal_weight"] * auxiliary["unimodal_sentiment"]
                + CONFIG["variance_weight"] * auxiliary["variance_floor"]
            + CONFIG["pair_weight"] * auxiliary["pair_sentiment"]
            + CONFIG["utility_weight"] * auxiliary["utility_calibration"])
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        values = {"train_total": loss, "task_mse": task, "first_task_mse": first_task,
                  **auxiliary, "context_norm": model.dberta.last_trace["context_norm"]}
        for name, value in values.items():
            totals[name] = totals.get(name, 0.0) + float(value.detach())
    return {name: value / len(loader) for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("check",), required=True)
    parser.add_argument("--mode", choices=("none", "fixed", "predicted"), required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-initial-sha")
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=91814)
    cli = parser.parse_args()
    for name in ("baseline_root", "repo", "backbone", "data", "out"):
        setattr(cli, name, getattr(cli, name).resolve())
    assert cli.seed == 91814 and cli.epochs == 100
    assert not cli.out.exists(), "Unique check/run output required"
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"]).decode().strip()
    identity, memory = [value.strip() for value in gpu.split(",")]
    assert identity == cli.expected_gpu_uuid and int(memory) <= 16, gpu
    compute = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"]).decode().strip()
    assert not compute, compute
    cli.out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(cli.baseline_root))
    helpers = importlib.import_module("run_careflow")
    control = importlib.import_module("run_control_baseline")
    assert helpers.digest(cli.data) == helpers.DATA_SHA
    assert subprocess.check_output(["git", "-C", str(cli.repo), "rev-parse", "HEAD"]).decode().strip() == helpers.COMMIT
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(2)
    author, arguments = helpers.load_author(cli)
    author._forward_eval = forward_batch
    author.set_random_seed(cli.seed)
    train_data = helpers.dataset(author, cli.data, "train")
    valid_data = helpers.dataset(author, cli.data, "dev")
    statistics = fit_statistics(train_data)
    assert len(train_data) == 1281 and len(valid_data) == 229
    order_rng = torch.Generator().manual_seed(cli.seed + 1327)
    orders = torch.stack([torch.randperm(len(train_data), generator=order_rng)[:1280] for _ in range(cli.epochs)]).numpy().astype("<i8")
    np.save(cli.out / "batch_orders.npy", orders, allow_pickle=False)
    def epoch_loader(epoch):
        batches = orders[epoch].reshape(40, 32).tolist()
        return torch.utils.data.DataLoader(train_data, batch_sampler=batches)
    train_loader = epoch_loader(0)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=128, shuffle=False)
    steps = int(len(train_data) / 32) * cli.epochs
    model, old_optimizer, old_scheduler = author.prep_for_training(steps)
    pretrained = helpers.pretrained_check(model, cli.backbone)
    core = model.dberta
    for name in ("reflow_a", "reflow_v", "reflow_a_b", "reflow_v_b", "rf_a", "rf_v", "rf_a_b", "rf_v_b"):
        delattr(core, name)
    core.own_flow = WholeStateFlow(cli.mode).to(author.DEVICE)
    install(core, statistics)
    core.to(author.DEVICE)
    # Released buffers contain an uninitialized scalar; only type_as reads their dtype.
    for encoder in (core.transa, core.transv):
        encoder.embed_positions._float_tensor.zero_()
    core.forward = MethodType(forward_v6, core)
    del old_optimizer, old_scheduler
    optimizer, scheduler = control.optimizer_for(author, model, steps)
    own_files = [Path(__file__).with_name(name) for name in ("check_counterfactual_v5.py", "counterfactual_flow_model.py", "legacy_flow_model.py", "encoder_adapter.py")]
    manifest = json.loads(Path(__file__).with_name("source_manifest.json").read_text())
    assert {p.name: helpers.digest(p) for p in own_files} == manifest["source_sha256"]
    def tensor_state_sha(module):
        sha = hashlib.sha256()
        for name, value in sorted(module.state_dict().items()):
            sha.update(name.encode())
            sha.update(str((tuple(value.shape), str(value.dtype))).encode())
            sha.update(value.detach().cpu().contiguous().numpy().tobytes())
        return sha.hexdigest()
    initial_state_sha = tensor_state_sha(model)
    initial_flow_sha = tensor_state_sha(core.own_flow)
    if cli.stage == "run":
        assert cli.expected_initial_sha == initial_state_sha, "Training initialization differs from checked initialization"
    helper_files = [cli.baseline_root / "run_careflow.py", cli.baseline_root / "run_control_baseline.py"]
    author_files = subprocess.check_output(["git", "-C", str(cli.repo), "ls-files"]).decode().splitlines()
    protocol = {"name": "inflow_counterfactual_v5_check", "mode": cli.mode, "config": CONFIG,
                "initial_model_sha256": initial_state_sha, "initial_flow_sha256": initial_flow_sha,
                "batch_order_sha256": digest_bytes(orders.tobytes()),
                "batch_order_epoch_sha256": [digest_bytes(row.tobytes()) for row in orders],
                "updates_per_epoch": 40, "total_updates": 4000,
                "gpu_uuid": identity,
                "placeholder_buffers": "Two positional embedding _float_tensor scalars zeroed; only dtype used by type_as",
                "parent_pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                "condition": "After first Euler stage: none / fixed .5 / predicted sigmoid(4u), six directed detached-state donor feedback channels; zero feedback output init",
                "utility_config": "TRAIN final-task marginal g=(p_without_i-y)^2-(p_ref-y)^2 with fixed reference .5; tanh(g/.05), detached target/input, balanced SmoothL1 .01; first10epochs shared fixed feedback; 20 mechanism updates only; 10 deterministic references training/1 inference; no label in u input",
                "observer_auxiliary": "same .05 total coefficient: half original source predictions, half stage1 predictions in every mode",
                "seed": cli.seed, "epochs": cli.epochs, "author_arguments": arguments,
                "author_commit": helpers.COMMIT,
                "own_source_sha256": {p.name: helpers.digest(p) for p in own_files},
                "helper_source_sha256": {p.name: helpers.digest(p) for p in helper_files},
                "author_source_sha256": {n: helpers.digest(cli.repo / n) for n in author_files},
                "data_sha256": helpers.DATA_SHA,
                "backbone_sha256": {p.name: helpers.digest(p) for p in cli.backbone.iterdir() if p.is_file()},
                "pretrained_tensors_verified": pretrained,
                "train_samples": len(train_data), "valid_samples": len(valid_data),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "flow_parameters": sum(p.numel() for p in core.own_flow.parameters()),
                "selection": "author mean of validation batch MSE, batch128",
                "optimizer": "author AdamW groups, lr1e-5, linear warmup10%, 100epochs",
                "encoder": "same encoder weights/architecture; text attention mask, A/V key padding mask, content-only pooling",
                "reader_mask": "content mask excludes CLS/SEP/padding and applies to states, velocities, losses, pooling and reading",
                "normalization": "per-channel train-aligned-token mean/std; constant channels zero; no batch-dependent minmax",
                "normalization_statistics": {m: {k: v.tolist() for k, v in values.items()} for m, values in statistics.items()},
                "deviations": ["Custom V6 encoder/flow, not unmodified official CaReFlow", "One pass/two Euler steps; stage1 condition also enters legacy FM and backward training objectives", "Same directed feedback/head architecture and objectives; only conditioning weights differ", "Geometric consensus/roles retained as hypotheses, not identified semantics", "All dev observations exploratory; no test-based selection"],
                "data_access": "TRAIN/dev requested from pickle container that also contains TEST; TEST entries never indexed or converted",
                "test_policy": "TEST forbidden; selected checkpoint end dev/default and predetermined frozen condition-off only"}
    helpers.write(cli.out / "protocol.json", protocol)
    assert cli.stage=='check'
    core.own_flow.set_epoch(11)
    batch=tuple(x.to(author.DEVICE) for x in next(iter(train_loader)))
    model.eval()
    with torch.no_grad():
        neutral=(*batch[:3],torch.zeros_like(batch[3]),batch[4])
        changed=(*batch[:3],torch.full_like(batch[3],123),batch[4])
        baseline=forward_batch(model,neutral)[0]
        original_u=core.own_flow._stage1['utility'].clone()
        original_w=core.own_flow._stage1['weights'].clone()
        replay=forward_batch(model,changed)[0]
        assert torch.equal(baseline,replay)
        assert torch.equal(original_u,core.own_flow._stage1['utility']) and torch.equal(original_w,core.own_flow._stage1['weights'])
        reference_u=core.own_flow._stage1['reference_prediction'].clone()
        for override in ('none','fixed','predicted'):
            core.own_flow.context_override=override
            pred=forward_batch(model,neutral)[0]
            assert torch.allclose(pred,baseline,atol=1e-6,rtol=1e-6)
        core.own_flow.context_override=None
    assert tensor_state_sha(model)==initial_state_sha
    core.own_flow.set_epoch(1)
    model.train()
    flags={id(m):m.training for m in model.modules()}
    optimizer.zero_grad(set_to_none=True)
    pred,_,_=forward_batch(model,batch)
    assert {id(m):m.training for m in model.modules()}==flags
    stage=core.own_flow._stage1
    assert stage['effective_mode']=='fixed'
    assert torch.max(torch.abs(stage['raw_utility']))==0
    assert not stage['reference_prediction'].requires_grad and not stage['without_prediction'].requires_grad and not stage['target'].requires_grad
    initial_zero_target=True
    initial_gradient=torch.autograd.grad(core.last_losses['utility_calibration'],[core.own_flow.utility_heads[0][-1].weight,core.own_flow.donor_feedback[0][-1].weight,core.proj_a.weight,core.predictor.weight if hasattr(core.predictor,'weight') else next(core.predictor.parameters())],allow_unused=True,retain_graph=True)
    assert initial_gradient[0] is not None and initial_gradient[0].abs().sum()==0
    assert all(g is None for g in initial_gradient[1:])
    # Consume the retained probe graph before the first independent update.
    objective(pred,batch[3],core.last_losses).backward()
    optimizer.zero_grad(set_to_none=True)
    del pred,stage,initial_gradient
    torch.cuda.empty_cache()
    timings=[];seen_nonzero=False;seen_utility_gradient=False;first_feedback_gradient=None
    train_batches=list(train_loader)
    for step in range(20):
        batch=tuple(x.to(author.DEVICE) for x in train_batches[step])
        flags={id(m):m.training for m in model.modules()}
        torch.cuda.synchronize();started=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        pred,_,_=forward_batch(model,batch)
        assert {id(m):m.training for m in model.modules()}==flags
        stage=core.own_flow._stage1
        assert stage['effective_mode']=='fixed' and torch.equal(stage['weights'],torch.full_like(stage['weights'],.5))
        target_check=((stage['without_prediction']-batch[3].view(-1,1)).square()-(stage['reference_prediction'][:,None]-batch[3].view(-1,1)).square()).detach()
        assert torch.equal(target_check,stage['raw_utility'])
        assert not stage['reference_prediction'].requires_grad and not stage['without_prediction'].requires_grad
        assert not stage['joint_without_prediction'].requires_grad and not stage['target'].requires_grad
        assert torch.equal(stage['interaction_residual'],stage['joint_utility']-stage['raw_utility'].reshape(len(batch[3]),3,2).sum(2))
        utility_grads=torch.autograd.grad(core.last_losses['utility_calibration'],[core.own_flow.utility_heads[0][-1].weight,core.own_flow.donor_feedback[0][-1].weight,core.proj_a.weight,next(core.predictor.parameters())],allow_unused=True,retain_graph=True)
        assert all(g is None for g in utility_grads[1:])
        seen_nonzero|=bool(stage['raw_utility'].abs().sum()>0)
        seen_utility_gradient|=bool(utility_grads[0] is not None and utility_grads[0].abs().sum()>0)
        loss=objective(pred,batch[3],core.last_losses);assert torch.isfinite(loss);loss.backward()
        if step==0:
            first_feedback_gradient=float(core.own_flow.donor_feedback[0][-1].weight.grad.abs().sum())
            assert first_feedback_gradient>0
        for parameter in [core.model.embeddings.word_embeddings.weight,core.proj_a.weight,core.proj_v.weight,core.own_flow.forward_fields[0].net[-1].weight,core.own_flow.pair_heads[0][-1].weight]:
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum()>0
        optimizer.step();scheduler.step();torch.cuda.synchronize();timings.append(time.monotonic()-started)
        print(json.dumps({'check_step':step+1,'seconds':timings[-1],'target_nonzero_fraction':float((stage['raw_utility']!=0).float().mean()),'utility_gradient_nonzero':bool(utility_grads[0].abs().sum()>0)}),flush=True)
    assert seen_nonzero and seen_utility_gradient
    hidden_gradient=core.own_flow.donor_feedback[0][1].weight.grad
    assert hidden_gradient is not None and torch.isfinite(hidden_gradient).all() and hidden_gradient.abs().sum()>0
    after_updates=tensor_state_sha(model);assert after_updates!=initial_state_sha
    core.own_flow.set_epoch(11);model.eval();state_before=tensor_state_sha(model)
    small=tuple(x[:4] for x in batch)
    with torch.no_grad():
        neutral=(*small[:3],torch.zeros_like(small[3]),small[4])
        changed=(*small[:3],torch.full_like(small[3],123),small[4])
        natural=forward_batch(model,neutral)[0];u=core.own_flow._stage1['utility'].clone();w=core.own_flow._stage1['weights'].clone()
        replay=forward_batch(model,changed)[0];assert torch.equal(natural,replay) and torch.equal(u,core.own_flow._stage1['utility']) and torch.equal(w,core.own_flow._stage1['weights'])
        assert core.own_flow._stage1['effective_mode']==cli.mode
        solo=torch.cat([forward_batch(model,tuple(x[i:i+1] for x in neutral))[0] for i in range(4)])
        assert torch.allclose(natural,solo,atol=2e-5,rtol=2e-5)
        perturb=[x.clone() for x in neutral];invalid=~perturb[4].bool();perturb[0][invalid]=1
        for i in (1,2):perturb[i][invalid]=1000
        assert torch.allclose(natural,forward_batch(model,tuple(perturb))[0],atol=2e-5,rtol=2e-5)
    assert tensor_state_sha(model)==state_before
    model.train();optimizer.zero_grad(set_to_none=True)
    pred,_,_=forward_batch(model,batch)
    task_grads=torch.autograd.grad((pred.view(-1)-batch[3].view(-1)).square().mean(),[core.own_flow.utility_heads[0][-1].weight,core.own_flow.donor_feedback[0][-1].weight],allow_unused=True)
    if cli.mode=='predicted':assert all(g is not None and g.abs().sum()>0 for g in task_grads)
    if cli.mode=='none':assert all(g is None or g.abs().sum()==0 for g in task_grads)
    if cli.mode=='fixed':assert task_grads[0] is None and task_grads[1] is not None and task_grads[1].abs().sum()>0
    output={'mode':cli.mode,'status':'COUNTERFACTUAL_MECHANISM_CHECK_COMPLETE','label_isolation_verified':True,'initial_zero_target_verified':initial_zero_target,
        'reference_and_utility_gradient_detach_verified':True,'module_training_flags_restored':True,'first_feedback_gradient':first_feedback_gradient,
        'nonzero_reference_targets_after_updates_verified':seen_nonzero,'nonzero_utility_gradient_after_updates_verified':seen_utility_gradient,
        'optimizer_updates':20,'shared_fixed_phase_verified':True,'post_phase_mode_gradient_verified':True,'batch_invariance_verified':True,'mask_invariance_verified':True,
        'initial_model_sha256':initial_state_sha,'initial_flow_sha256':initial_flow_sha,'after_shared_updates_model_sha256':after_updates,'batch_order_sha256':digest_bytes(orders.tobytes()),
        'steady_update_seconds_mean':float(np.mean(timings[5:])),'estimated_4000_update_seconds':float(np.mean(timings[5:])*4000),
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'reference_paths_train':10,'reference_paths_eval':1,'test_accessed':False,'performance_experiment_started':False}
    helpers.write(cli.out/'checks.json',output)
    print('COUNTERFACTUAL_MECHANISM_CHECK_COMPLETE',json.dumps(output),flush=True)

if __name__=='__main__':main()
