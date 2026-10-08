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
    parser.add_argument("--stage", choices=("run",), required=True)
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
    own_files = [Path(__file__).with_name(name) for name in ("run_counterfactual_v5.py", "counterfactual_flow_model.py", "legacy_flow_model.py", "encoder_adapter.py")]
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
    protocol = {"name": "inflow_counterfactual_v5", "mode": cli.mode, "config": CONFIG,
                "initial_model_sha256": initial_state_sha, "initial_flow_sha256": initial_flow_sha,
                "batch_order_sha256": digest_bytes(orders.tobytes()),
                "batch_order_epoch_sha256": [digest_bytes(row.tobytes()) for row in orders],
                "updates_per_epoch": 40, "total_updates": 4000,
                "gpu_uuid": identity,
                "placeholder_buffers": "Two positional embedding _float_tensor scalars zeroed; only dtype used by type_as",
                "parent_pythonhashseed": os.environ.get("PYTHONHASHSEED"),
                "condition": "After first Euler stage: none / fixed .5 / predicted sigmoid(4u), six directed detached-state donor feedback channels; zero feedback output init",
                "utility_config": "TRAIN final-task marginal g=(p_without_i-y)^2-(p_ref-y)^2 with fixed reference .5; tanh(g/.05), detached target/input, balanced SmoothL1 .01; first10epochs shared fixed feedback; 100 matched epochs; 10 deterministic references training/1 inference; no label in u input",
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
                "test_policy": "TEST forbidden; selected epoch determines effective phase; end dev/default and fixed condition-off only"}
    helpers.write(cli.out / "protocol.json", protocol)
    history, best, best_epoch = [], float("inf"), 0
    started = time.monotonic()
    for epoch in range(1, cli.epochs + 1):
        core.own_flow.set_epoch(epoch)
        losses = train_epoch(author, model, epoch_loader(epoch - 1), optimizer, scheduler)
        if epoch==10:
            helpers.write(cli.out / "shared_phase.json", {"epoch":10,"model_sha256":tensor_state_sha(model),"batch_order_sha256":protocol["batch_order_sha256"]})
        valid = author.eval_epoch(model, valid_loader)
        assert np.isfinite(valid)
        if valid < best:
            best, best_epoch = float(valid), epoch
            torch.save(model.state_dict(), cli.out / "best.pt")
        row = {"epoch": epoch, **losses, "valid_mse": float(valid),
               "best_epoch": best_epoch, "best_valid_mse": best,
               "elapsed_seconds": round(time.monotonic() - started, 1),
               "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
               "last_dev_batch_utility_weights":core.last_trace["utility_weights"].mean(0).cpu().tolist()}
        history.append(row)
        helpers.write(cli.out / "history.json", history)
        print(json.dumps(row), flush=True)
    selection = {"epochs": cli.epochs, "best_epoch": best_epoch, "valid_mse": best,
                 "selected_effective_mode": "fixed" if best_epoch<=10 else cli.mode,
                 "checkpoint_sha256": helpers.digest(cli.out / "best.pt"),
                 "protocol_sha256": helpers.digest(cli.out / "protocol.json")}
    helpers.write(cli.out / "selection.json", selection)
    assert helpers.digest(cli.data) == helpers.DATA_SHA
    for group, paths in (("own_source_sha256", own_files), ("helper_source_sha256", helper_files)):
        assert {p.name: helpers.digest(p) for p in paths} == protocol[group]
    assert {n: helpers.digest(cli.repo / n) for n in author_files} == protocol["author_source_sha256"]
    for name, sha in protocol["backbone_sha256"].items():
        assert helpers.digest(cli.backbone / name) == sha
    assert helpers.digest(cli.out / "best.pt") == selection["checkpoint_sha256"]
    model.load_state_dict(torch.load(cli.out / "best.pt", map_location="cpu"))
    core.own_flow.set_epoch(best_epoch)
    valid_metrics, vp, vy = helpers.collect(author, model, valid_loader)
    before_state = tensor_state_sha(model)
    core.own_flow.context_override = "none"
    off_metrics, offp, offy = helpers.collect(author, model, valid_loader)
    core.own_flow.context_override = None
    assert np.array_equal(vy, offy)
    assert before_state == tensor_state_sha(model)
    assert helpers.digest(cli.out / "best.pt") == selection["checkpoint_sha256"]
    extra={k:[] for k in ('own','pair','utility','predicted_weights','weights','reference_prediction')}
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            batch=tuple(x.to(author.DEVICE) for x in batch)
            neutral=(*batch[:3],torch.zeros_like(batch[3]),batch[4])
            forward_batch(model,neutral)
            for name in extra:extra[name].append(core.own_flow._stage1[name].cpu().numpy())
    extra={k:np.concatenate(v) for k,v in extra.items()}
    np.savez_compressed(cli.out / "predictions.npz", valid_pred=vp, valid_y=vy, condition_off_pred=offp,**extra)
    helpers.write(cli.out / "results.json", {"mode": cli.mode, "selection": selection,
                      "valid": valid_metrics, "frozen_condition_off": off_metrics,
                      "model_state_unchanged": True, "test_accessed": False})
    print("COUNTERFACTUAL_V5_RUN_COMPLETE", json.dumps(valid_metrics), flush=True)



if __name__ == "__main__":
    main()
