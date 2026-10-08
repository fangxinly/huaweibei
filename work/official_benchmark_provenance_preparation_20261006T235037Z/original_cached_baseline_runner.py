"""Run the pinned author CaReFlow implementation with validation-only selection.

Model, tokenization, normalization, optimizer and scheduler are unchanged.
The release evaluates test every epoch; this wrapper tests the selected model
once after all training epochs instead. Check mode verifies the pretrained
encoder, gradient flow, one optimizer step and original-size validation batch.
"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time

import numpy as np
import torch
import transformers

DATA_SHA = "5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b"
COMMIT = "5c9f9c7a0bb3f1202ebb3258052da2b99710c565"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def load_author(cli):
    original = sys.argv
    arguments = ["train_reflow_new.py", "--model", str(cli.backbone), "--dataset", "mosi",
                 "--n_epochs", str(cli.epochs), "--train_batch_size", "32", "--learning_rate", "1e-5",
                 "--ratio", "4", "--dropout_prob", "0.5", "--inter_dim", "150", "--share_dim", "100",
                 "--transformer_layer", "3", "--step_size", "2", "--loss_b_ratio", "0.1",
                 "--loss_f_ratio", "0.2", "--eps", "1e-3", "--seed", str(cli.seed)]
    sys.argv = arguments
    sys.path.insert(0, str(cli.repo))
    spec = importlib.util.spec_from_file_location("careflow_author_train", cli.repo / "train_reflow_new.py")
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    sys.argv = original
    original_tqdm = author.tqdm
    author.tqdm = lambda *a, **kw: original_tqdm(*a, disable=True, **kw)
    return author, arguments[1:]


def dataset(author, path, split):
    # Read only requested split; author conversion and TensorDataset stay intact.
    with path.open("rb") as stream:
        examples = pickle.load(stream)[split]
    return author.get_appropriate_dataset(examples)


def pretrained_check(model, backbone):
    state = torch.load(backbone / "pytorch_model.bin", map_location="cpu")
    nested = model.dberta.model.state_dict()
    matched = 0
    for name, tensor in nested.items():
        key = name if name in state else "deberta." + name
        if key in state:
            expected = state[key].to(dtype=tensor.dtype)
            if not torch.equal(tensor.detach().cpu(), expected):
                raise RuntimeError(f"pretrained encoder changed at initialization: {name}")
            matched += 1
    if matched < 190:
        raise RuntimeError(f"too few matched encoder tensors: {matched}")
    return matched


def collect(author, model, loader):
    model.eval()
    predictions, labels = [], []
    batch_mse = []
    with torch.no_grad():
        for batch in loader:
            batch = tuple(x.to(author.DEVICE) for x in batch)
            logits, _, _ = author._forward_eval(model, batch)
            p, y = logits.view(-1), batch[3].view(-1)
            predictions.append(p.cpu().numpy())
            labels.append(y.cpu().numpy())
            batch_mse.append(float((p - y).square().mean()))
    p, y = np.concatenate(predictions), np.concatenate(labels)
    nonzero = y != 0
    result = {"MAE": float(np.abs(p - y).mean()), "Corr": float(np.corrcoef(p, y)[0, 1]),
              "Has0_acc2": float(((p >= 0) == (y >= 0)).mean()),
              "Non0_acc2": float(((p[nonzero] >= 0) == (y[nonzero] >= 0)).mean()),
              "Non0_MAE": float(np.abs(p[nonzero] - y[nonzero]).mean()),
              "Non0_F1": float(author.f1_score(y[nonzero] >= 0, p[nonzero] >= 0, average="weighted")),
              "Acc7": float(author.multiclass_acc(np.clip(p, -3, 3), np.clip(y, -3, 3))),
              "author_batch_mse": float(np.mean(batch_mse)), "samples": len(y),
              "nonzero_samples": int(nonzero.sum())}
    return result, p, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("check", "run"), required=True)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backbone", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=128)
    cli = parser.parse_args()
    cli.repo, cli.backbone, cli.data, cli.out = [p.resolve() for p in (cli.repo, cli.backbone, cli.data, cli.out)]
    cli.out.mkdir(parents=True, exist_ok=True)
    assert digest(cli.data) == DATA_SHA
    commit = subprocess.check_output(["git", "-C", str(cli.repo), "rev-parse", "HEAD"]).decode().strip()
    assert commit == COMMIT
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(2)
    author, arguments = load_author(cli)
    author.set_random_seed(cli.seed)
    train_data = dataset(author, cli.data, "train")
    dev_data = dataset(author, cli.data, "dev")
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True, drop_last=True)
    dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=128, shuffle=False)
    steps = int(len(train_data) / 32) * cli.epochs
    model, optimizer, scheduler = author.prep_for_training(steps)
    matched = pretrained_check(model, cli.backbone)
    files = subprocess.check_output(["git", "-C", str(cli.repo), "ls-files"]).decode().splitlines()
    protocol = {"repository": "https://github.com/TmacMai/CaReFlow", "commit": commit,
                "source_sha256": {n: digest(cli.repo / n) for n in files},
                "runner_sha256": digest(__file__), "data_sha256": DATA_SHA,
                "backbone_sha256": {p.name: digest(p) for p in cli.backbone.iterdir() if p.is_file()},
                "arguments": arguments, "train_samples": len(train_data), "valid_samples": len(dev_data),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                "selection": "author mean of validation batch MSE, batch size 128",
                "deviations": ["test once after validation-only checkpoint selection", "local pretrained path; verified weights"],
                "pretrained_tensors_verified": matched}
    write(cli.out / "protocol.json", protocol)
    if cli.stage == "check":
        b = tuple(t.to(author.DEVICE) for t in next(iter(train_loader)))
        model.train()
        visual, acoustic = author.batch_minmax(b[1]), author.batch_minmax(b[2])
        logits, lf, lb = model(b[0], visual, acoustic, b[3], input_mask=b[4])
        main_loss = (logits.view(-1) - b[3].view(-1)).square().mean()
        loss = main_loss + 0.2 * lf + 0.1 * lb
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite loss")
        loss.backward()
        gradients = {"language": model.dberta.model.embeddings.word_embeddings.weight.grad,
                     "source": model.dberta.proj_a.weight.grad,
                     "forward_flow": model.dberta.reflow_a.prediction_v[0].weight.grad,
                     "backward_flow": model.dberta.reflow_a_b.prediction_v[0].weight.grad}
        for name, grad in gradients.items():
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics, _, _ = collect(author, model, dev_loader)
        checks = {"pretrained_tensors": matched, "finite_nonzero_gradients": list(gradients),
                  "loss": float(loss), "main_loss": float(main_loss), "forward_loss": float(lf),
                  "backward_loss": float(lb), "validation": metrics,
                  "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        write(cli.out / "checks.json", checks)
        print("CHECK_COMPLETE", json.dumps(checks), flush=True)
        return
    history, best, best_epoch = [], float("inf"), 0
    started = time.monotonic()
    for epoch in range(1, cli.epochs + 1):
        loss, forward, backward = author.train_epoch(model, train_loader, optimizer, scheduler)
        valid = author.eval_epoch(model, dev_loader)
        if not np.isfinite(loss + valid):
            raise RuntimeError("nonfinite training/validation loss")
        if valid < best:
            best, best_epoch = valid, epoch
            torch.save(model.state_dict(), cli.out / "best.pt")
        row = {"epoch": epoch, "train_loss": float(loss), "valid_mse": float(valid),
               "forward_loss": float(np.mean(forward)), "backward_loss": float(np.mean(backward)),
               "best_epoch": best_epoch, "best_valid_mse": float(best),
               "elapsed_seconds": round(time.monotonic() - started, 1),
               "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
        history.append(row)
        write(cli.out / "history.json", history)
        print(json.dumps(row), flush=True)
    selection = {"best_epoch": best_epoch, "valid_mse": best, "checkpoint_sha256": digest(cli.out / "best.pt"),
                 "protocol_sha256": digest(cli.out / "protocol.json"), "epochs": cli.epochs}
    write(cli.out / "selection.json", selection)
    # Hashes are checked again before reading the test split.
    assert digest(cli.data) == DATA_SHA
    assert {n: digest(cli.repo / n) for n in files} == protocol["source_sha256"]
    assert digest(cli.out / "best.pt") == selection["checkpoint_sha256"]
    model.load_state_dict(torch.load(cli.out / "best.pt", map_location="cpu"))
    valid_metrics, vp, vy = collect(author, model, dev_loader)
    test_data = dataset(author, cli.data, "test")
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=128, shuffle=False)
    metrics, p, y = collect(author, model, test_loader)
    np.savez_compressed(cli.out / "predictions.npz", test_pred=p, test_y=y, valid_pred=vp, valid_y=vy)
    write(cli.out / "results.json", {"selection": selection, "valid": valid_metrics, "test": metrics})
    print("RUN_COMPLETE", json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
