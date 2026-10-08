"""New same-T_k FIT/INNER scalar collection. Prepared locally, not GPU executed."""
import os
os.environ['HF_HUB_OFFLINE'] = os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
from types import MethodType, SimpleNamespace
import argparse, datetime, hashlib, json, math, shutil, subprocess, sys, time
import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    temporary = path.with_suffix('.pending.json')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--phase', choices=['precheck', 'execute'], required=True)
    args = parser.parse_args()
    started = time.monotonic()
    now = lambda: datetime.datetime.now(datetime.timezone.utc)
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    assert plan['status'] == 'LOCAL_FROZEN_SAME_TEACHER_COLLECTION_PROTOCOL_NOT_GPU_VALIDATED'
    assert sha(__file__) == plan['collector_sha256']
    root = Path(plan['teacher_root'])
    base = Path(plan['base_root'])
    new_root = root / plan['collection_subdirectory']
    assert new_root.resolve().is_relative_to(root.resolve())
    for relative, expected in plan['new_source_sha256'].items():
        assert sha(new_root / relative) == expected, relative
    for relative, expected in plan['shared_teacher_pins'].items():
        assert sha(root / relative) == expected, relative
    parent = json.loads((root / 'group_teacher_plan_v3.json').read_text(encoding='utf-8'))
    for relative, expected in parent['asset_sha256'].items():
        assert sha(base / relative) == expected, relative
    selected = plan['folds'][args.fold]
    for relative, expected in selected['original_small_pins'].items():
        assert sha(root / relative) == expected, relative
    completed = json.loads((root / 'run_v1/completion.json').read_text(encoding='utf-8'))
    history = json.loads((root / 'run_v1/history.json').read_text(encoding='utf-8'))
    assert completed['fold'] == args.fold and completed['completed_epochs'] == len(history) == 100
    assert completed['best_epoch'] == min(range(1, 101), key=lambda epoch: history[epoch-1]['inner_sample_mse'])
    assert json.loads((root / 'formal_exit.json').read_text())['exit_code'] == 0
    checkpoint = root / 'run_v1/selected_full_checkpoint.pt'
    assert checkpoint.stat().st_size == selected['full_checkpoint_bytes'] == completed['full_checkpoint_bytes']
    assert sha(checkpoint) == selected['full_checkpoint_sha256'] == completed['full_checkpoint_sha256']
    query = lambda command: subprocess.check_output(command, text=True).strip()
    gpu = query(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.used', '--format=csv,noheader,nounits'])
    compute = query(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory', '--format=csv,noheader,nounits'])
    assert gpu.split(',')[0].strip() == selected['expected_uuid'] and not compute
    remaining = (datetime.datetime.fromisoformat(plan['estimated_lease_end_utc']) - now()).total_seconds()
    assert remaining > plan['maximum_seconds'] + plan['preservation_margin_seconds']
    assert shutil.disk_usage(root).free >= plan['minimum_remote_free_bytes']
    space = json.loads((new_root / 'fresh_space_authorization.json').read_text(encoding='utf-8'))
    assert space['status'] == 'FRESH_LOCAL_AND_REMOTE_COLLECTION_SPACE_VERIFIED'
    assert 0 <= (now()-datetime.datetime.fromisoformat(space['utc'])).total_seconds() < 600
    assert space['local_permanent_free_bytes'] >= plan['minimum_local_permanent_free_bytes']
    assert space['expected_uuid'] == selected['expected_uuid'] and space['deletion_required'] is False
    if args.phase == 'execute':
        auth = json.loads((new_root / 'execution_authorization.json').read_text(encoding='utf-8'))
        receipt = new_root / 'precheck/receipt.json'
        assert auth['actual_gpu_precheck_and_independent_audit_passed'] is True
        assert auth['plan_sha256'] == sha(args.plan) and auth['collector_sha256'] == sha(__file__)
        assert auth['original_precheck_receipt_sha256'] == sha(receipt)
        assert json.loads(receipt.read_text())['passed'] is True
        assert auth['capture_source_coverage_verified'] is True
        assert auth['capture_source_sha256'] == plan['capture_source_sha256']
    out = new_root / args.phase
    assert not out.exists()
    out.mkdir(parents=True)
    write(out / 'inventory.json', {'utc': now().isoformat(), 'gpu': gpu, 'compute': compute,
          'argv': sys.argv, 'pid': os.getpid(), 'phase': args.phase,
          'plan_sha256': sha(args.plan), 'free_bytes': shutil.disk_usage(root).free,
          'platform_deadline_verified': False, 'remaining_estimate_seconds': remaining})
    import torch
    sys.path[:0] = [str(root), str(base/'assets'), str(base/'frozen_v5'), str(new_root)]
    from group_teacher_runtime_v2 import plain_masked, forward, tensor_sha
    from label_free_group_teacher_inputs_v1 import LabelFreeTeacherInputs
    from encoder_adapter import fit_statistics, install
    import run_careflow as helpers
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model_args = SimpleNamespace(repo=base/'assets/CaReFlow', backbone=base/'assets/deberta-v3-base',
                                 epochs=100, seed=91818)
    author, author_arguments = helpers.load_author(model_args)
    author.set_random_seed(91818)
    fold_ids = {role: np.load(root/f'{role}_{args.fold}.npy', allow_pickle=False) for role in ('fit', 'inner', 'outer')}
    data = LabelFreeTeacherInputs.from_pickle(base/'assets/mosi.pkl', fold_ids, author)
    fit = data.dataset(fold_ids['fit'], 'fit')
    inner = data.dataset(fold_ids['inner'], 'inner')
    assert torch.count_nonzero(fit.tensors[3]) == torch.count_nonzero(inner.tensors[3]) == 0
    stats = fit_statistics(fit)  # Outcome-free, original FIT features only.
    model, temporary_optimizer, temporary_scheduler = author.prep_for_training(100*math.ceil(len(fit)/32))
    # Original construction creates optimizer objects. They perform zero steps.
    del temporary_optimizer, temporary_scheduler
    core = model.dberta
    for name in ['reflow_a', 'reflow_v', 'reflow_a_b', 'reflow_v_b', 'rf_a', 'rf_v', 'rf_a_b', 'rf_v_b', 'pooler']:
        delattr(core, name)
    install(core, stats)
    for encoder in (core.transa, core.transv):
        encoder.embed_positions._float_tensor.zero_()
    fresh_norm = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()
                  if name.startswith(('dberta.v6_audio_', 'dberta.v6_visual_'))}
    loaded = torch.load(checkpoint, map_location='cpu')
    assert len(loaded) == 301 and all(torch.isfinite(value).all() for value in loaded.values())
    assert all(torch.equal(value, loaded[name]) for name, value in fresh_norm.items())
    model.load_state_dict(loaded, strict=True)
    del loaded
    core.forward = MethodType(plain_masked, core)
    model.to(author.DEVICE).eval().requires_grad_(False)
    before = tensor_sha(model.state_dict())
    assert before == completed['selected_model_tensor_sha256'] == selected['selected_model_tensor_sha256']
    assert not any(parameter.requires_grad or parameter.grad is not None for parameter in model.parameters())
    torch.cuda.reset_peak_memory_stats()
    longest = 0.0

    def collect_inputs(dataset):
        nonlocal longest
        pieces = []
        with torch.no_grad():
            for batch in torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False):
                tick = time.monotonic()
                moved = tuple(value.to(author.DEVICE) for value in batch)
                assert torch.count_nonzero(moved[3]) == 0
                prediction = forward(model, moved)
                assert torch.isfinite(prediction).all()
                pieces.append(prediction.cpu().numpy())
                longest = max(longest, time.monotonic()-tick)
                assert torch.cuda.max_memory_allocated() <= plan['maximum_peak_allocated_bytes']
                assert time.monotonic()-started <= plan['maximum_seconds']
        return np.concatenate(pieces)

    # Full original INNER batch layout: direct disk/input replay, labels replaced by 0.
    replay = collect_inputs(inner)
    with np.load(root/'run_v1/selected_inner_predictions.npz', allow_pickle=False) as stored:
        assert np.array_equal(stored['row_ids'], fold_ids['inner'])
        replay_error = float(np.max(np.abs(replay-stored['prediction'])))
    assert replay_error <= 1e-6
    witness_local = np.arange(min(32, len(fit)))
    if 620 in set(fold_ids['fit'].tolist()):
        witness_local = np.unique(np.append(witness_local, int(np.where(fold_ids['fit'] == 620)[0][0])))
    witness = tuple(value[torch.as_tensor(witness_local)].to(author.DEVICE) for value in fit.tensors)
    with torch.no_grad():
        old_prediction = forward(model, witness)
        changed = list(witness)
        changed[3] = torch.full_like(changed[3], 17.0)
        label_error = float(torch.max(torch.abs(forward(model, tuple(changed))-old_prediction)))
    assert label_error == 0.0
    for role in ('outer', 'fit'):
        try:
            data.dataset(fold_ids['outer'][:1], role)
        except PermissionError:
            continue
        raise AssertionError('OUTER guard failed')
    if args.phase == 'execute':
        fit_mu = collect_inputs(fit)
        for role, value in [('fit', fit_mu), ('inner', replay)]:
            np.savez_compressed(out/f'{role}_scalar_inputs.npz', row_ids=fold_ids[role],
                                fold=np.full(len(value), args.fold, dtype=np.int64), mu=value)
    after = tensor_sha(model.state_dict())
    assert before == after and all(parameter.grad is None for parameter in model.parameters())
    peak = torch.cuda.max_memory_allocated()
    conservative = longest*(math.ceil(len(fit)/128)+math.ceil(len(inner)/128))*2
    assert peak <= plan['maximum_peak_allocated_bytes'] and conservative <= plan['maximum_seconds']
    receipt = {'passed': True, 'utc': now().isoformat(), 'phase': args.phase, 'fold': args.fold,
        'source_sha256': sha(__file__), 'plan_sha256': sha(args.plan), 'checkpoint_sha256': sha(checkpoint),
        'before_tensor_sha256': before, 'after_tensor_sha256': after,
        'inner_strict_original_input_disk_replay_max_error': replay_error, 'label_replacement_max_error': label_error,
        'fit_only_normalization_buffers_exact': True, 'optimizer_objects_constructed_then_discarded': True,
        'optimizer_steps': 0, 'parameter_gradients_present': False, 'data_journal': data.journal,
        'actual_peak_allocated_bytes': peak, 'seconds': time.monotonic()-started,
        'conservative_collection_seconds': conservative, 'dev_requested': False, 'test_requested': False,
        'whole_pipeline_crossfit': False, 'new_outer_inference': False,
        'collection_outputs_written': args.phase == 'execute'}
    write(out/'receipt.json', receipt)
    print('SAME_TEACHER_LABEL_FREE_COLLECTION_'+args.phase.upper()+'_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
