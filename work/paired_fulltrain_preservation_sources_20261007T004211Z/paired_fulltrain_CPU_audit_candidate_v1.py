"""Original whole checkpoints/Adam/RNG/order/DEV arrays on a different node.

Torch deserialization and tensor arithmetic only. No model constructor or forward.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import sys
from paired_fulltrain_evidence_candidate_v1 import require, now, sha, read, write, plan_gate, stage_association, zip_audit

def state_sha(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        x = value.detach().cpu().contiguous()
        h.update(name.encode())
        h.update(str((tuple(x.shape), str(x.dtype))).encode())
        h.update(x.numpy().tobytes())
    return h.hexdigest()

def rng_sha(value):
    require(set(value) == {'python', 'numpy', 'torch', 'cuda'}, 'Incomplete original RNGs')
    return hashlib.sha256(pickle.dumps({'python': value['python'], 'numpy': value['numpy'],
        'torch': value['torch'].tolist(), 'cuda': [x.tolist() for x in value['cuda']]}, protocol=4)).hexdigest()

def verify_parameters(torch, checkpoint, model, steps, gradient_records=None):
    meta = checkpoint['metadata']
    info = meta['parameter_info']
    names = meta['optimizer_index_to_name']
    require(len(names) == len(set(names.values())) and set(names.values()) == set(info), 'Optimizer parameter identity coverage')
    for n, specification in info.items():
        t = model[n]
        require(list(t.shape) == specification['shape'] and str(t.dtype) == specification['dtype'] and
                t.numel() == specification['numel'], 'Original parameter shape/dtype mismatch: ' + n)
    for n, t in model.items():
        require(torch.is_tensor(t) and (not t.is_floating_point() or bool(torch.isfinite(t).all())), 'Invalid/nonfinite original state: ' + n)
    optimizer = checkpoint['optimizer']
    indices = [i for g in optimizer['param_groups'] for i in g['params']]
    require(len(indices) == len(set(indices)) == len(names) and set(map(str, indices)) == set(names), 'Adam group identity mismatch')
    missing = set(meta.get('missing_gradient_names', []))
    actual_missing = set()
    for index in indices:
        n = names[str(index)]
        s = optimizer['state'].get(index)
        expected_parameter_steps=steps if gradient_records is None else sum(n not in row['gradient_missing_tensors'] for row in gradient_records)
        if s is None:
            require(steps==0 or expected_parameter_steps==0, 'Adam missing despite recorded actual gradient: '+n)
            actual_missing.add(n)
            continue
        require(steps > 0 and set(s) == {'step', 'exp_avg', 'exp_avg_sq'}, 'Adam moment schema mismatch')
        require(float(s['step']) == expected_parameter_steps, 'Original Adam step mismatch: ' + n)
        for k in ('exp_avg', 'exp_avg_sq'):
            t = s[k]
            require(list(t.shape) == info[n]['shape'] and str(t.dtype) == info[n]['dtype'] and
                    bool(torch.isfinite(t).all()), 'Original complete Adam moment mismatch: ' + n)
        require(bool((s['exp_avg_sq'] >= 0).all()), 'Negative Adam second moment')
    if steps == 0:
        require(not optimizer['state'], 'Original clean Adam not empty')
    else:
        require(actual_missing <= missing, 'Adam absent for a parameter with an actual gradient')
        if meta['method'] == 'minimal_fixed_F':
            require(not actual_missing and not missing, 'Fixed retained optimizer/gradient coverage incomplete')
    scheduler = checkpoint['scheduler']
    require(scheduler['last_epoch'] == steps and len(scheduler['_last_lr']) == len(optimizer['param_groups']), 'Original scheduler step/groups mismatch')
    expected_lr = 1e-5 * (steps / 400 if steps < 400 else max(0, (4000 - steps) / 3600))
    require(all(abs(float(x) - expected_lr) <= 1e-16 for x in scheduler['_last_lr']), 'Original scheduler actual LR mismatch')
    require(all(abs(float(g['lr']) - expected_lr) <= 1e-16 for g in optimizer['param_groups']), 'Original Adam/scheduler LR mismatch')
    rng_digest = rng_sha(checkpoint['rng'])
    require(checkpoint['rng']['torch'].dtype == torch.uint8 and checkpoint['rng']['torch'].ndim == 1 and
            len(checkpoint['rng']['cuda']) == 1 and all(x.dtype == torch.uint8 and x.ndim == 1 for x in checkpoint['rng']['cuda']), 'Original Torch/CUDA RNG schema')
    require(meta.get('rng_sha256', meta.get('initial_rng_sha256')) == rng_digest, 'Original complete RNG SHA mismatch')
    return {'parameters': len(info), 'model_states': len(model), 'Adam_states': len(optimizer['state']),
            'Adam_missing_declared': sorted(actual_missing), 'steps': steps, 'rng_sha256': rng_digest, 'state_sha256': state_sha(model)}

def dev_batch_metric(np, prediction, y):
    require(prediction.shape == y.shape == (229,) and np.isfinite(prediction).all() and np.isfinite(y).all(), 'DEV original finite shapes')
    errors = prediction.astype(np.float64) - y.astype(np.float64)
    return float((np.mean(errors[:128] ** 2) + np.mean(errors[128:] ** 2)) / 2)

def verify_training_arrays(np, original, plan, receipt, resume, selected):
    out = Path(original) / 'out'
    ids = read(out / 'actual_official_row_identity.json')
    require(ids['sha256'] == receipt['metadata']['official_row_ID_identity_sha256'] == plan['official_train_dev_ID_identity_sha256'], 'Original official identity binding')
    require(hashlib.sha256(json.dumps(ids['ids'], sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest() == ids['sha256'], 'Original full official ID SHA')
    order = np.load(out / 'original_shared_TRAIN_orders.npy', allow_pickle=False)
    require(sha(out / 'original_shared_TRAIN_orders.npy') == plan['orders_sha256'] and order.shape == (100, 1281), 'Original physical shared order')
    lines = [json.loads(x) for x in (out / 'actual_TRAIN_steps.jsonl').read_text().splitlines()]
    require(len(lines) == 4000, 'Original 4000 updates absent')
    for index, row in enumerate(lines):
        epoch, step = divmod(index, 40)
        require(row['epoch'] == epoch + 1 and row['step_in_epoch'] == step + 1 and row['optimizer_steps'] == index + 1 and
                row['batch_size'] == 32 and row['rows'] == order[epoch, step * 32:(step + 1) * 32].tolist(), 'Original actual optimizer row/order mismatch')
        lr = 1e-5 * ((index + 1) / 400 if index + 1 < 400 else max(0, (4000 - index - 1) / 3600))
        require(all(abs(x - lr) <= 1e-16 for x in row['learning_rates']), 'Original update schedule mismatch')
    target = np.load(out / 'original_DEV_selection_targets.npy', allow_pickle=False)
    history = read(out / 'history.json')
    require(len(history) == 100 and history == resume['history'], 'Original full 100 history differs from resume')
    best_metric, best_epoch = float('inf'), None
    best_prediction = None
    max_error = 0.
    for epoch, row in enumerate(history, 1):
        file = out / ('DEV_epoch_%03d_prediction_only.npz' % epoch)
        require(sha(file) == row['DEV_prediction_file_SHA'], 'Original epoch frozen prediction SHA')
        with np.load(file, allow_pickle=False) as z:
            require(set(z.files) == {'row_ids', 'prediction', 'model_state_sha256'} and z['prediction'].dtype == np.float32 and
                    z['row_ids'].tolist() == ids['ids']['dev'] and str(z['model_state_sha256'].item()) == row['state_SHA'], 'Original epoch prediction ID/state/schema')
            prediction = z['prediction'].copy()
        metric = dev_batch_metric(np, prediction, target)
        max_error = max(max_error, abs(metric - row['DEV_author_batch_MSE_selection_only']))
        require(abs(metric - row['DEV_author_batch_MSE_selection_only']) <= 1e-12 and
                row['epoch'] == epoch and row['optimizer_steps'] == epoch * 40 and row['dropped_TRAIN_rows'] == order[epoch - 1, 1280:].tolist(), 'Original metric/order/update/omitted tail mismatch')
        if metric < best_metric:
            best_metric, best_epoch, best_prediction = metric, epoch, prediction
        require(row['best_epoch'] == best_epoch and abs(row['best_metric'] - best_metric) <= 1e-12, 'Original cumulative earliest strict best mismatch')
    require(best_epoch == receipt['metadata']['best_epoch'] and abs(best_metric - receipt['metadata']['best_metric']) <= 1e-12, 'Original unique selected best mismatch')
    require(np.array_equal(best_prediction, resume['best_prediction']) and np.array_equal(best_prediction, selected['DEV_selected_prediction']), 'Complete selected prediction association')
    with np.load(out / 'selected_best_DEV_replay.npz', allow_pickle=False) as z:
        require(z['row_ids'].tolist() == ids['ids']['dev'] and str(z['model_state_sha256'].item()) == receipt['metadata']['best_state_SHA'] and
                np.array_equal(z['prediction'], best_prediction), 'Original selected whole-file DEV replay array mismatch')
    journal = receipt['guard_journal']
    selection = [r for r in journal if r.get('labels_read') and r.get('role') == 'dev']
    require(len(selection) == 100 and all(x['purpose'] == 'checkpoint_selection_only' and x['rows'] == 229 for x in selection), 'Original DEV label role count')
    require([x['prediction_sha256'] for x in selection] == [r['DEV_prediction_file_SHA'] for r in history], 'Frozen prediction before each original DEV label access association')
    require(all(x.get('role') in ('train', 'dev') for x in journal), 'Unapproved original role journal')
    return {'epochs': 100, 'updates': 4000, 'best_epoch': best_epoch, 'best_metric_selection_only': best_metric,
            'independent_DEV_batch_MSE_max_error': max_error, 'no_final_TEST_or_new_five_metric_score': True}

def main():
    p = argparse.ArgumentParser()
    for n in ('root', 'original-root', 'bundle'):
        p.add_argument('--' + n, type=Path, required=True)
    for n in ('plan-sha', 'original-receipt-sha', 'original-exit-sha'):
        p.add_argument('--' + n, required=True)
    p.add_argument('--method', choices=('careflow', 'minimal_fixed_F'), required=True)
    p.add_argument('--stage', choices=('precheck', 'train'), required=True)
    a = p.parse_args()
    plan = plan_gate(a.bundle, a.plan_sha)
    require(sha(a.original_root / 'out/actual_stage_receipt.json') == a.original_receipt_sha and
            sha(a.original_root / 'natural_exit.json') == a.original_exit_sha, 'Exact original evidence SHA not provided')
    receipt, ex = stage_association(a.original_root, a.plan_sha, a.method)
    required_status = {'precheck': 'ACTUAL_PAIRED_METHOD_PRECHECK2_COMPLETE_NOT_TRAINED100',
                      'train': 'ACTUAL_PAIRED_METHOD_FULLTRAIN100_COMPLETE_PRESERVATION_PENDING_NO_FINAL_TEST'}[a.stage]
    require(receipt['status'] == required_status, 'Wrong original completed stage')
    require(a.root.parent == Path('/data/coding') and a.root.name.startswith('paired_fulltrain_cpu_') and
            not (a.root / 'out').exists(), 'Fresh original CPU root required')
    out = a.root / 'out'
    out.mkdir()
    raw = subprocess.run(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    require(raw.strip() == plan['original_CPU_gpu_UUID'] and raw.strip() != plan['assigned_gpu_UUID'][a.method], 'Different original CPU node UUID required')
    compute = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader'], capture_output=True, text=True, check=True).stdout
    require(not compute.strip(), 'Original CPU node GPU compute must be empty')
    write(out / 'actual_CPU_direct_identity.json', {'actual_utc': now(), 'GPU_UUID_raw': raw, 'compute_raw': compute, 'argv': sys.argv})
    require(shutil.disk_usage(a.root).free >= plan['remote_free_floor_bytes'], 'Original CPU preservation space floor')
    keys = ('clean_initial_full', 'precheck_after2_full') if a.stage == 'precheck' else ('complete_resume_full', 'selected_best_full')
    whole = {}
    for key in keys:
        item = receipt[key]
        file = a.original_root / 'out' / Path(item['path']).name
        require(file.stat().st_size == item['bytes'], 'Original whole checkpoint byte size')
        whole[key] = dict(zip_audit(file, item['sha256']), original_saved_copy=str(file), original_remote_path=item['path'])
    # Only after original source/receipt/natural-exit/full-byte gates: load originals.
    import numpy as np
    import torch
    torch.set_num_threads(2)
    checkpoints = {key: torch.load(whole[key]['original_saved_copy'], map_location='cpu') for key in keys}
    checks = {}
    if a.stage == 'precheck':
        clean, after = checkpoints['clean_initial_full'], checkpoints['precheck_after2_full']
        checks['clean'] = verify_parameters(torch, clean, clean['model'], 0)
        checks['after2'] = verify_parameters(torch, after, after['model'], 2,receipt['updates'])
        require(checks['clean']['state_sha256'] == receipt['clean_initial_state_sha256'] and
                checks['clean']['rng_sha256'] == receipt['clean_initial_rng_sha256'] and
                checks['after2']['state_sha256'] == after['metadata']['state_sha256'], 'Original precheck state/RNG identity')
        identity=read(a.original_root/'out/actual_official_row_identity.json')
        require(identity['sha256']==receipt['official_row_ID_identity_sha256']==plan['official_train_dev_ID_identity_sha256'],'Precheck official identity binding')
        require(hashlib.sha256(json.dumps(identity['ids'],sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()==identity['sha256'],'Precheck physical full ID SHA')
        with np.load(a.original_root / 'out/precheck_DEV_dummy_full_replay.npz', allow_pickle=False) as z:
            require(z['row_ids'].tolist()==identity['ids']['dev'] and z['prediction_dummy0'].dtype==np.float32 and z['prediction_dummy0'].shape == (229,) and np.isfinite(z['prediction_dummy0']).all() and
                    np.array_equal(z['prediction_dummy0'], z['prediction_dummy7']) and
                    np.array_equal(z['prediction_dummy0'], z['prediction_disk_replay']) and
                    str(z['model_state_sha256'].item()) == checks['after2']['state_sha256'], 'Original precheck dummy/whole replay arrays')
        require(receipt['DEV_true_labels_read'] is False and not any(x.get('role') == 'dev' and x.get('labels_read') for x in receipt['guard_journal']), 'Precheck must have no DEV labels')
    else:
        resume, selected = checkpoints['complete_resume_full'], checkpoints['selected_best_full']
        original_steps=[json.loads(x) for x in (a.original_root/'out/actual_TRAIN_steps.jsonl').read_text().splitlines()]
        require(len(original_steps)==4000,'Original 4000 gradient-step records absent')
        checks['resume'] = verify_parameters(torch, resume, resume['model'], 4000,original_steps)
        require(resume['metadata'] == selected['metadata'] == receipt['metadata'], 'Original full checkpoint metadata association')
        require(checks['resume']['state_sha256'] == receipt['metadata']['latest_state_SHA'] and
                state_sha(resume['best_state']) == state_sha(selected['model']) == receipt['metadata']['best_state_SHA'], 'Original latest/selected entire tensor state mismatch')
        require(set(resume['best_state']) == set(selected['model']) and
                all(torch.equal(v, selected['model'][n]) for n, v in resume['best_state'].items()), 'Original complete selected tensor equality')
        checks['arrays_history_orders'] = verify_training_arrays(np, a.original_root, plan, receipt, resume, selected)
    result = {'status': 'ACTUAL_PAIRED_ORIGINAL_WHOLE_' + a.stage.upper() + '_CPU_AUDIT_PASSED_NO_MODEL_FORWARD',
              'actual_utc': now(), 'pid': os.getpid(), 'argv': sys.argv, 'root': str(a.root), 'source_bundle': str(a.bundle),
              'method': a.method, 'stage': a.stage, 'plan_sha256': a.plan_sha, 'original_stage_receipt_sha256': a.original_receipt_sha,
              'original_stage_exit_sha256': a.original_exit_sha, 'original_root': receipt['root'], 'whole_checkpoint_records': whole,
              'checks': checks, 'CPU_model_forward': False, 'final_TEST_access': False, 'original_CPU_GPU_UUID': raw.strip()}
    write(out / 'actual_stage_receipt.json', result)
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()
