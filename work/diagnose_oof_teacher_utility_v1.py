"""New OOF-scalar-driven finite path on immutable C2. No fitting or label solver."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
from pathlib import Path
import argparse, datetime, hashlib, importlib.util, json, shutil, subprocess, sys, time
import numpy as np
import torch


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--phase', choices=['precheck', 'execute'], required=True)
    args = parser.parse_args()
    start = time.monotonic()
    plan = json.loads(args.plan.read_text())
    assert plan['status'] == 'FROZEN_NEW_OOF_TEACHER_UTILITY_DIAGNOSTIC'
    assert sha(__file__) == plan['source_sha256']
    for path, expected in plan['pinned_files'].items():
        assert sha(path) == expected, path
    root = Path(plan['new_root'])
    out = root / args.phase
    assert not out.exists()
    out.mkdir(parents=True)
    query = lambda argv: subprocess.check_output(argv, text=True).strip()
    gpu = query(['nvidia-smi', '--query-gpu=uuid,name,memory.total,memory.used', '--format=csv,noheader,nounits'])
    compute = query(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,used_memory', '--format=csv,noheader,nounits'])
    assert gpu.split(',')[0] == plan['expected_uuid'] and not compute
    now = datetime.datetime.now(datetime.timezone.utc)
    remaining = (datetime.datetime.fromisoformat(plan['estimated_lease_end_utc']) - now).total_seconds()
    assert remaining > plan['maximum_seconds'] + plan['preservation_reserve_seconds']
    assert shutil.disk_usage(root).free >= plan['minimum_remote_free_bytes']
    save_json(out / 'inventory.json', dict(utc=now.isoformat(), gpu=gpu, compute=compute,
              argv=sys.argv, pid=os.getpid(), free_bytes=shutil.disk_usage(root).free,
              estimated_lease_remaining_seconds=remaining, platform_deadline_verified=False))
    if args.phase == 'execute':
        gate = json.loads((root / 'precheck/receipt.json').read_text())
        authorization = json.loads((root / 'execution_authorization.json').read_text())
        assert gate['passed'] and authorization['precheck_receipt_sha256'] == sha(root / 'precheck/receipt.json')
        assert authorization['plan_sha256'] == sha(args.plan)
        assert authorization['source_sha256'] == sha(__file__)
        assert authorization['execute_authorized_after_actual_precheck'] is True
    spec = importlib.util.spec_from_file_location('immutable_original_oracle_helpers', plan['helper_source'])
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)  # __main__ never runs: no old GPU diagnostic repeated.
    base = Path(plan['base_root']); formal = Path(plan['formal_root'])
    sys.path[:0] = [str(formal), str(base)]
    from finite_task_risk_runtime_v2 import FrozenCoordinateLearner, PAIRS
    torch.set_num_threads(2); torch.manual_seed(91817)
    torch.use_deterministic_algorithms(True)
    selection = json.loads((formal / 'run/selection.json').read_text())
    assert selection['epochs'] == 100 and selection['best_epoch'] == 37 and selection['effective_mode'] == 'finite_vector'
    scales = json.loads((formal / 'train_scales.json').read_text())
    assert scales['beta'] == plan['beta']
    model = FrozenCoordinateLearner(base, 'finite_vector', scales).cuda()
    model.restore_addon(torch.load(formal / 'run/best_addon.pt', map_location='cpu'))
    model.eval(); model.requires_grad_(False)
    before = helpers.tensor_sha(model.state_dict())
    assert not any(p.requires_grad for p in model.parameters())
    fb = model.feedback
    assert fb.inner_steps == 3 and fb.step_size == .25 and fb.trust_fraction == .25
    cache_path = base / 'teacher_cache_v1/train_cache.npz'
    cache = np.load(cache_path, allow_pickle=False)
    input_names = ['state', 'mask', 'old_context', 'pooled_state', 'reference_prediction']
    inputs = {name: cache[name] for name in input_names}  # Never read cache y here.
    scalar = np.load(root / 'mu_input.npz', allow_pickle=False)
    assert set(scalar.files) == {'row_ids', 'fold', 'mu'}
    assert np.array_equal(scalar['row_ids'], np.arange(1281))
    mu = scalar['mu']; assert mu.shape == (1281,) and np.isfinite(mu).all()
    mapping = json.loads(Path(plan['mapping_path']).read_text())
    assert [r['row'] for r in mapping] == list(range(1281))
    videos = np.asarray([r['video_id'] for r in mapping]); assert len(np.unique(videos)) == 52
    with np.load(plan['old_diagnostic_arrays'], allow_pickle=False) as old:
        # Only immutable prediction controls loaded. Old labels/oracle branches not consumed.
        old_raw = old['raw_prediction'].copy()
        old_learned = old['original_prediction'].copy()
        old_p0 = old['p0'].copy()
    assert all(len(v) == 1281 for v in inputs.values())
    torch.cuda.reset_peak_memory_stats()

    def batch_for(rows):
        return {k: torch.as_tensor(v[rows], device='cuda') for k, v in inputs.items()}

    def geometry(batch):
        pool = batch['pooled_state'].detach()
        messages = torch.stack([model.donor[i](torch.cat([
            pool[:,r], pool[:,d], .5*(pool[:,r]-pool[:,d]), .5*(pool[:,r]+pool[:,d])], -1))
            for i, (r,d) in enumerate(PAIRS)], 1).detach()
        terminal = lambda cx: model.terminal(batch['state'].detach(), batch['mask'], cx)
        return messages, terminal

    def new_path(batch, target):
        messages, terminal = geometry(batch)
        p0 = batch['reference_prediction'].detach()
        rho = (p0 - target).detach()
        path = helpers.solve(fb, batch['old_context'], messages, p0, rho, terminal, y=None)
        # Pure target prediction risk, not the proximity regularizer, selects candidates.
        # float64 on unchanged model predictions; torch.argmin gives earliest exact tie.
        risk = (path['prediction_path'].double() - target.double()[:,None]).square()
        index = risk.argmin(1)
        path['mu_risk_path'] = risk
        path['selected_step'] = index
        path['selected_prediction'] = path['prediction_path'].gather(1,index[:,None])[:,0]
        return path

    if args.phase == 'precheck':
        evidence = []; longest_batch_seconds = 0
        # Full ordinary batch plus exact previously failing one-token TRAIN witness.
        for rows in [np.arange(32), np.asarray([620,621])]:
            began = time.monotonic(); b = batch_for(rows)
            target = torch.as_tensor(mu[rows], dtype=torch.float32, device='cuda')
            path = new_path(b, target)
            with_labels = dict(b, y=torch.arange(len(rows), device='cuda').float()+20)
            changed = new_path(with_labels, target)
            label_error = max(float((path[k]-changed[k]).abs().max()) for k in path)
            assert label_error == 0
            original = model(b, 'finite_vector')[0].detach()
            original_changed = model(with_labels, 'finite_vector')[0].detach()
            assert float((original-original_changed).abs().max()) == 0
            old_error = float((original.cpu()-torch.as_tensor(old_learned[rows])).abs().max())
            assert old_error <= 1e-6
            raw_error = float(np.abs(path['prediction_path'][:,0].cpu().numpy()-old_raw[rows]).max())
            assert raw_error <= 1e-6
            messages, terminal = geometry(b)
            equiv = helpers.check_equivalence(fb,b['old_context'],messages,b['reference_prediction'],target,terminal)
            lengths = b['mask'].sum(-1).cpu().tolist()
            if 620 in rows: assert lengths[list(rows).index(620)] == 1
            seconds = time.monotonic()-began
            longest_batch_seconds = max(longest_batch_seconds,seconds)
            evidence.append(dict(rows=rows.tolist(), valid_lengths=lengths, label_replacement_max_error=label_error,
                                 original_learned_replay_max_error=old_error, original_F_replay_max_error=raw_error,
                                 oof_mu_input_gradient_hvp=equiv, seconds=seconds))
        peak = torch.cuda.max_memory_allocated()
        assert peak <= plan['maximum_peak_allocated_bytes']
        assert longest_batch_seconds * 41 * 2 < plan['maximum_seconds']
        after = helpers.tensor_sha(model.state_dict())
        assert before == after and all(p.grad is None for p in model.parameters())
        save_json(out / 'receipt.json', dict(passed=True, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_sha256=sha(__file__), plan_sha256=sha(args.plan), pinned_files=plan['pinned_files'],
            model_state_before=before, model_state_after=after, no_parameter_grads=True, optimizer_steps=0,
            actual_peak_allocated_bytes=peak, seconds=time.monotonic()-start,
            pessimistic_41_batch_seconds=longest_batch_seconds*41*2, evidence=evidence,
            original_input_replay_scope='Fresh unchanged coordinate replay against immutable original Oracle controls; full original-input C2 disk replay is separately pinned prior evidence, not rerun here.'))
        print('OOF_UTILITY_PRECHECK_COMPLETE',peak,flush=True); return

    arrays = {}; raw_error = p0_error = 0.0
    for first in range(0,1281,32):
        rows = np.arange(first,min(1281,first+32)); b=batch_for(rows)
        target=torch.as_tensor(mu[rows],dtype=torch.float32,device='cuda')
        path = new_path(b,target)
        with torch.no_grad():
            p0replay = model.terminal(b['state'],b['mask'],.5*b['old_context'])
        p0_error=max(p0_error,float((p0replay-b['reference_prediction']).abs().max()))
        raw_error=max(raw_error,float(np.abs(path['prediction_path'][:,0].cpu().numpy()-old_raw[rows]).max()))
        for k,v in path.items(): arrays.setdefault(k,[]).append(v.detach().cpu().numpy())
        assert torch.cuda.max_memory_allocated() <= plan['maximum_peak_allocated_bytes']
        assert time.monotonic()-start <= plan['maximum_seconds']
        save_json(out/'progress.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=int(rows[-1])+1,pid=os.getpid()))
    merged={k:np.concatenate(v) for k,v in arrays.items()}
    assert raw_error<=1e-6 and p0_error<=1e-6
    assert np.array_equal(old_p0,inputs['reference_prediction'])
    after=helpers.tensor_sha(model.state_dict()); assert before==after
    assert all(p.grad is None for p in model.parameters())
    # Freeze all predictions before any real outcome labels are accessed.
    frozen=out/'predictions_frozen.npz'
    np.savez_compressed(frozen,row=np.arange(1281),video=videos,fold=scalar['fold'],
                        mu=mu,p0=inputs['reference_prediction'],**merged)
    prediction_sha=sha(frozen)
    save_json(out/'prediction_freeze.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        sha256=prediction_sha,bytes=frozen.stat().st_size,real_labels_read=False,
        solver_target='OOF scalar only',selector='earliest minimum OOF-mu squared risk, including original F'))
    labels=cache['y']  # Metrics only, after immutable prediction arrays are written and hashed.
    np.savez_compressed(out/'metric_labels.npz',row=np.arange(1281),y=labels)
    raw=merged['prediction_path'][:,0].astype('float64')
    metrics={}
    for name,pred in [('raw',raw),('oof_last',merged['prediction_path'][:,-1]),
                      ('oof_mu_best',merged['selected_prediction']),('old_learned',old_learned)]:
        pred=pred.astype('float64'); error=pred-labels.astype('float64')
        q=error**2-(raw-labels.astype('float64'))**2
        changed=np.abs(pred-raw)>1e-8
        metrics[name]=dict(mse=float(np.mean(error**2)),mae=float(np.mean(np.abs(error))),
            video_equal_mse=float(np.mean([np.mean(error[videos==v]**2) for v in np.unique(videos)])),
            changed_rows=int(changed.sum()), harmful_fraction_changed=float(np.mean(q[changed]>1e-12)) if changed.any() else None,
            improved_fraction_all=float(np.mean(q < -1e-12)),mean_risk_change=float(q.mean()))
    assert sha(frozen)==prediction_sha
    save_json(out/'receipt.json',dict(passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sha(__file__),plan_sha256=sha(args.plan),rows=1281,videos=52,batches=41,
        prediction_sha256=prediction_sha,labels_sha256=sha(out/'metric_labels.npz'),
        original_F_max_replay_error=raw_error,p0_max_replay_error=p0_error,
        model_state_before=before,model_state_after=after,no_parameter_grads=True,optimizer_steps=0,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),seconds=time.monotonic()-start,
        selected_step_counts=np.bincount(merged['selected_step'],minlength=4).tolist(),metrics=metrics,
        label_scope='Real TRAIN y read only after predictions file/hash freeze; no DEV/TEST read.',
        limits=plan['limits']))
    print('OOF_UTILITY_DIAGNOSTIC_COMPLETE',json.dumps(metrics),flush=True)


if __name__=='__main__': main()
