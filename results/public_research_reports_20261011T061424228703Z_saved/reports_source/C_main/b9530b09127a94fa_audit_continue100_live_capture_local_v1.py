"""Audit the original live capsule and resume records; not a model checkpoint."""
import datetime, hashlib, io, json, math, zipfile
from pathlib import Path, PurePosixPath
import numpy as np

ROOT = Path('D:/CodexBackups/selective_flow_20261003_1105')
BASE = ROOT/'staged_reference_continue100_actual_20261006T165611Z'
A = BASE/'a'
PREFIX = ROOT/'staged_reference_first10_actual_20261006T151954Z'
RUN = '/data/coding/minimal_fixed_fold0_continue100_actual_20261006T165611Z'
LOCAL_SOURCE = Path(__file__).parent/'minimal_fixed_staged_reference_v1_20261006T151117Z'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt = read(A/'capture_receipt.json')
exit_record = read(A/'actual_capture_exit.json')
assert exit_record['exit_code'] == 0 and exit_record['natural_wait_verified']
assert exit_record['child_pid'] == receipt['pid']
assert exit_record['child_full_argv'][1:] == receipt['argv']
assert exit_record['capture_source_sha256'] == receipt['capture_source_sha256'] == sha(Path(__file__).with_name('capture_staged_reference_live_v26.py'))
assert sha(A/'snapshot.zip') == receipt['snapshot_sha256']
assert (A/'snapshot.zip').stat().st_size == receipt['snapshot_bytes']
assert not receipt['formal100_complete'] and not receipt['weights_downloaded_this_capture']
raw = A/'original_small_files'
raw.mkdir(exist_ok=True)
with zipfile.ZipFile(A/'snapshot.zip') as z:
    assert z.testzip() is None
    assert len(z.namelist()) == len(set(z.namelist())) == receipt['members']
    mf = json.loads(z.read('member_manifest.json'))
    assert set(z.namelist()) == set(mf['small_members']) | {'member_manifest.json'}
    assert mf['run'] == RUN and mf['resumed_from_epoch'] == 10 and mf['resumed_from_steps'] == 220
    assert mf['new_source_and_run_and_full_argv_covered'] and not mf['formal100_complete']
    assert 'GPU-53696803-875e-eec8-2231-29db63579891' in mf['actual_inventory']['gpu']
    for name, entry in mf['small_members'].items():
        parts = PurePosixPath(name)
        assert not parts.is_absolute() and '..' not in parts.parts
        data = z.read(name)
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        assert len(data) == entry['bytes'] and entry['stable_before_after']
        out = raw/name
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists(): assert out.read_bytes() == data
        else: out.write_bytes(data)
    plan_bytes = z.read('source/staged_reference_plan.json')
    plan = json.loads(plan_bytes)
    assert hashlib.sha256(plan_bytes).hexdigest() == '67cc46eea7d2f7dbf0e1f681a28fbbdbb7281eb34c9b703bddca4bf0de7a3db4'
    for name, digest in {**plan['source_sha256'], **plan['role_order_sha256']}.items():
        assert hashlib.sha256(z.read('source/'+name)).hexdigest() == digest
        assert sha(LOCAL_SOURCE/name) == digest
    gate_name = 'preservation_inputs/stage10_preservation_joint_20261006T165454Z.json'
    assert z.read(gate_name) == (PREFIX/'complete_stage10_GPU_D_B_CPU_joint_audit.json').read_bytes()
    assert hashlib.sha256(z.read(gate_name)).hexdigest() == '6aaa3f96d53584b67a19090bb1b5aa07814433478f06b3f1b9c7c628d26a2bc6'
    start = json.loads(z.read('run/out/actual_training_start.json'))
    launch = json.loads(z.read('run/actual_child_launch.json'))
    progress = json.loads(z.read('run/out/progress.json'))
    assert start['phase'] == 'continue100' and start['start_epoch'] == 10 and start['optimizer_steps'] == 220
    assert start['state_sha256'] == '45aeab53724cb03e4631ffba63fddd587f36065afe58ab9921a0325351a5064a'
    assert start['parameters'] == 185402807 and start['parameter_tensors'] == 364
    assert not start['initial_optimizer_empty'] and start['precheck_optimizer_or_rng_not_inherited']
    assert launch['wrapper_pid'] == 1485 and launch['child_pid'] == start['pid'] == 1486
    assert launch['child_full_argv'][1:] == start['argv']
    assert launch['source_sha256'] == plan['source_sha256']['minimal_fixed_staged_training_v3.py']
    args = launch['child_full_argv']
    assert args[args.index('--resume')+1] == '/data/coding/minimal_fixed_fold0_stage10_actual_20261006T151300Z/out/complete_resume_full.pt'
    prefix = read(PREFIX/'a/original_small_files/run/out/actual_training_receipt.json')
    assert progress['history'][:10] == prefix['history']
    assert progress['phase'] == 'continue100' and not progress['formal100_complete']
    assert progress['epoch'] == len(progress['history']) >= 11
    assert progress['optimizer_steps'] == progress['epoch'] * 22
    orders = np.load(io.BytesIO(z.read('source/fit_orders_seed91819_100.npy')), allow_pickle=False)
    steps = [json.loads(line) for line in z.read('run/out/actual_fit_steps.jsonl').decode().splitlines()]
    for i, entry in enumerate(steps):
        epoch, batch = divmod(i, 22)
        assert entry['epoch'] == epoch+11 and entry['step_in_epoch'] == batch+1 and entry['optimizer_steps'] == i+221
        assert entry['rows'] == orders[epoch+10, batch*32:(batch+1)*32].tolist()
        assert entry['batch_size'] == (23 if batch == 21 else 32)
        assert entry['all_finite_nonNone_tensors'] == 364 and math.isfinite(entry['objective'])
    inner = np.load(io.BytesIO(z.read('source/inner_0.npy')), allow_pickle=False)
    for entry in progress['history'][10:]:
        assert entry['optimizer_steps'] == entry['epoch']*22 and entry['fit_rows'] == 695 and entry['batch_count'] == 22 and entry['tail_rows'] == 23
        assert not entry['OUTER_or_head_labels_read']
        pred_data = z.read('run/out/inner_epoch_%03d.npz'%entry['epoch'])
        assert hashlib.sha256(pred_data).hexdigest() == entry['inner_prediction_file_sha256']
        with np.load(io.BytesIO(pred_data), allow_pickle=False) as arrays:
            assert 'y' not in arrays.files
            assert np.array_equal(arrays['row_ids'], inner)
            assert all(np.isfinite(arrays[n]).all() for n in arrays.files if arrays[n].dtype.kind in 'biufc')
    assert 'natural_exit.json' not in [PurePosixPath(n).name for n in z.namelist()]
proof = {
    'status': 'ACTUAL_CONTINUE100_LIVE_CAPSULE_AND_ORIGINAL220_RESUME_PREFIX_AUDIT_PASSED',
    'actual_local_audit_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'actual_remote_capture_utc': receipt['actual_utc'], 'run_root': RUN,
    'actual_resumed_training_start_utc': start['actual_utc'],
    'wrapper_pid': launch['wrapper_pid'], 'child_pid': start['pid'],
    'resumed_epoch': 10, 'resumed_optimizer_steps': 220,
    'captured_complete_epoch': progress['epoch'], 'captured_complete_updates': progress['optimizer_steps'],
    'captured_fit_step_entries': len(steps), 'last_fit_step_observed': steps[-1]['optimizer_steps'],
    'original_first10_history_exact': True, 'frozen_fit_orders_every_captured_step_exact': True,
    'source_and_plan_unchanged': True, 'capture_sha256': receipt['snapshot_sha256'],
    'original_capture_natural_exit_sha256': sha(A/'actual_capture_exit.json'),
    'members': receipt['members'], 'capture_source_sha256': receipt['capture_source_sha256'],
    'local_auditor_sha256': sha(__file__),
    'formal100_complete': False, 'new_generalization_or_donor_benefit_claimed': False,
    'whole_checkpoint_saved_in_this_live_capsule': False,
    'scope': 'Original per-file live evidence and original full last10 preservation association; not atomic global model state, a new whole checkpoint, other-node CPU model forward, or final performance.'
}
(BASE/'local_live_capture_joint_audit.json').write_text(json.dumps(proof, indent=2)+'\n', encoding='utf-8')
print(json.dumps(proof))
