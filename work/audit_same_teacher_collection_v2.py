"""Independent NumPy receipt/array audit; no Torch or full-model CPU forward."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import math
import numpy as np


PLAN_SHA = '57b740fed4ee9ac2d9736956f2e6cbc3b9dcd30b102610abe49f563a493738f7'
WRAPPER_SHA = 'd0483173b6f3e9bba63389e5fd2d13e514a358ab43b103838672ab0d29842218'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def time_value(value):
    result = datetime.datetime.fromisoformat(value)
    assert result.tzinfo is not None
    return result


def audit_receipt_contract(receipt, inventory, exit_proof, plan, selected, phase):
    """Only actual original artifacts may satisfy this contract in audit()."""
    assert phase in ('precheck', 'execute')
    assert receipt['passed'] is True and receipt['phase'] == phase
    assert receipt['fold'] == selected['fold']
    assert receipt['source_sha256'] == plan['collector_sha256']
    assert receipt['plan_sha256'] == inventory['plan_sha256'] == exit_proof['plan_sha256'] == PLAN_SHA
    assert receipt['checkpoint_sha256'] == selected['full_checkpoint_sha256']
    assert receipt['before_tensor_sha256'] == receipt['after_tensor_sha256'] == selected['selected_model_tensor_sha256']
    error = receipt['inner_strict_original_input_disk_replay_max_error']
    assert math.isfinite(error) and 0 <= error <= plan['inner_original_input_disk_replay_tolerance']
    assert receipt['label_replacement_max_error'] == 0
    assert receipt['fit_only_normalization_buffers_exact'] is True
    assert receipt['optimizer_objects_constructed_then_discarded'] is True
    assert receipt['optimizer_steps'] == 0 and receipt['parameter_gradients_present'] is False
    for key in ('dev_requested', 'test_requested', 'whole_pipeline_crossfit', 'new_outer_inference'):
        assert receipt[key] is False, key
    assert receipt['collection_outputs_written'] is (phase == 'execute')
    assert receipt['whole_process_peak_includes_model_construction'] is True
    assert 0 < receipt['actual_peak_allocated_bytes'] <= plan['maximum_peak_allocated_bytes']
    assert 0 < receipt['seconds'] <= plan['maximum_seconds']
    assert 0 <= receipt['conservative_collection_seconds'] <= plan['maximum_seconds']
    assert 0 < receipt['actual_new_payload_bytes_before_receipt'] <= plan['maximum_new_payload_bytes']
    assert inventory['phase'] == phase and inventory['compute'] == ''
    assert inventory['gpu'].split(',')[0].strip() == selected['expected_uuid']
    assert inventory['free_bytes'] >= plan['minimum_remote_free_bytes']
    assert inventory['platform_deadline_verified'] is False
    assert inventory['remaining_estimate_seconds'] > plan['maximum_seconds'] + plan['preservation_margin_seconds']
    argv = inventory['argv']
    assert Path(argv[0]).name == 'collect_group_teacher_fit_inner_v2.py'
    assert len(argv) == 7 and argv[1] == '--plan' and argv[3:] == ['--fold', str(selected['fold']), '--phase', phase]
    assert exit_proof['child_argv'] == argv and exit_proof['child_pid'] == inventory['pid']
    assert exit_proof['phase'] == phase and exit_proof['fold'] == selected['fold'] and exit_proof['exit_code'] == 0
    assert exit_proof['collector_sha256'] == plan['collector_sha256']
    start, inv, rec, end = [time_value(v) for v in
                          (exit_proof['started_utc'], inventory['utc'], receipt['utc'], exit_proof['finished_utc'])]
    assert start <= inv <= rec <= end
    assert (rec - inv).total_seconds() <= receipt['seconds'] + 1


def audit(directory, original_teacher_directory, fold, phase):
    directory, original = Path(directory), Path(original_teacher_directory)
    plan_path = directory / 'same_teacher_collection_plan_v2.json'
    assert sha(plan_path) == PLAN_SHA
    plan = read(plan_path)
    assert plan['version'] == 2 and plan['require_actual_phase_exit_receipts'] is True
    assert sha(directory / 'run_same_teacher_collection_v2.py') == WRAPPER_SHA
    assert sha(directory / 'calibration_video_role_plan_v1.json') == plan['role_plan_sha256']
    selected = plan['folds'][fold]
    assert selected['fold'] == fold
    checked = {}
    for name, expected in plan['new_source_sha256'].items():
        checked['collection/' + name] = sha(directory / name)
        assert checked['collection/' + name] == expected, name
    for name, expected in {**plan['shared_teacher_pins'], **selected['original_small_pins']}.items():
        checked['original/' + name] = sha(original / name)
        assert checked['original/' + name] == expected, name
    completed, history = read(original / 'run_v1/completion.json'), read(original / 'run_v1/history.json')
    assert completed['fold'] == fold and completed['completed_epochs'] == len(history) == 100
    assert completed['best_epoch'] == min(range(1, 101), key=lambda e: history[e - 1]['inner_sample_mse'])
    assert completed['full_checkpoint_sha256'] == selected['full_checkpoint_sha256']
    assert completed['full_checkpoint_bytes'] == selected['full_checkpoint_bytes']
    assert completed['selected_model_tensor_sha256'] == selected['selected_model_tensor_sha256']
    assert read(original / 'formal_exit.json')['exit_code'] == 0
    rows = {role: np.load(original / f'{role}_{fold}.npy', allow_pickle=False) for role in ('fit', 'inner', 'outer')}
    sets = [set(rows[role].tolist()) for role in ('fit', 'inner', 'outer')]
    assert set.union(*sets) == set(range(1281))
    assert all(not sets[i] & sets[j] for i in range(3) for j in range(i))
    for role in rows:
        assert len(rows[role]) == selected[role + '_rows'] == len(set(rows[role].tolist()))
    calibration_roles = []
    for name, expected in selected['role_files'].items():
        assert sha(directory / name) == expected['sha256']
        ids = np.load(directory / name, allow_pickle=False)
        assert ids.ndim == 1 and np.issubdtype(ids.dtype, np.integer)
        assert len(ids) == expected['rows'] == len(set(ids.tolist()))
        calibration_roles.append(set(ids.tolist()))
    assert not calibration_roles[0] & calibration_roles[1]
    assert set.union(*calibration_roles) == set(rows['outer'].tolist())
    phase_root = directory / phase
    receipt_path, inventory_path = phase_root / 'receipt.json', phase_root / 'inventory.json'
    exit_path = directory / (phase + '_exit.json')
    receipt, inventory, exit_proof = read(receipt_path), read(inventory_path), read(exit_path)
    audit_receipt_contract(receipt, inventory, exit_proof, plan, selected, phase)
    assert exit_proof['receipt_sha256'] == sha(receipt_path)
    assert exit_proof['wrapper_sha256'] == WRAPPER_SHA
    assert Path(inventory['argv'][2]).as_posix() == (Path(plan['teacher_root']) / plan['collection_subdirectory'] / plan_path.name).as_posix()
    journal = receipt['data_journal']
    assert len(journal) == 2
    for item, role in zip(journal, ('fit', 'inner')):
        assert item['role'] == role and item['rows'] == rows[role].tolist()
        assert item['dummy_label'] == 0 and item['outcome_label_index_read'] is False
    required_outputs = {role + '_scalar_inputs.npz' for role in ('fit', 'inner')} if phase == 'execute' else set()
    assert set(receipt['output_files']) == required_outputs
    assert {path.name for path in phase_root.glob('*_scalar_inputs.npz')} == required_outputs
    original_inner_error = None
    for name in required_outputs:
        record = receipt['output_files'][name]
        path = phase_root / name
        assert path.stat().st_size == record['bytes'] and sha(path) == record['sha256']
        role = name.split('_')[0]
        with np.load(path, allow_pickle=False) as arrays:
            assert set(arrays.files) == {'row_ids', 'fold', 'mu'}
            assert np.issubdtype(arrays['row_ids'].dtype, np.integer)
            assert np.issubdtype(arrays['fold'].dtype, np.integer)
            assert np.array_equal(arrays['row_ids'], rows[role])
            assert np.array_equal(arrays['fold'], np.full(len(rows[role]), fold))
            assert arrays['mu'].size == len(rows[role]) and np.isfinite(arrays['mu']).all()
            if role == 'inner':
                with np.load(original / 'run_v1/selected_inner_predictions.npz', allow_pickle=False) as old:
                    assert np.array_equal(old['row_ids'], arrays['row_ids'])
                    assert old['prediction'].shape == arrays['mu'].shape
                    original_inner_error = float(np.max(np.abs(old['prediction'] - arrays['mu'])))
                    assert original_inner_error <= plan['inner_original_input_disk_replay_tolerance']
        checked['outputs/' + name] = sha(path)
    if phase == 'execute':
        prior_audit_path = directory / 'precheck/independent_precheck_audit.json'
        prior = read(prior_audit_path)
        auth = read(directory / 'execution_authorization.json')
        assert prior['status'] == 'ACTUAL_COLLECTION_PRECHECK_METADATA_INDEPENDENT_AUDIT_PASSED'
        assert prior['fold'] == fold and prior['original_gpu_receipt_sha256'] == sha(directory / 'precheck/receipt.json')
        assert auth['precheck_independent_audit_sha256'] == sha(prior_audit_path)
        assert auth['original_precheck_receipt_sha256'] == prior['original_gpu_receipt_sha256']
        assert auth['plan_sha256'] == PLAN_SHA and auth['collector_sha256'] == plan['collector_sha256']
        assert auth['actual_gpu_precheck_and_independent_audit_passed'] is True
        assert auth['capture_source_coverage_verified'] is True and auth['capture_source_sha256'] == plan['capture_source_sha256']
        assert read(directory / 'precheck_exit.json')['exit_code'] == 0
    status = ('ACTUAL_COLLECTION_PRECHECK_METADATA_INDEPENDENT_AUDIT_PASSED' if phase == 'precheck'
              else 'ACTUAL_COLLECTION_FIT_INNER_SCALAR_ARRAYS_INDEPENDENT_AUDIT_PASSED')
    return {'status': status, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'fold': fold, 'phase': phase, 'plan_sha256': PLAN_SHA,
            'original_gpu_receipt_sha256': sha(receipt_path), 'actual_child_exit_sha256': sha(exit_path),
            'checked_files': checked, 'inner_array_replay_max_error': original_inner_error,
            'full_weight_new_downloaded': False, 'cpu_full_model_forward': False,
            'calibration_fitted': False, 'new_outer_inference': False,
            'audit_source_sha256': sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--original-teacher-directory', type=Path, required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1, 2], required=True)
    parser.add_argument('--phase', choices=['precheck', 'execute'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.directory, args.original_teacher_directory, args.fold, args.phase)
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(result['status'], flush=True)
