"""Freeze video roles and inference preparation only. No outcome array loaded."""
from pathlib import Path
import datetime, hashlib, json, math, shutil
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    workspace = Path(__file__).resolve().parent.parent
    work = workspace/'work'
    source_plan = work/'group_teacher_plan_20261005T1650Z'
    teacher_plan_path = source_plan/'group_teacher_plan_v3.json'
    parent = json.loads(teacher_plan_path.read_text(encoding='utf-8'))
    assert sha(teacher_plan_path) == 'f0878166da09433c27119a08fdfd8682f9b9e29f2bfe0a758f29eee45cf131a4'
    mapping_path = work/'train_row_video_mapping.json'
    assert sha(mapping_path) == '62536b398854df5c6abf1987ff6c8373aabc980a273687484da2df275149b12c'
    mapping = json.loads(mapping_path.read_text(encoding='utf-8'))
    assert [row['row'] for row in mapping] == list(range(1281))
    videos = np.asarray([row['video_id'] for row in mapping])
    destination = work/'correction_calibration_preparation_20261006T0354Z'
    assert not destination.exists(), 'Preserve prior preparation; never overwrite'
    destination.mkdir()
    salt = 'finite-message-amplitude-roles-v1-20261006T0354Z'
    fold_array = np.full(1281, -1, dtype=np.int64)
    role_array = np.full(1281, '', dtype='U11')
    folds = []
    completed_root = Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_completed_20261006')
    uuids = ['GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7',
             'GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067',
             'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa']
    for k, node in enumerate(('a', 'b', 'c')):
        original = completed_root/node
        completed = json.loads((original/'run_v1/completion.json').read_text(encoding='utf-8'))
        history = json.loads((original/'run_v1/history.json').read_text(encoding='utf-8'))
        assert completed['fold'] == k and completed['completed_epochs'] == len(history) == 100
        assert completed['best_epoch'] == min(range(1, 101), key=lambda e: history[e-1]['inner_sample_mse'])
        assert completed['inner_strict_disk_replay_max_error'] <= 1e-6
        ids = {role: np.load(source_plan/f'{role}_{k}.npy', allow_pickle=False) for role in ('fit', 'inner', 'outer')}
        for role, values in ids.items():
            assert sha(source_plan/f'{role}_{k}.npy') == parent['folds'][k]['files'][f'{role}_{k}.npy']['sha256']
            assert np.array_equal(values, np.sort(values))
        ranked = sorted(parent['folds'][k]['outer_videos'],
                        key=lambda video: (hashlib.sha256((salt+'|'+str(k)+'|'+video).encode()).hexdigest(), video))
        calibration_videos = ranked[:math.ceil(len(ranked)/3)]
        evaluation_videos = ranked[math.ceil(len(ranked)/3):]
        cal = ids['outer'][np.isin(videos[ids['outer']], calibration_videos)]
        eva = ids['outer'][np.isin(videos[ids['outer']], evaluation_videos)]
        assert not set(cal)&set(eva)
        assert np.array_equal(np.sort(np.concatenate([cal, eva])), ids['outer'])
        assert not set(videos[cal])&set(videos[eva])
        fold_array[ids['outer']] = k
        role_array[cal] = 'calibration'
        role_array[eva] = 'evaluation'
        files = {}
        for role, values in [('calibration', cal), ('evaluation', eva)]:
            path = destination/f'{role}_{k}.npy'
            np.save(path, values)
            files[path.name] = {'sha256': sha(path), 'rows': len(values), 'videos': sorted(set(videos[values]))}
        pins = {relative: sha(original/relative) for relative in
                ['run_v1/protocol.json', 'run_v1/history.json', 'run_v1/completion.json',
                 'run_v1/selected_inner_predictions.npz', 'run_v1/outer_scalar_predictions.npz',
                 'formal_exit.json', f'fit_{k}.npy', f'inner_{k}.npy', f'outer_{k}.npy']}
        folds.append({'fold': k, 'node': node, 'expected_uuid': uuids[k],
                      'fit_rows': len(ids['fit']), 'inner_rows': len(ids['inner']), 'outer_rows': len(ids['outer']),
                      'calibration_rows': len(cal), 'evaluation_rows': len(eva),
                      'calibration_videos': calibration_videos, 'evaluation_videos': evaluation_videos,
                      'role_files': files, 'original_small_pins': pins,
                      'full_checkpoint_sha256': completed['full_checkpoint_sha256'],
                      'full_checkpoint_bytes': completed['full_checkpoint_bytes'],
                      'selected_model_tensor_sha256': completed['selected_model_tensor_sha256']})
    assert (fold_array >= 0).all() and np.all(role_array != '') and len(np.unique(videos)) == 52
    np.savez_compressed(destination/'row_fold_calibration_roles.npz', row_ids=np.arange(1281),
                        fold=fold_array, video=videos, role=role_array)
    sources = ['label_free_group_teacher_inputs_v1.py', 'collect_group_teacher_fit_inner_v1.py',
               'message_output_backtrack_v1.py', 'prepare_correction_calibration_roles_v1.py']
    for name in sources:
        shutil.copy2(work/name, destination/name)
    roles = {'status': 'VIDEO_ROLES_FROZEN_BEFORE_NEW_CALIBRATOR_FIT_NOT_PRISTINE_CONFIRMATION',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'salt': salt,
        'rule': 'First ceil(number of OUTER videos/3) SHA-ranked videos are CAL; remainder EVAL',
        'rows': 1281, 'videos': 52, 'folds': folds,
        'role_array_sha256': sha(destination/'row_fold_calibration_roles.npz'),
        'labels_read_by_mapping_builder': False, 'metrics_read_by_mapping_builder': False,
        'prior_all_train_exploration': True, 'reference_full_train_fitted_dev_selected': True,
        'independent_confirmation_claimed': False, 'whole_pipeline_crossfit': False,
        'outer_cal_labels_forbidden_in_solver': True,
        'evaluation_labels_forbidden_before_prediction_freeze': True,
        'targets_for_student_fold': 'Only same T_k FIT/INNER scalar targets; never a mixture of T_j',
        'calibrator_rule_draft': 'CAL-only bounded least squares lambda in [0,1]; no lambda fitted in this preparation',
        'new_gpu_collection_executed': False, 'new_calibrator_fitted': False,
        'new_message_control_executed': False, 'new_training_started': False}
    role_path = destination/'calibration_video_role_plan_v1.json'
    role_path.write_text(json.dumps(roles, ensure_ascii=False, indent=2), encoding='utf-8')
    plan = {'status': 'LOCAL_FROZEN_SAME_TEACHER_COLLECTION_PROTOCOL_NOT_GPU_VALIDATED',
        'utc': roles['utc'], 'collector_sha256': sha(work/'collect_group_teacher_fit_inner_v1.py'),
        'new_source_sha256': {name: sha(work/name) for name in sources},
        'shared_teacher_pins': {'group_teacher_runtime_v2.py': sha(work/'group_teacher_runtime_v2.py'),
                               'group_teacher_plan_v3.json': sha(teacher_plan_path)},
        'folds': folds, 'role_plan_sha256': sha(role_path),
        'teacher_root': '/data/coding/group_teacher_v1_20261005T1650Z',
        'base_root': '/data/coding/soft_vector_research_20261005T1220Z',
        'collection_subdirectory': 'correction_calibration_preparation_20261006T0354Z',
        'capture_source_sha256': '6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad',
        'capture_coverage_requires_actual_audit': True,
        'maximum_seconds': 1200, 'maximum_peak_allocated_bytes': 4*1024**3,
        'minimum_remote_free_bytes': 2*1024**3, 'minimum_local_permanent_free_bytes': 1024**3,
        'maximum_new_payload_bytes': 1024**3, 'preservation_margin_seconds': 7200,
        'estimated_lease_end_utc': '2026-10-06T12:08:17+00:00', 'platform_deadline_verified': False,
        'budget_actual_gpu_validation_pending': True, 'old_full_weights_retransferred': False,
        'inference_batch_size': 128, 'inner_original_input_disk_replay_tolerance': 1e-6,
        'outcome_labels_consumed': False, 'optimizer_steps': 0,
        'original_setup_creates_discarded_optimizer_objects': True,
        'normalization': 'Same T_k FIT features only; new zero-label fit stats exact to stored checkpoint buffers',
        'new_outer_inference': False, 'student_training_authorized_by_this_plan': False}
    path = destination/'same_teacher_collection_plan_v1.json'
    path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': plan['status'], 'directory': str(destination),
                      'role_plan_sha256': sha(role_path), 'collection_plan_sha256': sha(path),
                      'fold_rows': [{key: f[key] for key in ['fold', 'calibration_rows', 'evaluation_rows']} for f in folds]}))


if __name__ == '__main__':
    main()
