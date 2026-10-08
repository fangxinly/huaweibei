"""Independent NumPy/AST preparation checks; no real outcome labels or GPU."""
from pathlib import Path
import ast, datetime, hashlib, importlib.util, json
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    work = Path(__file__).resolve().parent
    root = work/'correction_calibration_preparation_20261006T0354Z'
    plan = json.loads((root/'same_teacher_collection_plan_v1.json').read_text(encoding='utf-8'))
    roles_path = root/'calibration_video_role_plan_v1.json'
    assert sha(roles_path) == plan['role_plan_sha256']
    roles = json.loads(roles_path.read_text(encoding='utf-8'))
    for name, expected in plan['new_source_sha256'].items():
        assert sha(root/name) == sha(work/name) == expected
        ast.parse((root/name).read_text(encoding='utf-8'))
    mapping = json.loads((work/'train_row_video_mapping.json').read_text(encoding='utf-8'))
    videos = np.asarray([row['video_id'] for row in mapping])
    with np.load(root/'row_fold_calibration_roles.npz', allow_pickle=False) as z:
        row, fold, video, role = [z[key].copy() for key in ['row_ids', 'fold', 'video', 'role']]
    assert np.array_equal(row, np.arange(1281)) and np.array_equal(video, videos)
    assert sha(root/'row_fold_calibration_roles.npz') == roles['role_array_sha256']
    all_cal, all_eval, results = [], [], []
    source = work/'group_teacher_plan_20261005T1650Z'
    for record in roles['folds']:
        k = record['fold']
        arrays = {name: np.load(source/f'{name}_{k}.npy', allow_pickle=False)
                  for name in ['fit', 'inner', 'outer']}
        split = {name: np.load(root/f'{name}_{k}.npy', allow_pickle=False)
                 for name in ['calibration', 'evaluation']}
        for name, ids in split.items():
            expected = record['role_files'][f'{name}_{k}.npy']
            assert sha(root/f'{name}_{k}.npy') == expected['sha256']
            assert np.array_equal(ids, np.sort(ids)) and len(np.unique(ids)) == len(ids)
            assert np.array_equal(row[(fold == k)&(role == name)], ids)
            assert set(videos[ids]) == set(record[name+'_videos'])
        assert not set(split['calibration'])&set(split['evaluation'])
        assert not set(videos[split['calibration']])&set(videos[split['evaluation']])
        assert np.array_equal(np.sort(np.r_[split['calibration'], split['evaluation']]), arrays['outer'])
        for own_training_role in ['fit', 'inner']:
            assert not set(videos[arrays[own_training_role]])&set(videos[arrays['outer']])
        ranked = sorted(set(videos[arrays['outer']]),
                        key=lambda v: (hashlib.sha256((roles['salt']+'|'+str(k)+'|'+v).encode()).hexdigest(), v))
        assert record['calibration_videos'] == ranked[:(len(ranked)+2)//3]
        all_cal.extend(split['calibration'].tolist())
        all_eval.extend(split['evaluation'].tolist())
        results.append({'fold': k, 'fit': len(arrays['fit']), 'inner': len(arrays['inner']),
                        'calibration': len(split['calibration']), 'evaluation': len(split['evaluation']),
                        'video_disjoint': True, 'same_Tk_training_targets_only': True})
    assert sorted(all_cal+all_eval) == list(range(1281))
    assert len(set(videos[all_cal])) == 18 and len(set(videos[all_eval])) == 34
    guard = load(root/'label_free_group_teacher_inputs_v1.py', 'new_inference_guard')
    class PoisonOutcome:
        def __array__(self, *args, **kwargs):
            raise AssertionError('Outcome conversion was attempted')
    class Author:
        @staticmethod
        def get_appropriate_dataset(samples):
            assert all(np.array_equal(sample[1], np.zeros((1, 1))) for sample in samples)
            return samples
    examples = [(('feature', i), PoisonOutcome(), ('segment', i)) for i in range(1281)]
    fold_ids = {name: np.load(source/f'{name}_0.npy', allow_pickle=False) for name in ['outer', 'inner', 'fit']}
    inputs = guard.LabelFreeTeacherInputs(examples, fold_ids, Author())
    assert len(inputs.dataset(fold_ids['fit'], 'fit')) == 695
    assert len(inputs.dataset(fold_ids['inner'], 'inner')) == 153
    rejected = 0
    for rows, role_name in [(fold_ids['outer'][:1], 'fit'), (fold_ids['outer'][:1], 'outer'),
                            (fold_ids['fit'][:1], 'inner'), (np.repeat(fold_ids['fit'][:1], 2), 'fit')]:
        try:
            inputs.dataset(rows, role_name)
        except PermissionError:
            rejected += 1
        else:
            raise AssertionError('Guard accepted unauthorized row request')
    assert rejected == 4
    runtime = ast.parse((work/'group_teacher_runtime_v2.py').read_text(encoding='utf-8'))
    masked = next(n for n in runtime.body if isinstance(n, ast.FunctionDef) and n.name == 'plain_masked')
    assert not any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load) and n.id == 'label_ids' for n in ast.walk(masked))
    collector = ast.parse((root/'collect_group_teacher_fit_inner_v1.py').read_text(encoding='utf-8'))
    assert not any(isinstance(n, ast.Attribute) and n.attr in ['backward', 'step'] for n in ast.walk(collector))
    assert not any(isinstance(n, ast.Constant) and n.value in ['dev', 'test'] for n in ast.walk(collector))
    primitive = load(root/'message_output_backtrack_v1.py', 'prepared_output_backtrack')
    original = np.zeros((3, 1), dtype=np.float64)
    proposal = np.ones_like(original)
    terminal = lambda message: np.expm1(message[:, 0])
    selected = primitive.backtrack_message(original, proposal, terminal, np.full(3, .1), .02)
    assert selected['accepted'].all()
    assert np.max(np.abs(selected['prediction'])) <= .02
    assert np.array_equal(selected['prediction'], terminal(selected['messages']))
    free_output_interpolation = selected['fraction']*terminal(proposal)
    assert np.max(np.abs(free_output_interpolation-selected['prediction'])) > .001
    wrong_direction = primitive.backtrack_message(original, proposal, terminal, np.full(3, -.1), .02)
    assert not wrong_direction['accepted'].any() and np.array_equal(wrong_direction['messages'], original)
    zero_cap = primitive.backtrack_message(original, proposal, terminal, np.full(3, .1), 0)
    assert not zero_cap['accepted'].any()
    large_empirical_penalty = primitive.backtrack_message(original, proposal, terminal, np.full(3, .1), .02, epsilon=10)
    assert not large_empirical_penalty['accepted'].any()
    cross_row_rejected = False
    try:
        primitive.backtrack_message(np.zeros((2, 1)), np.ones((2, 1)),
            lambda z: z[:, 0]+z[:, 0].mean(), np.array([2., 0.]), 3)
    except ValueError as error:
        assert 'cross-row' in str(error)
        cross_row_rejected = True
    assert cross_row_rejected and not selected['true_labels_consumed']
    result = {'status': 'LOCAL_VIDEO_ROLES_GUARDS_AND_NONLINEAR_MESSAGE_BACKTRACK_CHECKED_NOT_GPU_EXECUTED',
        'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rows': 1281, 'videos': 52,
        'calibration_rows': len(all_cal), 'evaluation_rows': len(all_eval),
        'calibration_videos': 18, 'evaluation_videos': 34, 'folds': results,
        'poison_label_adapter_passed': True, 'unauthorized_role_requests_rejected': rejected,
        'nonlinear_terminal_exact_message_replay_passed': True,
        'nonlinear_message_control_differs_from_output_interpolation': True,
        'zero_cap_and_wrong_direction_fallback_passed': True, 'cross_row_coupling_rejected': True,
        'collector_AST_optimizer_step_or_backward_absent': True,
        'actual_gpu_precheck': False, 'actual_fit_inner_scalar_collection': False,
        'actual_calibrator_fit': False, 'actual_new_message_risk_result': False,
        'actual_new_training': False, 'fresh_remote_state_known': False,
        'true_conditional_risk_bound_proved': False, 'whole_pipeline_crossfit': False,
        'prior_train_exploration_and_full_train_reference_retained': True,
        'collection_plan_sha256': sha(root/'same_teacher_collection_plan_v1.json'),
        'role_plan_sha256': sha(roles_path), 'new_source_sha256': plan['new_source_sha256'],
        'scope': 'Preparation and meaningful synthetic/role-guard tests only. Actual teacher reload, norms, labels, GPU memory/time and message benefit still require real remote execution.'}
    out = work.parent/'outputs/纠错幅度校准角色与消息回退本地机制核验.json'
    assert not out.exists()
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['status', 'calibration_rows', 'evaluation_rows',
                                                 'calibration_videos', 'evaluation_videos', 'folds']}))


if __name__ == '__main__':
    main()
