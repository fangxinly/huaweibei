"""Fixed existing paths x fixed scalar selectors, no new generation or training."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path('D:/CodexBackups/selective_flow_20261003_1105')
new_source = root / 'oof_utility_completed_20261006T033054Z/original_C/execute/predictions_frozen.npz'
old_source = root / 'train_oracle_completed_20261006T0158Z/c/out/diagnostics.npz'
assert sha(new_source) == '068b0dca18dcaa571f687e262b1b2438983d132374b89b82240abdb9689b6f0f'
assert sha(old_source) == '92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658'
with np.load(new_source, allow_pickle=False) as z:
    row, video = z['row'].copy(), z['video'].copy()
    new_path, teacher_mu = z['prediction_path'].astype(np.float64), z['mu'].astype(np.float64)
with np.load(old_source, allow_pickle=False) as z:
    assert np.array_equal(row, z['row']) and np.array_equal(video, z['video'])
    old_path = z['learned_prediction_path'].astype(np.float64)
    learned_mu = z['p0'].astype(np.float64) - z['estimated_residual'].astype(np.float64)
assert np.max(np.abs(new_path[:, 0] - old_path[:, 0])) <= 1e-6
assert all(np.isfinite(v).all() for v in (new_path, old_path, teacher_mu, learned_mu))
paths = {'old_learned': old_path, 'oof_generated': new_path}
targets = {'old_learned': learned_mu, 'oof_teacher': teacher_mu}
out = Path(__file__).resolve().parents[1] / 'outputs'
frozen_path = out / '现有两路径交叉接受器CPU冻结数组.npz'
assert not frozen_path.exists()
frozen = {'row': row, 'video': video, 'old_learned_target': learned_mu, 'oof_teacher_target': teacher_mu}
for generator, values in paths.items():
    for selector, target in targets.items():
        name = generator + '__' + selector
        step = np.argmin((values - target[:, None])**2, axis=1)
        frozen[name + '_step'] = step
        frozen[name + '_prediction'] = values[np.arange(len(row)), step]
np.savez_compressed(frozen_path, **frozen)
frozen_sha = sha(frozen_path)
# True TRAIN outcomes are accessed only after these fixed-rule selections are frozen.
with np.load(old_source, allow_pickle=False) as z:
    y = z['y'].astype(np.float64)
results = {}
with np.load(frozen_path, allow_pickle=False) as z:
    assert sha(frozen_path) == frozen_sha
    for generator, values in paths.items():
        base_prediction = values[:, 0]
        baseline_error = (base_prediction - y)**2
        for selector, target in targets.items():
            name = generator + '__' + selector
            step, prediction = z[name + '_step'], z[name + '_prediction']
            assert np.array_equal(step, np.argmin((values - target[:, None])**2, axis=1))
            assert np.array_equal(prediction, values[np.arange(len(row)), step])
            selected_error = (prediction - y)**2
            modified = step != 0
            results[name] = {'mse': float(np.mean(selected_error)),
                             'mae': float(np.mean(np.abs(prediction - y))),
                             'raw_mse': float(np.mean(baseline_error)),
                             'net_gain_fraction': float(1 - np.mean(selected_error) / np.mean(baseline_error)),
                             'selected_step_counts': np.bincount(step, minlength=4).tolist(),
                             'modified_rows': int(modified.sum()),
                             'harmful_fraction_among_modified': float(np.mean(selected_error[modified] > baseline_error[modified])) if modified.any() else None}
proof = {'status': 'EXISTING_PATHS_FIXED_2_BY_2_ACCEPTANCE_CPU_DIAGNOSTIC_COMPLETED_NOT_NEW_SOLVER',
         'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rows': len(row),
         'selector': 'Earliest minimum float64 (prediction - scalar_target)^2; same 4-step pools include original F',
         'conditions': results, 'frozen_selection_array_sha256': frozen_sha,
         'new_input_sha256': sha(new_source), 'old_input_sha256': sha(old_source),
         'source_sha256': sha(__file__), 'labels_enter_selector': False,
         'metrics_read_after_selection_array_written_and_sha_frozen': True,
         'actual_gpu_execution': False, 'new_candidate_generation': False, 'calibration_parameters_fitted': False,
         'claim_boundary': 'Exploratory fixed full-TRAIN-fit/DEV-selected reference; old learned scalar is trained, not OOF; not independent confirmation, new control training or generalization.'}
result_path = out / '现有两路径交叉接受器CPU诊断.json'
assert not result_path.exists()
result_path.write_text(json.dumps(proof, indent=2), encoding='utf-8')
print(json.dumps(proof))
