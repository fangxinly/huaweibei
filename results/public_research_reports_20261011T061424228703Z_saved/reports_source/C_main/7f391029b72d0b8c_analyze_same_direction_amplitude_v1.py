"""Post-freeze TRAIN restricted-class calculation, not calibration or a controller."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


base = Path('D:/CodexBackups/selective_flow_20261003_1105')
prediction_path = base / 'oof_utility_completed_20261006T033054Z/original_C/execute/predictions_frozen.npz'
labels_path = prediction_path.with_name('metric_labels.npz')
old_path = base / 'train_oracle_completed_20261006T0158Z/c/out/diagnostics.npz'
assert sha(prediction_path) == '068b0dca18dcaa571f687e262b1b2438983d132374b89b82240abdb9689b6f0f'
assert sha(old_path) == '92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658'
with np.load(prediction_path, allow_pickle=False) as z:
    row, video, fold, mu, path, selected = [z[k].copy() for k in
                                         ('row', 'video', 'fold', 'mu', 'prediction_path', 'selected_prediction')]
with np.load(labels_path, allow_pickle=False) as z:
    y = z['y'].astype(np.float64)
with np.load(old_path, allow_pickle=False) as z:
    assert np.array_equal(row, z['row']) and np.array_equal(video, z['video'])
    assert np.array_equal(y, z['y'].astype(np.float64))
    original_old = z['raw_prediction'].astype(np.float64)
    old_control = z['learned_prediction_path'][:, -1].astype(np.float64)
assert np.array_equal(row, np.arange(1281)) and len(np.unique(video)) == 52
raw = path[:, 0].astype(np.float64)
assert np.max(np.abs(raw - original_old)) <= 1e-6
error = raw - y
raw_mse = float(np.mean(error**2))
old_mse = float(np.mean((old_control - y)**2))


def restricted(endpoint):
    displacement = endpoint.astype(np.float64) - raw
    linear = float(np.mean(2 * error * displacement))
    quadratic = float(np.mean(displacement**2))
    stationary = -linear / (2 * quadratic) if quadratic else 0.0
    coefficient = float(np.clip(stationary, 0.0, 1.0))
    direct_minimum = float(np.mean((raw + coefficient * displacement - y)**2))
    derived_minimum = raw_mse + coefficient * linear + coefficient**2 * quadratic
    assert abs(direct_minimum - derived_minimum) <= 1e-14
    # Check the derived risk polynomial against actual outputs, including both boundaries.
    for scalar in np.linspace(0, 1, 101):
        direct = float(np.mean((raw + scalar * displacement - y)**2))
        derived = raw_mse + scalar * linear + scalar**2 * quadratic
        assert abs(direct - derived) <= 1e-14 and direct_minimum <= direct + 1e-14
    return {'linear_coefficient': linear, 'quadratic_coefficient': quadratic,
            'posthoc_scalar_stationary': stationary, 'posthoc_scalar_clipped_0_1': coefficient,
            'restricted_minimum_mse': direct_minimum,
            'restricted_gain_fraction_vs_raw': (raw_mse - direct_minimum) / raw_mse,
            'cannot_beat_old_learned_endpoint_with_one_global_scalar': direct_minimum > old_mse,
            'gap_to_old_learned_mse': direct_minimum - old_mse,
            'actual_msg_controller_executed': False, 'coefficient_selected_for_deployment': False}


results = {f'new_path_step_{j}': restricted(path[:, j]) for j in range(1, 4)}
results['new_mu_selected_output'] = restricted(selected)
results['teacher_prediction_output'] = restricted(mu)
result = {'status': 'POSTFREEZE_TRAIN_GLOBAL_OUTPUT_SHRINK_RESTRICTED_CLASS_EXACT_NOT_CALIBRATION',
          'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'rows': 1281, 'videos': 52, 'raw_f_mse': raw_mse, 'old_learned_control_mse': old_mse,
          'old_learned_gain_fraction': (raw_mse - old_mse) / raw_mse,
          'conditions': results,
          'input_prediction_sha256': sha(prediction_path), 'input_old_diagnostic_sha256': sha(old_path),
          'input_metric_labels_sha256': sha(labels_path), 'source_sha256': sha(__file__),
          'claims': {'only_minimum_in_fixed_direction_one_global_scalar_class': True,
                     'not_bound_for_sample_dependent_or_nonlinear_message_control': True,
                     'not_independent_calibration_or_evaluation': True,
                     'not_new_gpu_or_parameter_update': True,
                     'reference_full_train_fitted_dev_selected': True,
                     'no_cal_eval_subgroup_metrics_or_fitting': True},
          'calibration_parameters_fitted': False, 'actual_message_output_interpolation_deployed': False}
out = Path(__file__).resolve().parents[1] / 'outputs/固定新路径全局幅度收缩受限最优CPU分析.json'
assert not out.exists()
out.write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps(result))
