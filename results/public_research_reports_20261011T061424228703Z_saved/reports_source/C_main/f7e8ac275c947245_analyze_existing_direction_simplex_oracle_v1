"""Label-known, posthoc TRAIN convex-output-span diagnostic; never a controller."""
from pathlib import Path
import datetime
import hashlib
import json
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[1]
backup = Path('D:/CodexBackups/selective_flow_20261003_1105')
new_path = backup / 'oof_utility_completed_20261006T033054Z/original_C/execute/predictions_frozen.npz'
old_path = backup / 'train_oracle_completed_20261006T0158Z/c/out/diagnostics.npz'
labels_path = new_path.with_name('metric_labels.npz')
assert sha(new_path) == '068b0dca18dcaa571f687e262b1b2438983d132374b89b82240abdb9689b6f0f'
assert sha(old_path) == '92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658'
assert sha(labels_path) == 'ffd3f8b52b6620e616f0ec715b64fdb0398d0e7fc4b8be2d252234cfe2ff769f'
with np.load(new_path, allow_pickle=False) as z:
    row, video = z['row'].copy(), z['video'].copy()
    raw = z['prediction_path'][:, 0].astype(np.float64)
    new_outputs = np.column_stack((z['prediction_path'][:, 1:], z['selected_prediction'])).astype(np.float64)
with np.load(old_path, allow_pickle=False) as z:
    assert np.array_equal(row, z['row']) and np.array_equal(video, z['video'])
    assert np.max(np.abs(raw - z['raw_prediction'])) <= 1e-6
    old_output = z['learned_prediction_path'][:, -1].astype(np.float64)
assert np.array_equal(row, np.arange(1281)) and len(np.unique(video)) == 52
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
out = root / 'work' / ('existing_direction_simplex_oracle_' + stamp)
out.mkdir(exist_ok=False)
basis_path = out / 'label_free_frozen_basis.npz'
np.savez_compressed(basis_path, row=row, video=video, raw=raw, old_output=old_output, new_outputs=new_outputs)
basis_sha = sha(basis_path)
# Only now read labels. These labels fit the following posthoc oracle coefficients;
# the coefficients are not label-free selection, CAL estimates, or deployable parameters.
with np.load(labels_path, allow_pickle=False) as z:
    y = z['y'].astype(np.float64)
with np.load(old_path, allow_pickle=False) as z:
    assert np.array_equal(y, z['y'].astype(np.float64))
error = raw - y
raw_mse = float(np.mean(error ** 2))
old_mse = float(np.mean((old_output - y) ** 2))
old_d = old_output - raw
old_alpha = float(np.clip(-np.mean(error * old_d) / np.mean(old_d ** 2), 0, 1))
old_shrunk_mse = float(np.mean((error + old_alpha * old_d) ** 2))
conditions = {}
for j, endpoint in enumerate(new_outputs.T):
    d = np.column_stack((old_d, endpoint - raw))
    gram, linear = d.T @ d / len(y), d.T @ error / len(y)
    assert np.linalg.eigvalsh(gram).min() >= -1e-14
    candidates = [(0., 0.), (1., 0.), (0., 1.)]
    candidates += [(float(np.clip(-linear[0] / gram[0, 0], 0, 1)), 0.),
                   (0., float(np.clip(-linear[1] / gram[1, 1], 0, 1)))]
    edge = d[:, 0] - d[:, 1]
    t = float(np.clip(-np.mean((error + d[:, 1]) * edge) / np.mean(edge ** 2), 0, 1))
    candidates.append((t, 1. - t))
    interior = np.linalg.solve(gram, -linear)
    if np.all(interior >= 0) and interior.sum() <= 1:
        candidates.append(tuple(interior))
    candidates = np.asarray(candidates, dtype=np.float64)
    values = np.mean((error[:, None] + d @ candidates.T) ** 2, axis=0)
    idx = int(np.argmin(values))
    c, mse = candidates[idx], float(values[idx])
    # Independent convex first-order certificate: the gradient supports the
    # minimizing point against every vertex of the feasible triangle.
    gradient = 2 * (gram @ c + linear)
    vertex_margins = (np.array([[0., 0.], [1., 0.], [0., 1.]]) - c) @ gradient
    assert vertex_margins.min() >= -1e-12
    assert abs(mse - (raw_mse + 2 * linear @ c + c @ gram @ c)) <= 1e-14
    max_identity_error, grid_min = 0., float('inf')
    for a in np.linspace(0, 1, 51):
        for b in np.linspace(0, 1 - a, 51):
            direct = float(np.mean((error + a * d[:, 0] + b * d[:, 1]) ** 2))
            w = np.array([a, b])
            max_identity_error = max(max_identity_error, abs(direct - (raw_mse + 2 * linear @ w + w @ gram @ w)))
            grid_min = min(grid_min, direct)
    assert max_identity_error <= 1e-14 and mse <= grid_min + 1e-14
    conditions['new_step_' + str(j + 1) if j < 3 else 'new_mu_selected'] = {
        'posthoc_oracle_old_weight': float(c[0]), 'posthoc_oracle_new_weight': float(c[1]),
        'original_F_weight': float(1 - c.sum()), 'simplex_minimum_mse': mse,
        'gain_fraction_vs_original_F': (raw_mse - mse) / raw_mse,
        'additional_mse_reduction_vs_old_global_shrink': old_shrunk_mse - mse,
        'output_direction_uncentered_cosine': float(gram[0, 1] / np.sqrt(gram[0, 0] * gram[1, 1])),
        'gram': gram.tolist(), 'linear': linear.tolist(),
        'convex_vertex_certificate_min': float(vertex_margins.min()),
        'direct_polynomial_max_error': max_identity_error, 'grid_min_mse': grid_min}
result = {
    'status': 'LABEL_KNOWN_POSTHOC_TRAIN_FIXED_OUTPUT_CONVEX_SPAN_ORACLE_DIAGNOSTIC',
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'rows': 1281, 'videos': 52, 'raw_mse': raw_mse, 'old_endpoint_mse': old_mse,
    'old_global_shrink_posthoc_alpha': old_alpha, 'old_global_shrink_minimum_mse': old_shrunk_mse,
    'conditions': conditions, 'basis_sha256': basis_sha,
    'source_sha256': sha(__file__), 'new_input_sha256': sha(new_path),
    'old_input_sha256': sha(old_path), 'labels_sha256': sha(labels_path),
    'claims': {'basis_written_and_sha_frozen_before_label_read': True,
               'labels_used_in_posthoc_coefficient_optimization': True,
               'feasible_class': 'One global convex mixture of original F, existing old endpoint and one existing new endpoint',
               'not_new_message_generation_or_model_training': True,
               'not_calibration_or_CAL_EVAL_subgroup_evaluation': True,
               'not_generalization_or_whole_pipeline_crossfit': True,
               'not_nonlinear_message_or_sample_dependent_gate_bound': True,
               'posthoc_coefficients_not_selected_for_deployment': True,
               'reference_full_TRAIN_fitted_DEV_selected': True},
    'actual_gpu': False, 'actual_message_controller': False, 'actual_calibration': False}
(out / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print(json.dumps({'output_dir': str(out), 'result': result}))
