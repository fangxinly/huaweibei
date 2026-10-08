"""Independent CPU certificate audit; not a remote audit or model forward."""
from pathlib import Path
import datetime
import hashlib
import json
import sys
import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


directory = Path(sys.argv[1]).resolve()
result = json.loads((directory / 'result.json').read_text(encoding='utf-8'))
assert result['status'] == 'LABEL_KNOWN_POSTHOC_TRAIN_FIXED_OUTPUT_CONVEX_SPAN_ORACLE_DIAGNOSTIC'
assert result['claims']['labels_used_in_posthoc_coefficient_optimization'] is True
assert result['actual_gpu'] is result['actual_message_controller'] is result['actual_calibration'] is False
assert sha(directory / 'label_free_frozen_basis.npz') == result['basis_sha256']
base = Path('D:/CodexBackups/selective_flow_20261003_1105')
new_path = base / 'oof_utility_completed_20261006T033054Z/original_C/execute/predictions_frozen.npz'
old_path = base / 'train_oracle_completed_20261006T0158Z/c/out/diagnostics.npz'
labels_path = new_path.with_name('metric_labels.npz')
assert sha(new_path) == result['new_input_sha256'] == '068b0dca18dcaa571f687e262b1b2438983d132374b89b82240abdb9689b6f0f'
assert sha(old_path) == result['old_input_sha256'] == '92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658'
assert sha(labels_path) == result['labels_sha256'] == 'ffd3f8b52b6620e616f0ec715b64fdb0398d0e7fc4b8be2d252234cfe2ff769f'
with np.load(directory / 'label_free_frozen_basis.npz', allow_pickle=False) as z:
    assert set(z.files) == {'row', 'video', 'raw', 'old_output', 'new_outputs'}
    row, video, raw, old, new = [z[k].copy() for k in ('row', 'video', 'raw', 'old_output', 'new_outputs')]
with np.load(new_path, allow_pickle=False) as z:
    assert np.array_equal(row, z['row']) and np.array_equal(video, z['video'])
    assert np.array_equal(raw, z['prediction_path'][:, 0].astype(np.float64))
    assert np.array_equal(new[:, :3], z['prediction_path'][:, 1:].astype(np.float64))
    assert np.array_equal(new[:, 3], z['selected_prediction'].astype(np.float64))
with np.load(old_path, allow_pickle=False) as z:
    assert np.array_equal(old, z['learned_prediction_path'][:, -1].astype(np.float64))
with np.load(labels_path, allow_pickle=False) as z:
    y = z['y'].astype(np.float64)
assert np.array_equal(row, np.arange(1281)) and len(np.unique(video)) == 52
checks = []
for j, (name, condition) in enumerate(result['conditions'].items()):
    weights = np.array([condition['posthoc_oracle_old_weight'], condition['posthoc_oracle_new_weight'],
                        condition['original_F_weight']])
    assert np.all(weights >= -1e-14) and abs(weights.sum() - 1) < 1e-14
    outputs = np.column_stack((old, new[:, j], raw))
    fitted = outputs @ weights
    error = fitted - y
    mse = float(error @ error / len(y))
    assert abs(mse - condition['simplex_minimum_mse']) < 1e-14
    # For convex mean-square loss, supporting all three vertices certifies
    # a global minimum over their output simplex, independently of the solver.
    certificate = 2 * np.mean(error[:, None] * (outputs - fitted[:, None]), axis=0)
    assert certificate.min() >= -1e-12
    direction = outputs[:, :2] - raw[:, None]
    gram = np.einsum('ni,nj->ij', direction, direction) / len(y)
    assert np.linalg.eigvalsh(gram).min() >= -1e-14
    assert np.max(np.abs(gram - condition['gram'])) < 1e-14
    assert abs(result['raw_mse'] - np.mean((raw - y)**2)) < 1e-14
    checks.append({'condition': name, 'mse_rebuilt': mse, 'support_certificate_min': float(certificate.min())})
receipt = {'status': 'LOCAL_INDEPENDENT_NUMPY_CONVEX_CERTIFICATE_AUDIT_PASSED',
           'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'rows': 1281, 'videos': 52,
           'result_sha256': sha(directory / 'result.json'), 'basis_sha256': result['basis_sha256'],
           'audit_source_sha256': sha(__file__), 'conditions': checks,
           'posthoc_label_known_only': True, 'remote_CPU_or_model_forward': False,
           'new_calibration_or_message_generation': False}
destination = directory / 'independent_cpu_audit.json'
assert not destination.exists()
destination.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
print(json.dumps(receipt))
