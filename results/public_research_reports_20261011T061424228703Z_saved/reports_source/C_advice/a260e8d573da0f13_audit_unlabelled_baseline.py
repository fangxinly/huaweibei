"""Reviewer-only local audit. Never access outcome label members or fit parameters."""
from pathlib import Path
import hashlib
import json
import numpy as np

root = Path(__file__).resolve().parents[1]
backup = Path('D:/CodexBackups/selective_flow_20261003_1105')
pred = backup / 'oof_utility_completed_20261006T033054Z/original_C/execute/predictions_frozen.npz'
old = backup / 'train_oracle_completed_20261006T0158Z/c/out/diagnostics.npz'
expected = {
    pred: '068b0dca18dcaa571f687e262b1b2438983d132374b89b82240abdb9689b6f0f',
    old: '92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658',
}
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
for p, digest in expected.items():
    assert sha(p) == digest
with np.load(pred, allow_pickle=False) as z:
    members = list(z.files)
    row, video, fold, p0, mu = [z[k].copy() for k in ('row', 'video', 'fold', 'p0', 'mu')]
    pf = z['prediction_path'][:, 0].astype(np.float64)
with np.load(old, allow_pickle=False) as z:
    assert np.array_equal(row, z['row']) and np.array_equal(video, z['video'])
    assert np.array_equal(p0, z['p0'])
    pf_error = float(np.max(np.abs(pf - z['raw_prediction'].astype(np.float64))))
    old_target = z['p0'].astype(np.float64) - z['estimated_residual'].astype(np.float64)
assert pf_error <= 1e-6
assert np.array_equal(row, np.arange(1281)) and len(np.unique(video)) == 52
direction_sign_agreement = float(np.mean(np.sign(old_target-pf) == np.sign(mu.astype(np.float64)-pf)))
collection = backup / 'same_teacher_collection_actual_20261006T053835Z'
digests = {
    ('a', 'fit'): '0884ea7a8b127e2e20a124edae6a58264afc29226b96d854a070e1aedbc43afc',
    ('a', 'inner'): '5b6ffbf9e0877c64c7ba18cf7f0ba097ce9ecdfb2f79ebe73ebb872179efea0a',
    ('b', 'fit'): '777c3e5a2f0072249484a15d500566b17aa083dda6207881b4d6104db6068ae3',
    ('b', 'inner'): '4eabe95fb8fcedcf71a5c5ae0b1aa5b216942ed3bde24d1c927a4461494587c9',
    ('c', 'fit'): '65ae4fedd9c0bb2d0682bdb1d90ea0067e01c8d6e07f8d9d546a6d8553e827c2',
    ('c', 'inner'): '5884828e911ae7d0e7d6d99ab896d7974421df81599a5267e5ffa962403c0270',
}
scalars = []
for k, node in enumerate(('a', 'b', 'c')):
    for role in ('fit', 'inner'):
        p = collection / node / 'execute' / (role + '_scalar_inputs.npz')
        assert sha(p) == digests[(node, role)]
        with np.load(p, allow_pickle=False) as z:
            assert set(z.files) == {'row_ids', 'fold', 'mu'}
            rows, teacher_fold, values = [z[n].copy() for n in ('row_ids', 'fold', 'mu')]
            assert len(np.unique(rows)) == len(rows)
            assert np.all(teacher_fold == k) and np.isfinite(values).all()
            assert not np.any(fold[rows] == k)
        scalars.append(dict(path=str(p), sha256=digests[(node, role)], rows=len(rows),
                            fold=k, role=role, no_own_outer_rows=True))
result = dict(
    scope='Local reviewer audit of unlabelled arrays; no label member access, CAL fit, GPU or model forward',
    outcome_labels_read=False, original_sources_modified=False,
    prediction_members=members, prediction_sha256=expected[pred], old_sha256=expected[old],
    rows=len(row), videos=len(np.unique(video)), raw_replay_max_error=pf_error,
    p0_minus_pF_rms=float(np.sqrt(np.mean((p0.astype(np.float64)-pf)**2))),
    p0_minus_pF_max_abs=float(np.max(np.abs(p0.astype(np.float64)-pf))),
    teacher_minus_pF_rms=float(np.sqrt(np.mean((mu.astype(np.float64)-pf)**2))),
    old_target_and_teacher_target_sign_agreement=direction_sign_agreement,
    sign_agreement_is_not_actual_gpu_gradient_measurement=True,
    scalar_inputs=scalars,
)
(root / 'work/unlabelled_baseline_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(result, ensure_ascii=False, indent=2))
