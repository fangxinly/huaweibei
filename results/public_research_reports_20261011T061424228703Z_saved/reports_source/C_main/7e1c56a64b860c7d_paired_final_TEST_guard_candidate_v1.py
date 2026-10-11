"""Prospective joint TEST role gate; no loader, training, inference or executable CLI.

Trusted pickle can materialize label bytes. This audited access gate is not a
security sandbox. A separate exact-source execution protocol remains necessary.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import numpy as np

METHODS = ('minimal_fixed_F', 'careflow')
STATES = {
    'minimal_fixed_F': 'af2c58a2c354b92a5e55c7d3d2e01e2fd521d8b2f078b12ea5f59711c130da29',
    'careflow': 'd11ca1b31541a42495a96aa3559dd999812405c77658d90fa6fa8a0ad3a6d7f3',
}
CHECKPOINTS = {
    'minimal_fixed_F': '7ec2173145bee6a80d6e7367e03f98f49cf87bf5d9eee5eb53558dbf81c839e8',
    'careflow': '9890b4fd2138e49a6500c52e72b0cbcc58151dc26e4b927656b39b6aa9e280f7',
}

def require(condition, message):
    if not condition:
        raise PermissionError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()

def read_pinned(spec):
    data = Path(spec['path']).read_bytes()
    require(hashlib.sha256(data).hexdigest() == spec['sha256'], 'Physical pinned file changed')
    return json.loads(data.decode('utf-8'))

def canonical_id(value):
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    require(isinstance(value, str) and bool(value), 'Invalid original segment ID')
    return value

class PairedFinalTestGuard:
    def __init__(self, records, identity_spec):
        # Identity acquisition must already have its own frozen input-only plan.
        identity = read_pinned(identity_spec)
        require(identity['status'] == 'ACTUAL_OFFICIAL_TEST_INPUT_ONLY_IDENTITY_COMPLETE',
                'Input-only identity is not qualified')
        self.ids = tuple(canonical_id(x) for x in identity['test_ids'])
        require(bool(self.ids) and len(set(self.ids)) == len(self.ids), 'Duplicate TEST IDs')
        require(len(records) == len(self.ids), 'TEST row count changed')
        require(tuple(canonical_id(r[2]) for r in records) == self.ids,
                'TEST original complete order changed')
        other = set(identity['train_ids']) | set(identity['dev_ids'])
        require(not (set(self.ids) & other), 'TEST overlaps TRAIN/DEV IDs')
        self._records = records
        self.identity_spec = dict(identity_spec)
        self.journal = []
        self._label_attempted = False

    def inputs_only(self, dummy=0):
        require(dummy in (0, 7), 'Only declared dummy labels allowed')
        out = [(r[0], np.full((1, 1), dummy, dtype=np.float32), r[2])
               for r in self._records]
        self.journal.append({'role': 'test', 'labels_read': False,
                             'purpose': 'fixed_pair_prediction_only', 'rows': len(out)})
        return out

    def targets_once_after_joint_preservation(self, protocol_spec):
        require(not self._label_attempted, 'This instance already attempted labels')
        require(tuple(canonical_id(r[2]) for r in self._records) == self.ids,
                'Records changed order since input-only identity gate')
        read_pinned(self.identity_spec)
        plan = read_pinned(protocol_spec)
        require(plan['status'] == 'FINAL_TEST_PAIR_SCORE_PROTOCOL_FROZEN' and
                plan['methods'] == list(METHODS) and plan['batch'] == 128 and
                plan['keep_tail'] is True and plan['readout'] == 'fixed_direct' and
                plan['old_TEST_access_disclosed'] is True,
                'Final joint source/protocol not frozen')
        require(plan['identity'] == self.identity_spec, 'Identity protocol mismatch')
        require(plan['guard_source_sha256'] == sha(__file__), 'Final guard source changed')
        require(set(plan['predictions']) == set(METHODS), 'Both fixed methods required')
        frozen = {}
        for method in METHODS:
            spec = plan['predictions'][method]
            require(spec['state_sha256'] == STATES[method] and
                    spec['selected_checkpoint_sha256'] == CHECKPOINTS[method],
                    'Fixed selected model changed')
            # The joint source independently validates original process/captures.
            # This gate pins that completed audit and its physical copied arrays.
            joint = read_pinned(spec['preservation_joint'])
            require(joint['status'] == 'ACTUAL_FINAL_TEST_PREDICTION_D_OTHER_CPU_CAPTURE_JOINT_PASSED' and
                    joint['method'] == method and joint['labels_read'] is False and
                    joint['state_sha256'] == STATES[method] and
                    joint['selected_checkpoint_sha256'] == CHECKPOINTS[method] and
                    joint['identity_sha256'] == self.identity_spec['sha256'] and
                    joint['prediction_sha256'] == spec['sha256'] and
                    joint['original_child_natural_exit_code'] == 0,
                    'Both input-only prediction preservation joints required')
            data = Path(spec['D_prediction_path']).read_bytes()
            require(hashlib.sha256(data).hexdigest() == spec['sha256'], 'D prediction bytes mismatch')
            require(sha(spec['CPU_capture_prediction_path']) == spec['sha256'],
                    'CPU capture prediction bytes mismatch')
            with np.load(io.BytesIO(data), allow_pickle=False) as z:
                require(set(z.files) == {'row_ids', 'prediction', 'model_state_sha256'},
                        'Prediction-only schema required')
                require(tuple(z['row_ids'].tolist()) == self.ids and
                        str(z['model_state_sha256'].item()) == STATES[method],
                        'Prediction ID/state mismatch')
                p = z['prediction']
                require(p.dtype == np.float32 and p.shape == (len(self.ids),) and
                        np.isfinite(p).all(), 'Whole original finite FP32 predictions required')
                frozen[method] = p.copy()
        # Exclusive durable intent BEFORE reading any label. Failure after this
        # point burns the attempt; removing token to retry is forbidden.
        token = Path(plan['once_token_path'])
        fd = os.open(token, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        self._label_attempted = True
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump({'protocol_sha256': protocol_spec['sha256'],
                       'identity_sha256': self.identity_spec['sha256'],
                       'prediction_sha256': {m: plan['predictions'][m]['sha256'] for m in METHODS},
                       'intent_before_labels': True}, f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        labels = []
        for record in self._records:
            a = np.asarray(record[1])
            require(a.size == 1 and a.dtype.kind in 'fi', 'Original scalar target required')
            labels.append(a.reshape(-1)[0])
        # Same author target FP32 conversion as TRAIN/DEV; retain raw scale.
        y = np.asarray(labels, dtype=np.float32).astype(np.float64)
        require(np.isfinite(y).all() and (np.abs(y) <= 3).all(), 'Original finite target scale')
        self.journal.append({'role': 'test', 'labels_read': True,
                             'purpose': 'once_all_five_joint_fixed_score', 'rows': len(y),
                             'protocol_sha256': protocol_spec['sha256']})
        return y, frozen
