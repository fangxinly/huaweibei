"""Input and supervision roles for a new fold-0 reference; no Torch dependency.

Trusted pickle deserialization is a container operation, not a guarantee that
other-split bytes do not exist in memory. Only its TRAIN entry may be indexed.
OUTER inference remains disabled in this preparation: formal completion and
strict disk replay authorization are not implemented here.
"""
import hashlib
from pathlib import Path
import numpy as np

def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def checked_rows(rows):
    x = np.asarray(rows)
    if x.ndim != 1 or x.dtype.kind not in 'iu' or x.dtype.kind == 'b':
        raise PermissionError('ROWS_MUST_BE_ONE_DIMENSIONAL_INTEGERS')
    if len(x) == 0 or np.any(x < 0) or np.any(x >= 1281) or len(set(x.tolist())) != len(x):
        raise PermissionError('ROW_RANGE_OR_DUPLICATE_GUARD')
    return x.astype(np.int64, copy=True)

class FoldZeroGuard:
    def __init__(self, train_examples, ids, row_video_ids):
        if len(train_examples) != 1281 or len(row_video_ids) != 1281:
            raise ValueError('OFFICIAL_TRAIN_1281_REQUIRED')
        self._examples = train_examples
        self.ids = {role: checked_rows(ids[role]) for role in ('fit', 'inner', 'outer')}
        if tuple(len(self.ids[r]) for r in ('fit', 'inner', 'outer')) != (695, 153, 433):
            raise ValueError('FROZEN_FOLD_ZERO_COUNTS_DIFFER')
        self._allowed = {role: set(rows.tolist()) for role, rows in self.ids.items()}
        videos = {r: {str(row_video_ids[i]) for i in rows} for r, rows in self.ids.items()}
        if tuple(len(videos[r]) for r in ('fit', 'inner', 'outer')) != (30, 4, 18):
            raise ValueError('FROZEN_FOLD_ZERO_VIDEO_COUNTS_DIFFER')
        for a, b in (('fit', 'inner'), ('fit', 'outer'), ('inner', 'outer')):
            if self._allowed[a] & self._allowed[b] or videos[a] & videos[b]:
                raise ValueError('ROW_OR_VIDEO_ISOLATION_VIOLATION')
        self.journal = []

    def _rows(self, rows, role):
        if role not in ('fit', 'inner'):
            raise PermissionError('OUTER_DEV_TEST_INPUTS_DISABLED_IN_PRECHECK_CANDIDATE')
        rows = checked_rows(rows)
        if not set(rows.tolist()) <= self._allowed[role]:
            raise PermissionError('ROLE_ROW_MEMBERSHIP_VIOLATION')
        return rows

    def inputs_only(self, rows, role):
        rows = self._rows(rows, role)
        # Deliberately never evaluate example[1], including INNER labels.
        records = [(self._examples[i][0], np.zeros((1, 1), dtype=np.float32),
                    self._examples[i][2]) for i in rows]
        self.journal.append({'role': role, 'rows': rows.tolist(), 'labels_read': False})
        return records

    def fit_supervision(self, rows):
        rows = self._rows(rows, 'fit')
        records = [self._examples[i] for i in rows]
        self.journal.append({'role': 'fit', 'rows': rows.tolist(), 'labels_read': True,
                             'purpose': 'fit_task_and_unimodal_objectives_only'})
        return records

    def inner_labels_after_frozen_predictions(self, path, expected_file_sha):
        # Chronological selection gate, not authorization to claim new validation.
        if len(expected_file_sha) != 64 or file_sha(path) != expected_file_sha:
            raise PermissionError('INNER_PREDICTIONS_NOT_FROZEN_TO_EXPECTED_SHA')
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {'row_ids', 'prediction', 'model_state_sha256'}:
                raise PermissionError('INNER_PREDICTION_SCHEMA')
            if not np.array_equal(z['row_ids'], self.ids['inner']):
                raise PermissionError('INNER_PREDICTION_ROW_ORDER')
            if z['prediction'].shape != (153,) or not np.isfinite(z['prediction']).all():
                raise PermissionError('INNER_PREDICTION_SHAPE_OR_FINITE')
            state_sha = str(z['model_state_sha256'].item())
            if len(state_sha) != 64 or any(c not in '0123456789abcdef' for c in state_sha):
                raise PermissionError('MODEL_STATE_SHA_FORMAT')
        self.journal.append({'role': 'inner', 'rows': self.ids['inner'].tolist(),
                             'labels_read': True, 'purpose': 'checkpoint_selection_only',
                             'prediction_file_sha256': expected_file_sha,
                             'model_state_sha256': state_sha})
        return np.asarray([np.asarray(self._examples[i][1]).reshape(-1)[0]
                           for i in self.ids['inner']], dtype=np.float64)
