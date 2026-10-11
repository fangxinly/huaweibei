"""New inference-only adapter; no mutation of the completed teacher runtime."""
from pathlib import Path
import pickle
import numpy as np


class LabelFreeTeacherInputs:
    def __init__(self, examples, fold_ids, author):
        if len(examples) != 1281:
            raise ValueError('Expected the official 1281 TRAIN rows')
        self.examples = examples
        self.author = author
        self.ids = {name: np.asarray(value, dtype=np.int64) for name, value in fold_ids.items()}
        if set(self.ids) != {'fit', 'inner', 'outer'}:
            raise ValueError('All three original roles must be supplied')
        sets = [set(self.ids[name].tolist()) for name in ('fit', 'inner', 'outer')]
        if any(len(set(self.ids[name].tolist())) != len(self.ids[name]) for name in self.ids):
            raise ValueError('Duplicate role row')
        if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
            raise ValueError('Role overlap')
        if set.union(*sets) != set(range(1281)):
            raise ValueError('Incomplete TRAIN partition')
        self.journal = []

    @classmethod
    def from_pickle(cls, path, fold_ids, author):
        # The container pickle is loaded; only the TRAIN entry is requested.
        # Outcome labels exist in the loaded tuples but index 1 is never read.
        with Path(path).open('rb') as stream:
            examples = pickle.load(stream)['train']
        return cls(examples, fold_ids, author)

    def dataset(self, rows, role):
        rows = np.asarray(rows, dtype=np.int64)
        if role not in ('fit', 'inner'):
            raise PermissionError('Only same-teacher FIT/INNER collection is authorized')
        if rows.ndim != 1 or len(np.unique(rows)) != len(rows):
            raise PermissionError('Invalid requested row array')
        if not set(rows.tolist()) <= set(self.ids[role].tolist()):
            raise PermissionError('Other-role or OUTER row requested')
        samples = [(self.examples[int(row)][0], np.zeros((1, 1), dtype=np.float32),
                    self.examples[int(row)][2]) for row in rows]
        self.journal.append({'role': role, 'rows': rows.tolist(),
                             'outcome_label_index_read': False, 'dummy_label': 0})
        return self.author.get_appropriate_dataset(samples)
