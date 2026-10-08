"""Approved official TRAIN/DEV roles; labels become available only at explicit gates."""
import hashlib
from pathlib import Path
import numpy as np

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def canonical_id(value):
    if isinstance(value,bytes):return value.decode('utf-8')
    if isinstance(value,str):return value
    raise ValueError('Official segment ID must be a string or UTF8 bytes')

class FullTrainDevGuard:
    def __init__(self,train_records,dev_records):
        if len(train_records)!=1281 or len(dev_records)!=229:
            raise ValueError('Official TRAIN/DEV counts mismatch')
        self._records={'train':train_records,'dev':dev_records}
        self.ids={r:tuple(canonical_id(x[2]) for x in records) for r,records in self._records.items()}
        if any(len(set(ids))!=len(ids) for ids in self.ids.values()) or set(self.ids['train'])&set(self.ids['dev']):
            raise ValueError('Duplicate or overlapping official IDs')
        self.journal=[]

    def inputs_only(self,role,dummy=0):
        if role not in ('train','dev'):
            raise PermissionError('Unapproved role')
        records=[(r[0],np.full((1,1),dummy,dtype=np.float32),r[2]) for r in self._records[role]]
        self.journal.append({'role':role,'purpose':'inputs_only','labels_read':False,'rows':len(records)})
        return records

    def train_supervision(self):
        self.journal.append({'role':'train','purpose':'fullTRAIN_fitting','labels_read':True,'rows':1281})
        return list(self._records['train'])

    def dev_labels_after_frozen_prediction(self,path,expected_sha,state_sha):
        if len(expected_sha)!=64 or file_sha(path)!=expected_sha:
            raise PermissionError('DEV physical prediction SHA not frozen')
        if len(state_sha)!=64 or any(c not in '0123456789abcdef' for c in state_sha):
            raise PermissionError('Model state SHA invalid')
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'row_ids','prediction','model_state_sha256'}:
                raise PermissionError('DEV prediction schema mismatch')
            if tuple(z['row_ids'].tolist())!=self.ids['dev'] or str(z['model_state_sha256'].item())!=state_sha:
                raise PermissionError('DEV row/state identity mismatch')
            if z['prediction'].shape!=(229,) or not np.isfinite(z['prediction']).all():
                raise PermissionError('DEV prediction finite shape mismatch')
        values=np.asarray([np.asarray(r[1]).reshape(-1)[0] for r in self._records['dev']],dtype=np.float64)
        if not np.isfinite(values).all():raise ValueError('Nonfinite DEV target')
        self.journal.append({'role':'dev','purpose':'checkpoint_selection_only','labels_read':True,
                             'prediction_sha256':expected_sha,'model_state_sha256':state_sha,'rows':229})
        return values
