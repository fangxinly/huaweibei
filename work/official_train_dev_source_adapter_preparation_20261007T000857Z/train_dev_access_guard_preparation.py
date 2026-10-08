"""Preparation only. Caller supplies an approved split; this module cannot open datasets."""
ALLOWED_ROLES = frozenset(('train', 'dev'))
EXPECTED_ROWS = {'train': 1281, 'dev': 229}

def approved_split(mapping, role, row_ids):
    # Reject before invoking even one operation on a potentially sensitive mapping.
    if role not in ALLOWED_ROLES:
        raise PermissionError('Only train and dev are approved during preparation/training')
    rows = mapping[role]
    ids = tuple(row_ids)
    if len(rows) != EXPECTED_ROWS[role] or len(ids) != len(rows):
        raise ValueError('Official row count or ID alignment mismatch')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate row IDs')
    return rows, ids

def require_disjoint_ids(train_ids, dev_ids):
    if set(train_ids).intersection(dev_ids):
        raise ValueError('Train and dev IDs overlap')
