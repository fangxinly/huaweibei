"""Label-free video split and strict role access for the requested pooled 5-fold CV."""
import hashlib
import json
import re
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def video_id(row_id):
    match=re.fullmatch(r'(.+)\[(\d+)\]',row_id)
    if not match:raise ValueError('Unrecognized segment ID: '+row_id)
    return match.group(1)


def tie(seed, purpose, video):
    return hashlib.sha256(f'{seed}|{purpose}|{video}'.encode()).hexdigest()


def make_folds(ids, seed=128):
    ids=list(ids)
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate segment IDs')
    groups={}
    for row_id in ids:groups.setdefault(video_id(row_id),[]).append(row_id)
    if len(groups)<15:raise ValueError('Insufficient videos for grouped inner and outer roles')
    bins=[[] for _ in range(5)];sizes=[0]*5
    ordered=sorted(groups,key=lambda v:(-len(groups[v]),tie(seed,'outer',v)))
    for v in ordered:
        i=min(range(5),key=lambda i:(sizes[i],len(bins[i]),i))
        bins[i].append(v);sizes[i]+=len(groups[v])
    folds=[]
    for i,outer in enumerate(bins):
        rest=set(groups)-set(outer)
        # Hash-order prefix nearest 15% of non-outer rows. No label stratification.
        inner_order=sorted(rest,key=lambda v:tie(seed,f'inner{i}',v))
        target=.15*sum(len(groups[v]) for v in rest)
        prefix_sizes=[];size=0
        for v in inner_order[:-1]:size+=len(groups[v]);prefix_sizes.append(size)
        stop=min(range(len(prefix_sizes)),key=lambda k:(abs(prefix_sizes[k]-target),k))+1
        inner=set(inner_order[:stop]);fit=rest-inner
        row_roles={role:[s for s in ids if video_id(s) in vs]
                   for role,vs in [('fit',fit),('inner',inner),('outer',set(outer))]}
        folds.append({'fold':i,'row_ids':row_roles,
                      'video_ids':{r:sorted({video_id(s) for s in ss}) for r,ss in row_roles.items()},
                      'rows':{r:len(ss) for r,ss in row_roles.items()}})
    validate_folds(ids,folds)
    return folds


def validate_folds(ids,folds):
    ids=list(ids)
    if len(folds)!=5 or len(ids)!=len(set(ids)):raise ValueError('Invalid CV inventory')
    seen=[]
    for i,f in enumerate(folds):
        if f['fold']!=i:raise ValueError('Fold order differs')
        roles=f['row_ids']
        if set(roles)!={'fit','inner','outer'}:raise ValueError('Role inventory differs')
        union=[];videos=[]
        for role in ('fit','inner','outer'):
            ss=roles[role]
            if not ss or len(ss)!=len(set(ss)):raise ValueError('Empty or duplicate role')
            if ss!=[s for s in ids if s in set(ss)]:raise ValueError('Canonical row order differs')
            vs={video_id(s) for s in ss};union.extend(ss);videos.append(vs)
            if f['video_ids'][role]!=sorted(vs) or f['rows'][role]!=len(ss):
                raise ValueError('Declared role identity differs from actual IDs')
        if len(union)!=len(ids) or set(union)!=set(ids):raise ValueError('Incomplete role partition')
        if any(videos[a]&videos[b] for a,b in ((0,1),(0,2),(1,2))):
            raise PermissionError('Video leakage between fit/inner/outer')
        seen.extend(roles['outer'])
    if len(seen)!=len(ids) or len(seen)!=len(set(seen)) or set(seen)!=set(ids):
        raise ValueError('Outer folds must cover every row exactly once')


class FoldGuard:
    def __init__(self, records, fold):
        # Index inputs/ID only. Materializing the trusted original pickle is disclosed.
        self.by_id={}
        for record in records:
            s=record[2].decode() if isinstance(record[2],bytes) else record[2]
            if s in self.by_id:raise ValueError('Duplicate physical record')
            self.by_id[s]=record
        self.fold=fold;self.journal=[]
        if set(self.by_id)!=set(sum(fold['row_ids'].values(),[])):
            raise ValueError('Physical inventory differs from frozen split')

    def inputs(self, role, dummy=0):
        import numpy as np
        if role not in ('fit','inner','outer'):raise ValueError('Unapproved role')
        ss=self.fold['row_ids'][role]
        result=[(self.by_id[s][0],np.full((1,1),dummy,dtype=np.float32),s) for s in ss]
        self.journal.append({'role':role,'purpose':'inputs','labels_read':False,'rows':len(ss)})
        return result

    def supervision(self, role):
        if role!='fit':raise PermissionError('Only FIT may provide training supervision')
        ss=self.fold['row_ids']['fit']
        self.journal.append({'role':'fit','purpose':'training','labels_read':True,'rows':len(ss)})
        return [self.by_id[s] for s in ss]

    def inner_labels(self, path, digest, state_sha):
        import numpy as np
        if sha(path)!=digest:raise PermissionError('INNER prediction not frozen')
        with np.load(path,allow_pickle=False) as z:
            if set(z.files)!={'row_ids','prediction','model_state_sha256'}:
                raise PermissionError('INNER schema differs')
            if z['row_ids'].tolist()!=self.fold['row_ids']['inner'] or str(z['model_state_sha256'].item())!=state_sha:
                raise PermissionError('INNER row/state identity differs')
            if z['prediction'].shape!=(self.fold['rows']['inner'],) or not np.isfinite(z['prediction']).all():
                raise ValueError('Invalid INNER predictions')
        ss=self.fold['row_ids']['inner']
        labels=np.asarray([np.asarray(self.by_id[s][1]).reshape(-1)[0] for s in ss],dtype=np.float32).astype(np.float64)
        if not np.isfinite(labels).all() or (np.abs(labels)>3).any():raise ValueError('Invalid INNER targets')
        self.journal.append({'role':'inner','purpose':'strict_earliest_checkpoint_selection',
                             'labels_read':True,'prediction_sha256':digest,'rows':len(ss)})
        return labels

    def outer_labels(self, *args, **kwargs):
        raise PermissionError('Training/runtime cannot read OUTER labels; separate all-ten saved-prediction scoring required')


def batches(order,batch=32):
    order=list(order)
    if len(order)!=len(set(order)) or set(order)!=set(range(len(order))):
        raise ValueError('Invalid complete positional permutation')
    return [order[i:i+batch] for i in range(0,len(order),batch)]


def inner_mse(prediction, labels):
    import numpy as np
    p=np.asarray(prediction,dtype=np.float64);y=np.asarray(labels,dtype=np.float64)
    if p.shape!=y.shape or p.ndim!=1 or not len(p) or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid INNER arrays')
    return float(np.square(p-y).mean())
