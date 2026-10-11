"""Separate local NumPy reconstruction via weighted normal equations."""
import argparse, datetime, hashlib, json, math, zipfile
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_labels(path,rows):
    values=[]
    with zipfile.ZipFile(path) as z,z.open('y.npy') as s:
        version=np.lib.format.read_magic(s)
        reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=reader(s);offset=s.tell()
        assert shape==(1281,) and dtype==np.dtype('float64') and not fortran
        for r in rows:
            s.seek(offset+8*int(r));values.append(np.frombuffer(s.read(8),dtype=dtype)[0])
    return np.asarray(values)
def reconstruct(p,y,v,q):
    groups=np.unique(v);w=np.zeros(len(v))
    for group in groups:w[v==group]=1/(len(groups)*np.sum(v==group))
    mean=np.sum(w*p);scale=np.sqrt(np.sum(w*(p-mean)**2));c=np.sum(w*(y-p))
    if scale<=1e-6:a=c;b=0.;corr=np.full(len(q),c)
    else:
        X=np.column_stack([np.ones(len(p)),(p-mean)/scale])
        a,b=np.linalg.solve(X.T@np.diag(w)@X+np.diag([0.,.01]),X.T@np.diag(w)@(y-p))
        corr=a+b*(q-mean)/scale
    return np.column_stack([q,q+c,q+corr])
def check_metrics(pred,p,y,v,report):
    for j,name in enumerate(['zero','constant','affine']):
        error=(pred[:,j]-y)**2;Q=error-(p-y)**2
        qvideo=np.array([np.mean(Q[v==g]) for g in np.unique(v)])
        computed={'pooled_mse':error.mean(),'pooled_mae':abs(pred[:,j]-y).mean(),'video_mse':np.mean([error[v==g].mean() for g in np.unique(v)]),'video_q':qvideo.mean()}
        for key,value in computed.items():assert abs(float(value)-report[name][key])<1e-12
        delete=[np.delete(qvideo,k).mean() for k in range(len(qvideo))]
        stable=bool(qvideo.mean()<0 and sum(qvideo<0)>=math.ceil(2*len(qvideo)/3) and max(delete)<0)
        assert report[name]['stable_heuristic']==stable

a=argparse.ArgumentParser();a.add_argument('--root',required=True);args=a.parse_args();root=Path(args.root)
plan=json.loads((root/'plan.json').read_text(encoding='utf-8'));e=root/'execute';rec=json.loads((e/'receipt.json').read_text(encoding='utf-8'))
assert rec['plan_sha256']==sha(root/'plan.json') and rec['source_sha256']==plan['source_sha256']
for entry in plan['pins'].values():assert sha(entry['path'])==entry['sha256']
with np.load(e/'scoped_inputs.npz',allow_pickle=False) as z: inputs={k:z[k] for k in z.files}
f=inputs['fit_rows'];n=inputs['inner_rows'];pv=inputs['fit_p'];yv=inputs['fit_y'];fv=inputs['fit_video'];p=inputs['inner_p'];y=inputs['inner_y'];v=inputs['inner_video']
assert np.array_equal(f,np.load(plan['pins']['fit_rows']['path'])) and np.array_equal(n,np.load(plan['pins']['inner_rows']['path']))
assert not set(f)&set(n) and len(set(v))==4
assert np.array_equal(yv,read_labels(plan['pins']['label_container']['path'],f)) and np.array_equal(y,read_labels(plan['pins']['label_container']['path'],n))
for key,rolep in [('fit_predictions',pv),('inner_predictions',p)]:
    with np.load(plan['pins'][key]['path']) as z:assert np.array_equal(z['mu'].astype(np.float64),rolep)
A=reconstruct(pv,yv,fv,p);B=np.empty_like(A);excluded=[]
for group in np.unique(v):
    train=v!=group;held=~train;B[held]=reconstruct(p[train],y[train],v[train],p[held]);excluded.append(dict(video=str(group),train_rows=n[train].tolist(),held_rows=n[held].tolist()))
errors={}
for arm,expected in [('A',A),('B',B)]:
    with np.load(e/(arm+'_frozen_predictions.npz')) as z:
        actual=z['pred'];assert np.array_equal(z['row_ids'],n);errors[arm]=float(abs(expected-actual).max());assert errors[arm]<1e-12
    freeze=json.loads((e/(arm+'_prediction_freeze.json')).read_text(encoding='utf-8'));assert freeze['predictions_sha256']==sha(e/(arm+'_frozen_predictions.npz'))
result=json.loads((e/'results.json').read_text(encoding='utf-8'));check_metrics(A,p,y,v,result['A']);check_metrics(B,p,y,v,result['B'])
journal=json.loads((e/'label_access_journal.json').read_text(encoding='utf-8'));afreeze=json.loads((e/'A_prediction_freeze.json').read_text(encoding='utf-8'))
assert afreeze['utc']<journal[1]['utc'] and journal[0]['role']=='fit' and journal[1]['role']=='inner'
for access in journal:
    assert set(access['rows'])<=set(f if access['role']=='fit' else n)
    if access['purpose'].startswith('B_fit_excluding_video_'):
        excluded_group=access['purpose'].split('B_fit_excluding_video_')[1];assert not set(access['rows'])&set(n[v==excluded_group])
assert rec['seconds']<=plan['maximum_seconds'] and rec['peak_process_working_set_bytes']<=plan['maximum_peak_working_set_bytes']
proof=dict(status='SEPARATE_LOCAL_CPU_NORMAL_EQUATION_AND_SCOPED_ARRAY_AUDIT_PASSED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),plan_sha256=sha(root/'plan.json'),A_B_prediction_max_errors=errors,all_metrics_rebuilt=True,original_predictions_and_allowed_labels_exact=True,A_frozen_before_INNER_label_access=True,B_held_video_exclusion=excluded,new_GPU_forward=False,other_node_CPU_execution=False,teacher_selection_reuse_not_repaired=True)
(root/'independent_local_array_audit.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'status':proof['status'],'max_errors':errors}))
