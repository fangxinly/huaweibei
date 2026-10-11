import hashlib,json,math,zipfile
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'work/locked_T0_constant_outer_development_20261006T1324Z';OUT=BASE/'execute'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((BASE/'plan.json').read_text(encoding='utf-8'));rec=json.loads((OUT/'receipt.json').read_text())
assert rec['plan_sha256']==sha(BASE/'plan.json') and rec['source_sha256']==sha(ROOT/'work/locked_T0_constant_outer_development_v1.py')
for p in plan['pins'].values():assert sha(p['path'])==p['sha256']
for name,e in rec['files'].items():assert sha(OUT/name)==e['sha256']
freeze=json.loads((OUT/'prediction_freeze.json').read_text());assert freeze['predictions_sha256']==sha(OUT/'locked_predictions.npz') and freeze['labels_read_before_freeze'] is False
rows=np.load(plan['pins']['outer_rows']['path'],allow_pickle=False);fit=np.load(plan['pins']['fit_rows']['path'],allow_pickle=False);inner=np.load(plan['pins']['inner_rows']['path'],allow_pickle=False)
assert not set(rows)&(set(fit)|set(inner))
with np.load(plan['pins']['outer_predictions']['path'],allow_pickle=False) as orig,np.load(OUT/'scoped_development_values.npz',allow_pickle=False) as z,np.load(OUT/'locked_predictions.npz',allow_pickle=False) as pred:
    assert np.array_equal(rows,z['row_ids']) and np.array_equal(rows,pred['row_ids']) and np.array_equal(rows,orig['row_ids'])
    p=orig['mu'].astype(np.float64);y=z['y'];v=z['video'];c=plan['locked_constant']
    assert np.array_equal(p,z['p_F']) and np.array_equal(pred['predictions'],np.column_stack((p,p+c))) and np.array_equal(pred['predictions'],z['predictions'])
    # Independent exact-row label reconstruction, still no other role decoded.
    with zipfile.ZipFile(plan['pins']['labels']['path']) as pack,pack.open('y.npy') as stream:
        version=np.lib.format.read_magic(stream);reader=np.lib.format.read_array_header_1_0 if version==(1,0) else np.lib.format.read_array_header_2_0
        shape,fortran,dtype=reader(stream);offset=stream.tell();assert shape==(1281,) and not fortran and dtype==np.dtype('<f8')
        check=[]
        for row in rows:stream.seek(offset+int(row)*dtype.itemsize);check.append(np.frombuffer(stream.read(dtype.itemsize),dtype=dtype)[0])
    assert np.array_equal(y,np.asarray(check))
    residual=y-p;q=c*c-2*c*residual
    max_error=float(np.max(abs(q-z['Q'])));assert max_error<1e-12
    groups=np.unique(v);assert len(groups)==18 and len(rows)==433
    byvideo=np.array([np.mean(q[v==g]) for g in groups])
    F=float(np.mean([np.mean((p[v==g]-y[v==g])**2) for g in groups]));C=F+float(byvideo.mean())
    deleted=np.array([(byvideo.sum()-byvideo[i])/17 for i in range(18)])
    passed=bool(byvideo.mean()<0 and np.sum(byvideo<0)>=12 and np.max(deleted)<0)
    result=json.loads((OUT/'results.json').read_text())
    assert abs(F-result['video_equal_F_mse'])<1e-12 and abs(C-result['video_equal_constant_mse'])<1e-12
    assert np.max(abs(deleted-np.asarray(result['delete_one_video_Q'])))<1e-12
    assert passed is result['predeclared_cost_heuristic_passed'] and int(np.sum(byvideo<0))==result['improved_videos']
    for g,old in zip(groups,result['per_video']):assert g==old['video'] and abs(np.mean(q[v==g])-old['Q'])<1e-12
    largest_gain=float(np.max(-byvideo));net_gain=float(-np.sum(byvideo))
    summary={'Q_median':float(np.median(byvideo)),'delete_one_Q_min':float(deleted.min()),'delete_one_Q_max':float(deleted.max()),'largest_positive_gain_as_percent_net_gain':100*largest_gain/net_gain if net_gain>0 else None}
j=json.loads((OUT/'label_access_journal.json').read_text());assert len(j)==1 and j[0]['rows']==rows.tolist()
assert datetime.fromisoformat(freeze['utc'])<datetime.fromisoformat(j[0]['utc'])
audit={'status':'LOCAL_INDEPENDENT_LOCKED_CONSTANT_OUTER_ARRAY_LABEL_ROLE_PAIRED_METRIC_AUDIT_PASSED','actual_utc':datetime.now(timezone.utc).isoformat(),'original_receipt_sha256':sha(OUT/'receipt.json'),'audit_source_sha256':sha(__file__),'max_Q_reconstruction_error':max_error,'summary':summary,'cost_heuristic_passed':passed,'new_GPU_or_fit':False,'other_node_CPU':False,'new_confirmation':False,'label_scope':'Only same exact T0 OUTER433 decoded for independent original-label comparison; no other official row labels decoded.'}
(BASE/'independent_local_audit.json').write_text(json.dumps(audit,indent=2,allow_nan=False)+'\n',encoding='utf-8')
print(json.dumps(audit))
