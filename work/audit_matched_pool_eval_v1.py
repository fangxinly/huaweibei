from pathlib import Path
import argparse,hashlib,json,numpy as np
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
report=read(r/'evaluation/receipt.json');gate=read(r/'evaluation/label_access_gate.json');phase=read(r/'execute/receipt.json')
assert report['prediction_sha256']==gate['all_eight_predictions_sha256']==sha(r/'execute/predictions_frozen.npz')==phase['prediction_sha256']
assert gate['EVAL_values_not_yet_decoded'] and not report['CAL_refit'] and not report['DEV_read'] and not report['TEST_read']
assert report['eval_arrays_sha256']==sha(r/'evaluation/eval_arrays.npz')
z=np.load(r/'evaluation/eval_arrays.npz',allow_pickle=False);roles=np.load(r/'roles.npz',allow_pickle=False)
ids=np.flatnonzero(roles['role']=='evaluation');assert np.array_equal(ids,z['row']) and len(ids)==863 and len(np.unique(z['video']))==34
source=np.load(r/'execute/predictions_frozen.npz',allow_pickle=False)
for k in ['fold','video','pf','selected_index','candidate_valid']:assert np.array_equal(z[k],source[k][ids])
assert np.array_equal(z['native'],source['native_prediction'][ids])
assert np.array_equal(z['selected_prediction'],source['selected_prediction'][ids])
y=z['y'];pf=z['pf'];native=z['native'];video=z['video'];unique=np.unique(video)
preds={'F':pf,'native':native};names=['old_O','old_T','old_OT','old_Opm','cal_O','cal_T','cal_OT','cal_Opm']
for i,n in enumerate(names):preds[n]=z['selected_prediction'][:,i]
for i,n in enumerate(['oracle_O','oracle_T','oracle_OT','oracle_Opm']):preds[n]=z['oracle_prediction'][:,i]
for n,pred in preds.items():
    e=pred-y;q=2*(pf-y)*(pred-pf)+(pred-pf)**2;eq=e**2;rr=report['metrics'][n]
    assert abs(float(eq.mean())-rr['mse'])<1e-12 and abs(float(np.abs(e).mean())-rr['mae'])<1e-12
    assert abs(q.mean()-rr['mean_risk_change_F'])<1e-12
    assert abs(float(np.mean([eq[video==v].mean() for v in unique]))-rr['video_equal_mse'])<1e-12
    for item in rr['per_video']:
        mask=video==item['video'];assert item['rows']==int(mask.sum()) and abs(item['mse']-eq[mask].mean())<1e-12
pools=[[0,1,2,3,4],[0,5,6,7,8],[0,1,2,3,4,5,6,7,8],[0,1,2,3,4,9,10,11,12]]
for i,ii in enumerate(pools):
    error=(z['candidate_prediction'][:,ii]-y[:,None])**2;error[~z['candidate_valid'][:,ii]]=np.inf
    idx=np.asarray(ii)[error.argmin(1)];assert np.array_equal(idx,z['oracle_index'][:,i])
    assert np.array_equal(z['candidate_prediction'][np.arange(len(ids)),idx],z['oracle_prediction'][:,i])
assert np.all(z['selected_index'][z['fold']==0,4:]==0)
proof=dict(status='EVAL_FIXED_ROLE_AND_RISK_METRICS_INDEPENDENTLY_RECONSTRUCTED',rows=863,videos=34,prediction_sha256=phase['prediction_sha256'],metric_arrays_sha256=report['eval_arrays_sha256'],metric_report_sha256=sha(r/'evaluation/receipt.json'),post_freeze_oracle_only=True,no_parameter_selection=True,no_model_forward=True)
a.out.write_text(json.dumps(proof,indent=2));print(json.dumps(proof))
