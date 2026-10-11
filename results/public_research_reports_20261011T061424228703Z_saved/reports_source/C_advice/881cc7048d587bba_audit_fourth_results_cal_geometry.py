"""Review published metrics and CAL unlabelled saved geometry. No model or raw y."""
from pathlib import Path
import hashlib
import json
import numpy as np

root=Path(__file__).resolve().parents[1]
evidence=Path('D:/CodexBackups/selective_flow_20261003_1105/matched_message_pools_actual_v2_20261006T065508Z')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
prediction=evidence/'execute/predictions_frozen.npz'
assert sha(prediction)=='b6c23484a9fff5d444fa33605f3d8eeee82942411e5fe93cb97f9c1bee7c0c9f'
gpu=json.loads((evidence/'execute/receipt.json').read_text(encoding='utf-8'))
evaluation=json.loads((evidence/'evaluation/receipt.json').read_text(encoding='utf-8'))
joint=json.loads((evidence/'joint_preservation_audit.json').read_text(encoding='utf-8'))
assert gpu['prediction_sha256']==evaluation['prediction_sha256']==joint['original_GPU_prediction_sha256']==sha(prediction)
assert gpu['model_state_before']==gpu['model_state_after']
with np.load(prediction,allow_pickle=False) as z:
    members=list(z.files); assert 'y' not in members
    roles=z['role']; role_names=[str(s) for s in np.unique(roles)]
    assert set(role_names)=={'calibration','evaluation'}
    cal=(roles=='calibration')
    assert int(cal.sum())==418, role_names
    so=z['old_shift'][cal].astype(np.float64)
    radius=z['normalized_radius'][cal].astype(np.float64)
    rn=np.linalg.norm(so.reshape(418,-1),axis=1)
    delta=z['native_prediction'][cal].astype(np.float64)-z['pf'][cal].astype(np.float64)
    cap=z['cap'][cal].astype(np.float64)
    native_legal=np.abs(delta)<=cap
    old_valid=z['candidate_valid'][cal][:,[0,1,2,3,4]]
    summary=dict(CAL_rows=418,CAL_videos=len(np.unique(z['video'][cal])),
        zero_native_radii=int(np.sum(rn==0)),
        native_radius_to_previous_radius_quantiles=np.quantile(rn/radius,[0,.25,.5,.75,1]).tolist(),
        native_radius_smaller_than_old_grid_minimum_rows=int(np.sum(rn<.125*radius)),
        native_output_within_cap_rows=int(native_legal.sum()),
        native_output_within_cap_with_replay_margin_rows=int(np.sum(np.abs(delta)+1e-6<=cap)),
        native_output_within_cap_with_two_replay_margins_rows=int(np.sum(np.abs(delta)+2e-6<=cap)),
        native_output_changes_above_replay_tolerance_rows=int(np.sum(np.abs(delta)>1e-6)),
        old_pool_rows_with_nonF_cap_valid_nominal=int(np.sum(old_valid[:,1:].any(axis=1))))
metrics={name:{k:v for k,v in values.items() if k!='per_video'}
         for name,values in evaluation['metrics'].items()}
selected_names=['F','native','old_OT','cal_OT','cal_Opm','oracle_OT','oracle_Opm']
out=dict(scope='Published EVAL receipt aggregates and raw CAL geometry/predictions only; no raw outcome labels, new GPU or fitting',
         prediction_sha256=sha(prediction),members=members,role_names=role_names,
         raw_y_read=False,new_eval_metrics_computed=False,new_CAL_fitted=False,
         CAL_unlabelled_geometry=summary,published_metrics={name:metrics[name] for name in selected_names},
         cal_OT_minus_F=metrics['cal_OT']['mse']-metrics['F']['mse'],
         cal_OT_minus_native=metrics['cal_OT']['mse']-metrics['native']['mse'],
         oracle_OT_minus_oracle_Opm=metrics['oracle_OT']['mse']-metrics['oracle_Opm']['mse'])
(root/'work/fourth_results_cal_geometry_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
