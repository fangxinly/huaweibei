from pathlib import Path
import hashlib,json,numpy as np
d=Path('D:/CodexBackups/selective_flow_20261003_1105/native_radius_cal_mechanism_actual_20261006T072031Z')
old=Path('D:/CodexBackups/selective_flow_20261003_1105/matched_message_pools_actual_v2_20261006T065508Z')
z=np.load(d/'mechanism/cal_candidate_arrays.npz',allow_pickle=False);o=np.load(old/'execute/predictions_frozen.npz',allow_pickle=False);rows=z['row']
assert np.array_equal(np.flatnonzero(o['role']=='calibration'),rows) and len(rows)==418
pred=z['candidate_prediction'].astype(np.float64);delta=pred-pred[:,0,None];oldp=o['candidate_prediction'][rows].astype(np.float64);olddelta=oldp-oldp[:,0,None]
newvalid=z['candidate_valid'];oldvalid=o['candidate_valid'][rows]
summary={}
for name,ids in [('OT',[0,1,2,3,4,5,6,7,8]),('Opm',[0,1,2,3,4,9,10,11,12]),('O',[0,1,2,3,4])]:
    def stat(mask):
        nv=newvalid[mask][:,ids[1:]];ov=oldvalid[mask][:,ids[1:]]
        nd=delta[mask][:,ids[1:]];od=olddelta[mask][:,ids[1:]]
        return {'rows':int(mask.sum()),'old_rows_nonF_legal':int(ov.any(1).sum()),'new_rows_nonF_legal':int(nv.any(1).sum()),
            'old_rows_legal_output_above_replay_resolution':int((ov&(np.abs(od)>1e-6)).any(1).sum()),'new_rows_legal_output_above_replay_resolution':int((nv&(np.abs(nd)>1e-6)).any(1).sum()),
            'new_valid_output_abs_shift_quantiles':np.quantile(np.abs(nd[nv]),[0,.25,.5,.75,1]).tolist()}
    summary[name]={'all':stat(np.ones(418,bool)),'per_fold':{str(k):stat(z['fold']==k) for k in range(3)},'per_video':{str(v):stat(z['video']==v) for v in np.unique(z['video'])}}
proof={'status':'CAL_ONLY_NONF_FINITE_OUTPUT_COVERAGE_ACCOUNTING_COMPLETE','GPU_array_sha256':hashlib.sha256((d/'mechanism/cal_candidate_arrays.npz').read_bytes()).hexdigest(),'labels_read':False,'new_fit':False,'model_forward':False,
    'zero_radius_rows_in_actual_CAL':int((z['normalized_native_radius']==0).sum()),'zero_radius_synthetic_GPU_boundary_tested':False,'scope':'Actual CAL nonzero radii only; general exact-zero fallback is not validated by this finite-message check. 1e-6 is descriptive resolution, not risk margin.',
    'pools':summary}
(d/'nonF_CAL_coverage_analysis.json').write_text(json.dumps(proof,indent=2));print(json.dumps({name:v['all'] for name,v in summary.items()}))
