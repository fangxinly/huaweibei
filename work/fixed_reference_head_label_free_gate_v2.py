"""Preparation only: no Torch, task labels, GPU or head fitting.

Diagnostics are about a single predeclared candidate, not measured benefit.
"""
import numpy as np

def allowed_head_inputs(examples, rows, reserved_fit_rows, row_video):
    rows=np.asarray(rows)
    allowed=np.asarray(reserved_fit_rows)
    if rows.ndim!=1 or rows.dtype.kind not in 'iu' or len(rows)==0:
        raise PermissionError('INTEGER_ROWS_ONLY')
    if len(set(rows.tolist()))!=len(rows) or not set(rows.tolist())<=set(allowed.tolist()):
        raise PermissionError('ONLY_RESERVED_HEAD_FIT_INPUTS')
    if len(allowed)!=232 or len(set(row_video[int(i)] for i in allowed))!=9:
        raise PermissionError('FROZEN_232_NINE_VIDEO_ROLE')
    # Never evaluate example[1]. Discard supplied task label access entirely.
    return [(examples[int(i)][0],np.zeros((1,1),np.float32),examples[int(i)][2]) for i in rows]

def candidate_gate(pF,pC,rows,videos,restored):
    arrays=[np.asarray(x) for x in (pF,pC,restored)]
    rows=np.asarray(rows);videos=np.asarray(videos)
    if any(x.shape!=(232,) or x.dtype!=np.float32 or not np.isfinite(x).all() for x in arrays):
        raise ValueError('EXACT_FINITE_FLOAT32_SCALAR_ARRAYS')
    if rows.shape!=(232,) or len(set(rows.tolist()))!=232 or videos.shape!=(232,) or len(np.unique(videos))!=9:
        raise ValueError('FROZEN_FIT_ROLE_ARRAYS')
    replay=float(np.max(np.abs(arrays[0].astype(np.float64)-arrays[2])))
    if replay>1e-6:raise ValueError('RESTORED_REFERENCE_REPLAY')
    f,c=arrays[:2];delta=c.astype(np.float64)-f.astype(np.float64)
    weights=np.zeros(232,np.float64)
    for v in np.unique(videos):weights[videos==v]=1/(9*np.sum(videos==v))
    mass=weights*delta**2;total=float(mass.sum())
    vmass=np.asarray([mass[videos==v].sum() for v in np.unique(videos)],np.float64)
    active_videos=int(np.sum(vmass>0))
    ess=None if total==0 else float(total**2/np.sum(vmass**2))
    # Empirical feasibility stop only. No calibration/safety or iid claim.
    eligible=total>0
    reasons=[]
    if total==0:reasons.append('EXACT_ZERO_DELTA_NO_W_SIGNAL')
    lo=np.minimum(f,c);hi=np.maximum(f,c)
    acc2_possible=(f>=0)!=(c>=0)
    acc7_possible=np.round(np.clip(f,-3,3))!=np.round(np.clip(c,-3,3))
    boundaries=np.arange(-2.5,3,.5)[::2]
    endpoint_ties=int(np.sum(np.any((f[:,None]==boundaries)|(c[:,None]==boundaries),axis=1)))
    ulp=np.maximum(np.abs(np.nextafter(f,np.float32(np.inf)).astype(np.float64)-f),np.abs(f.astype(np.float64)-np.nextafter(f,np.float32(-np.inf))))
    return {'scope':'HEAD_FIT232_NINE_VIDEOS_LABEL_FREE_ONLY','exact_nonzero_delta_rows':int(np.sum(delta!=0)),
      'delta_abs_max':float(np.max(np.abs(delta))),'video_equal_abs_delta_mean':float(np.sum(weights*np.abs(delta))),
      'video_equal_delta_squared_mass':total,'W_active_videos':active_videos,'W_effective_video_mass':ess,
      'W_empirical_feasibility_pass':eligible,'stop_reasons':reasons,'delta_at_least_one_reference_float32_ULP_rows':int(np.sum(np.abs(delta)>=ulp)),
      'reference_restore_error':replay,'endpoint_acc2_class_diff_rows':int(np.sum(acc2_possible)),
      'endpoint_acc7_class_diff_rows':int(np.sum(acc7_possible)),'half_integer_endpoint_tie_rows':endpoint_ties,
      'decision_boundary_interval_endpoint_range':{'min':float(lo.min()),'max':float(hi.max())},
      'boundary_counts_are_not_actual_accuracy_or_gain':True,'labels_used':False,'epsilon_or_amplitude_rescue':False}
