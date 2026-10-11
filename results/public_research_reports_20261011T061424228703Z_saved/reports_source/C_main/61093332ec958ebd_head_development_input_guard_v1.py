"""New exact201 input-only permission. Never changes the completed232 guard."""
import numpy as np

def allowed_head_inputs(examples,rows,reserved_eval_rows,row_video):
 rows=np.asarray(rows);allowed=np.asarray(reserved_eval_rows)
 if rows.ndim!=1 or rows.dtype.kind not in 'iu' or len(rows)!=201 or not np.array_equal(rows,allowed) or len(np.unique(rows))!=201:
  raise PermissionError('ONLY_EXACT_RESERVED201_INPUT_ORDER')
 if len(examples)!=1281 or len(row_video)!=1281 or len({row_video[int(i)] for i in rows})!=9:
  raise PermissionError('EXACT201_NINE_VIDEO_ROLE')
 return [(examples[int(i)][0],np.zeros((1,1),np.float32),examples[int(i)][2]) for i in rows]

def candidate_gate(pF,pC,rows,videos,restored):
 f,c,b=map(np.asarray,(pF,pC,restored));ids=np.asarray(rows);v=np.asarray(videos)
 if any(x.shape!=(201,) or x.dtype!=np.float32 or not np.isfinite(x).all() for x in (f,c,b)):
  raise ValueError('EXACT201_FINITE_FP32_OUTPUTS')
 if ids.shape!=(201,) or len(np.unique(ids))!=201 or v.shape!=(201,) or len(np.unique(v))!=9:
  raise ValueError('EXACT201_ROW_VIDEO_ALIGNMENT')
 error=float(np.max(np.abs(f.astype(np.float64)-b)))
 if error>1e-6:raise ValueError('DEVELOPMENT_REFERENCE_REPLAY')
 d=c.astype(np.float64)-f;weights=np.asarray([1/(9*np.sum(v==x)) for x in v]);mass=weights*d*d;total=float(mass.sum());vm=np.asarray([mass[v==x].sum() for x in np.unique(v)])
 return {'scope':'HEADEVAL201_NINE_VIDEOS_INPUTS_ONLY_NO_MODEL_SELECTION','exact_nonzero_delta_rows':int(np.sum(d!=0)),'delta_abs_max':float(np.max(np.abs(d))),'video_equal_delta_squared_mass':total,'W_active_videos':int(np.sum(vm>0)),'W_effective_video_mass':None if total==0 else float(total**2/(vm@vm)),'reference_restore_error':error,'endpoint_acc2_class_diff_rows':int(np.sum((f>=0)!=(c>=0))),'endpoint_acc7_class_diff_rows':int(np.sum(np.round(np.clip(f,-3,3))!=np.round(np.clip(c,-3,3)))),'labels_used':False,'boundary_counts_are_not_actual_accuracy_or_gain':True,'diagnostics_do_not_change_heads_readouts_or_rules':True,'no_evaluation_delta_rescue':True}
