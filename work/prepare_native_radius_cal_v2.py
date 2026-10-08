from pathlib import Path
import ast,datetime,hashlib,json,shutil,numpy as np
cwd=Path(__file__).parent.parent;work=cwd/'work';old=Path('D:/CodexBackups/selective_flow_20261003_1105/matched_message_pools_actual_v2_20261006T065508Z')
utc=datetime.datetime.now(datetime.timezone.utc);stamp=utc.strftime('%Y%m%dT%H%M%SZ')
d=old.parent/('native_radius_cal_mechanism_actual_'+stamp);d.mkdir(exist_ok=False)
root='/data/coding/group_teacher_v1_20261005T1650Z/native_radius_cal_mechanism_v1_'+stamp
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();plan=json.loads((old/'plan.json').read_text());z=np.load(old/'execute/predictions_frozen.npz',allow_pickle=False)
rows=np.flatnonzero(z['role']=='calibration');assert len(rows)==418 and 454 in rows and 620 in rows
np.save(d/'authorized_cal_rows.npy',rows)
names=['diagnose_native_radius_cal_v1.py','audit_native_radius_cal_v1.py','run_native_radius_cal_v1.py','run_same_teacher_capture_v1.py']
for name in names:ast.parse((work/name).read_text(encoding='utf-8'));shutil.copy2(work/name,d/name)
plan.update(status='FROZEN_CAL_ONLY_NATIVE_RADIUS_MECHANISM_V1',new_root=root,frozen_utc=utc.isoformat(),source_sha256=sha(d/names[0]),prior_predictions='/data/coding/group_teacher_v1_20261005T1650Z/matched_message_pools_v2_20261006T065508Z/execute/predictions_frozen.npz',expected_model_state='13f0d54e10ceafc9d676e03adfc2298aea8b05d11db4aec6ca6e4b405dfbed8b',maximum_seconds=1200,maximum_peak_allocated_bytes=1073741824,maximum_new_array_bytes=134217728,
    scope='CAL418/18 only unlabeled real-message mechanical coverage. No EVAL terminal forwards or metrics, no new selector/fit/training, no rerun of saved paths.',
    radius='common norm of saved old native raw-message displacement in TRAIN scale metric; old amplitude1 exact reconstructed native message, same1/8,1/4,1/2,1 for three directions.',
    candidate_order='F; old native displacement four amplitudes; saved teacher endpoint unit four amplitudes at native radius; negative native displacement four amplitudes.',
    stopping='No new100 or adapted EVAL confirmation score. Feasibility only; preserve original negative EVAL.',
    EVAL_metrics=False,CAL_labels_read=False,selected_predictions=False,original_input_full_model_replay=False)
for key in ['main_comparison','directions','acceptors','eval','eval_metrics','wrapper_revision','previous_precheck']:plan.pop(key,None)
plan['pinned_files'][plan['prior_predictions']]=sha(old/'execute/predictions_frozen.npz')
for name in names+['authorized_cal_rows.npy']:plan['pinned_files'][root+'/'+name]=sha(d/name)
(d/'plan.json').write_text(json.dumps(plan,indent=2))
# A meaningful unlabeled fixture: reconstruct old amp1 and verify shared radius/trust before GPU.
f=z['original_messages'][rows];sc=z['message_scale'][rows];native=(f/sc+z['old_shift'][rows])*sc
shift=(native-f)/sc;rad=np.linalg.norm(shift.reshape(418,-1),axis=1)
assert np.isfinite(native).all() and np.max(rad/np.linalg.norm((f/sc).reshape(418,-1),axis=1))<.25001
roundtrip=(f+(native-f)).astype(np.float64);native64=native.astype(np.float64)
err=np.abs(roundtrip-native64);bound=4*np.finfo(np.float32).eps*(np.abs(f)+np.abs(native)+1e-6)
assert np.all(err<=bound)
assert 'native_message if amp==1.' in (d/names[0]).read_text(encoding='utf-8')
proof={'status':'AST_AND_CAL_ONLY_NATIVE_RADIUS_NUMPY_FIXTURE_PASSED','rows':418,'labels_read':False,'prediction_fit':False,'GPU_executed':False,'files':{name:sha(d/name) for name in names+['plan.json','authorized_cal_rows.npy']},'source_files_prior_unmodified':sha(old/'diagnose_matched_message_pools_v1.py')==plan['pinned_files']['/data/coding/group_teacher_v1_20261005T1650Z/matched_message_pools_v2_20261006T065508Z/diagnose_matched_message_pools_v1.py'],'free_D':shutil.disk_usage(d).free,'free_C':shutil.disk_usage(cwd).free}
proof['old_amp1_scientific_source_exact_native_branch']=True
proof['v1_local_fixture_float32_cancellation_refused_and_retained']=True
proof['float32_cancel_roundtrip_max_error']=float(err.max())
assert proof['free_D']>1073741824 and proof['free_C']>1073741824
(d/'local_freeze.json').write_text(json.dumps(proof,indent=2));(cwd/'outputs/native_radius_current_root.json').write_text(json.dumps({'D':d.as_posix(),'remote':root,'stamp':stamp},indent=2))
print(json.dumps({'D':d.as_posix(),'remote':root,'stamp':stamp,'source_sha256':plan['source_sha256']}))
