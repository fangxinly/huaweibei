from pathlib import Path
import ast,datetime,hashlib,json,shutil,numpy as np
w=Path(__file__).parent;cwd=w.parent
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('matched_message_pools_actual_'+stamp)
assert shutil.disk_usage('D:/').free>2*1024**3 and shutil.disk_usage('C:/').free>1024**3
dest.mkdir();remote='/data/coding/group_teacher_v1_20261005T1650Z/matched_message_pools_v1_'+stamp
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=Path('D:/CodexBackups/selective_flow_20261003_1105/scalar_geometry_actual_20261006T060719Z')
cal=Path('D:/CodexBackups/selective_flow_20261003_1105/cal_video_equal_signal_actual_20261006T063535Z')
assert json.loads((cal/'joint_preservation_audit.json').read_text())['status']=='CAL_SIGNAL_D_OTHER_NODE_CPU_AND_ACTUAL_CAPTURE_JOINTLY_VERIFIED'
r=json.loads((cal/'execute/calibration_result.json').read_text())
params=dict(lambda_by_fold=[f['lambda_video_equal'] for f in r['folds']],cap_by_fold=[f['output_cap_old_native_abs_shift_quantile95'] for f in r['folds']],
    original_result_sha256=sha(cal/'execute/calibration_result.json'),role_plan_sha256='785c27b7bfb53796e3ee5d6cc9737009c85a29c1124a6633538f0e526722f0cc',no_new_label_fit=True)
(dest/'cal_parameters.json').write_text(json.dumps(params,indent=2))
shutil.copy2(cal/'row_fold_calibration_roles.npz',dest/'roles.npz');shutil.copy2(old/'mu_input.npz',dest/'mu_input.npz')
files=['diagnose_matched_message_pools_v1.py','audit_matched_message_pools_v1.py','run_matched_message_pools_phase_v1.py','run_same_teacher_capture_v1.py']
for name in files:
    ast.parse((w/name).read_text());shutil.copy2(w/name,dest/name)
base=json.loads((old/'plan.json').read_text());pins={k:v for k,v in base['pinned_files'].items() if '/scalar_geometry_diagnostic_v1_' not in k}
teacher='/data/coding/group_teacher_v1_20261005T1650Z/oof_utility_diagnostic_v1_20261006T033054Z/execute/predictions_frozen.npz'
pins[teacher]='068b0dca18dcaa571f687e262b1b2438983d132374b89b82240abdb9689b6f0f'
for name in files+['roles.npz','mu_input.npz','cal_parameters.json']:pins[remote+'/'+name]=sha(dest/name)
plan={k:base[k] for k in ['base_root','formal_root','mapping_path','helper_source','old_diagnostic_arrays','expected_uuid','estimated_lease_end_utc','preservation_reserve_seconds','minimum_remote_free_bytes','beta'] if k in base}
plan['beta']=65294.20088076077
plan.update(status='FROZEN_MATCHED_REAL_MESSAGE_POOLS_V1',frozen_utc=now.isoformat(),new_root=remote,teacher_path_arrays=teacher,
    source_sha256=sha(dest/files[0]),pinned_files=pins,maximum_seconds=2700,maximum_peak_allocated_bytes=1024**3,maximum_new_array_bytes=512*1024**2,
    main_comparison='P_OT vs P_Oplusminus; same nine nominal candidates, output cap and CAL acceptor.',
    directions='normalized original old three-step last message displacement, teacher three-step last displacement, negative old displacement; fixed F proximity anchor.',
    radius='.25*norm(F/train_message_rms), amplitudes .125,.25,.5,1; teacher generator uses original mu; CAL lambda only scores candidates.',
    candidate_order='F then old amplitudes ascending then teacher/negative-old amplitudes ascending. No native baseline asymmetrically added.',
    shared_forward_budget='13 nominal actual terminal calls, exact message duplicates identified; all forwarded at unchanged batch size for deterministic replay. No extra directions added.',
    acceptors='old scalar and pF+lambda_fold*(mu-pF), float64 Q relative to exact F; cap first, earliest strictly negative then F; lambda0 CAL exact F.',
    eval='Only fixed EVAL863/34 videos after all eight predictions and SHA frozen; CAL418 already fit, no refit. Entire TRAIN was explored previously; fullTRAINfit/DEVselected reference, not pristine confirmation or whole pipeline crossfit.',
    eval_metrics='row/video equal MSE, MAE, Q vs F/native; acceptance and harmful/beneficial fractions, all34 videos; each same-cap pool finite-label oracle only post-freeze capacity diagnostic.',
    stopping='If teacher union offers no added capacity over old+negative-old or selectors cannot improve native control, no new100. No epsilon/eta or parameter searching.',
    scope='New finite real-message inference on immutable cached coordinate terminal. No full original DeBERTa forward repeated, training, DEV or TEST.')
(dest/'plan.json').write_text(json.dumps(plan,indent=2))
manifest={name:dict(sha256=sha(dest/name),bytes=(dest/name).stat().st_size) for name in files+['plan.json','roles.npz','mu_input.npz','cal_parameters.json']}
(dest/'local_freeze.json').write_text(json.dumps(dict(utc=now.isoformat(),files=manifest,remote=remote,local_C_free=shutil.disk_usage('C:/').free,local_D_free=shutil.disk_usage('D:/').free,executed=False),indent=2))
(cwd/'outputs/匹配消息候选短实验执行接续.json').write_text(json.dumps(dict(stage='FROZEN_LOCAL_NOT_DEPLOYED',local_root=str(dest),remote_root=remote,source_sha256=plan['source_sha256'],plan_sha256=sha(dest/'plan.json')),indent=2))
print(json.dumps(dict(local_root=str(dest),remote_root=remote,source_sha256=plan['source_sha256'],plan_sha256=sha(dest/'plan.json'))))
