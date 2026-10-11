from pathlib import Path
import ast,datetime,hashlib,json,shutil
import numpy as np
base=Path(__file__).parent.parent;work=base/'work';D=Path('D:/CodexBackups/selective_flow_20261003_1105')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
assert shutil.disk_usage(D).free>1073741824 and shutil.disk_usage('C:/').free>1073741824
dest=D/('cal_video_equal_signal_actual_'+stamp);dest.mkdir(exist_ok=False)
roles=work/'correction_calibration_collection_v2_20261006T0414Z'
assert sha(roles/'calibration_video_role_plan_v1.json')=='785c27b7bfb53796e3ee5d6cc9737009c85a29c1124a6633538f0e526722f0cc'
for name in ['calibration_video_role_plan_v1.json','row_fold_calibration_roles.npz','calibration_0.npy','calibration_1.npy','calibration_2.npy']:shutil.copy2(roles/name,dest/name)
geometry=D/'scalar_geometry_actual_20261006T060719Z/execute/geometry_frozen.npz'
assert sha(geometry)=='e0016c23628ca0584784e417465277e4ef516cc884ff22ea4f2113c26f9ead91'
old=D/'train_oracle_completed_20261006T0158Z/c/out/diagnostics.npz'
assert sha(old)=='92ee406345d041826a3f9564447c761e59745878c9770c3f9025a4a948eb2658'
with np.load(geometry,allow_pickle=False) as g,np.load(old,allow_pickle=False) as o:
    fields={key:g[key] for key in ['row','fold','video','pf','mu']}
    assert np.array_equal(o['row'],fields['row']) and np.array_equal(o['video'],fields['video'])
    assert np.max(np.abs(o['raw_prediction']-fields['pf']))<=1e-6
    fields['old_prediction']=o['original_prediction']
    np.savez_compressed(dest/'geometry_inputs.npz',**fields)
original_label_dir=D/'oof_utility_completed_20261006T033054Z/original_C/execute'
receipt=json.loads((original_label_dir/'receipt.json').read_text())
assert receipt['labels_sha256']==sha(original_label_dir/'metric_labels.npz')=='ffd3f8b52b6620e616f0ec715b64fdb0398d0e7fc4b8be2d252234cfe2ff769f'
shutil.copy2(original_label_dir/'metric_labels.npz',dest/'metric_labels.npz')
source=work/'calibrate_video_equal_residual_v1.py';ast.parse(source.read_text(encoding='utf-8'));shutil.copy2(source,dest/source.name)
pins={p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!=source.name}
plan=dict(status='FROZEN_CAL_ONLY_VIDEO_EQUAL_RESIDUAL_SIGNAL',frozen_utc=now.isoformat(),source_sha256=sha(source),input_pins=pins,cal_counts=[136,148,134],eval_counts=[297,276,290],cal_videos_per_fold=6,folds=3,
    estimator='Per own fold video-equal C=mean_v mean_rows((y-pf)*(mu-pf)); B=mean_v mean_rows((mu-pf)^2); lambda=clip(C/B,0,1), B=0 ->0.',
    output_cap='Fixed CAL original native absolute output shift quantile=.95 method=linear; no quantile search.',
    sensitivity='All six leave-one-video-out values reported, never pick parameter/drop a video.',
    labels='Only frozen 418 CAL numeric elements decoded via bounded selective NPY reader; raw byte SHA/ZIP seeking not a semantic EVAL read. Source labels were already explored earlier, not pristine confirmation.',
    eval_metrics_allowed=False,model_or_message_forward=False,whole_pipeline_crossfit=False,new_training=False,
    original_geometry_sha256=sha(geometry),original_old_control_array_sha256=sha(old),original_label_receipt_sha256=sha(original_label_dir/'receipt.json'),
    stop_scope='C<=0 excludes only global positive shrinkage; does not rule out all nonlinear candidates. Leave-one-video-out sign instability discourages teacher-led expansion.',
    maximum_seconds=60,maximum_new_data_bytes=10485760,permanent_space_checked=True)
(dest/'plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2),encoding='utf-8')
proof=dict(status='CAL_SIGNAL_PLAN_FROZEN_BEFORE_REAL_LABEL_DECODE',actual_utc=now.isoformat(),root=str(dest),source_sha256=sha(source),plan_sha256=sha(dest/'plan.json'),input_pins=pins,real_CAL_fit_not_yet_executed=True,real_EVAL_values_not_decoded=True)
(base/'outputs/视频等权CAL信号执行接续.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
(dest/'freeze_receipt.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(proof,ensure_ascii=True))
