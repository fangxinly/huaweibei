from pathlib import Path
import hashlib,json,shutil,datetime,ast
w=Path(__file__).parent;r=Path('D:/CodexBackups/selective_flow_20261003_1105/matched_message_pools_actual_20261006T064950Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name in ['evaluate_matched_message_pools_v1.py','audit_matched_pool_eval_v1.py']:
    ast.parse((w/name).read_text());shutil.copy2(w/name,r/name)
plan=dict(status='FROZEN_EVAL_AFTER_ALL_PREDICTIONS_ACCOUNTING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_plan_sha256=sha(r/'plan.json'),
    source_sha256=sha(r/'evaluate_matched_message_pools_v1.py'),auditor_sha256=sha(r/'audit_matched_pool_eval_v1.py'),
    label_archive='D:/CodexBackups/selective_flow_20261003_1105/cal_video_equal_signal_actual_20261006T063535Z/metric_labels.npz',
    label_archive_sha256='ffd3f8b52b6620e616f0ec715b64fdb0398d0e7fc4b8be2d252234cfe2ff769f',role_sha256=sha(r/'roles.npz'),
    limits='Previously explored TRAIN863 evaluation role; reference fullTRAINfit/DEVselected. Descriptive finite candidate evidence, not untouched confirmation/whole-pipeline crossfit/generalization/SOTA. No CAL refit, cap/lambda selection, DEV/TEST or new100.',
    protocol='All eight arms frozen SHA before numeric EVAL read. Main CAL P_OT9 vs P_Opm9, native unchanged separate baseline. Fixed-cap finite-label oracle only after freeze, no parameter selection.')
assert sha(plan['label_archive'])==plan['label_archive_sha256']
(r/'evaluation_plan.json').write_text(json.dumps(plan,indent=2));print(json.dumps(dict(status=plan['status'],evaluation_plan_sha256=sha(r/'evaluation_plan.json'))))
