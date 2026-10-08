from pathlib import Path
import datetime,hashlib,json,shutil,zipfile
base=Path(__file__).parent.parent;root=Path('D:/CodexBackups/selective_flow_20261003_1105/cal_video_equal_signal_actual_20261006T063535Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert shutil.disk_usage(root).free>1073741824
for phase in ['precheck','execute']:assert json.loads((root/(phase+'_exit.json')).read_text())['exit_code']==0
for name in ['audit_cal_video_equal_signal_v1.py','run_cal_video_equal_signal_v1.py','freeze_cal_video_equal_signal_v1.py']:shutil.copy2(base/'work'/name,root/name)
assert json.loads((root/'local_independent_audit.json').read_text())['status']=='CAL_ONLY_VIDEO_EQUAL_INDEPENDENT_CPU_AUDIT_PASSED'
members={};dest=root/'preservation';dest.mkdir(exist_ok=False)
with zipfile.ZipFile(dest/'snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(root.rglob('*')):
        if not p.is_file() or dest in p.parents or p.name=='metric_labels.npz' or '__pycache__' in p.parts:continue
        name=p.relative_to(root).as_posix();data=p.read_bytes();z.writestr(name,data);members[name]=dict(bytes=len(data),sha256=sha(p))
    z.writestr('member_manifest.json',json.dumps(members,ensure_ascii=False,indent=2))
with zipfile.ZipFile(dest/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    assert 'metric_labels.npz' not in z.namelist()
r=dict(status='CAL_SIGNAL_ORIGINAL_PACKAGE_COMPLETE',actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=sha(dest/'snapshot.zip'),bytes=(dest/'snapshot.zip').stat().st_size,members=len(members),full_TRAIN_label_archive_transferred=False,real_outcome_rows_in_packet=418,real_outcome_videos_in_packet=18,no_EVAL_ground_truth_packet=True)
(dest/'receipt.json').write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r))
