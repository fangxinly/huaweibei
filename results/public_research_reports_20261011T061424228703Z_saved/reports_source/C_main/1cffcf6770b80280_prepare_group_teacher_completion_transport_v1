"""Freeze auxiliary preservation payload; never change formal training sources."""
from pathlib import Path
import ast,datetime,hashlib,json,shutil,zipfile
base=Path(__file__).resolve().parent.parent
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('group_teacher_preservation_payload_'+stamp)
assert shutil.disk_usage(dest.parent).free>8*1024**3
dest.mkdir()
names=['audit_group_teacher_completed_files_v1.py','manifest_group_teacher_completed_v1.py','verify_group_teacher_completed_cpu_v1.py','train_row_video_mapping.json']
meta={}
with zipfile.ZipFile(dest/'payload.zip','w',compression=zipfile.ZIP_STORED) as z:
 for name in names:
  source=base/'work'/name
  if name.endswith('.py'):ast.parse(source.read_text(encoding='utf-8'))
  raw=source.read_bytes();meta[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
  shutil.copy2(source,dest/name);z.writestr(name,raw)
report={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'New preservation helper files and immutable original mapping only; formal sources unchanged. Not a completion or CPU preservation claim.','destination':str(dest),'files':meta,'zip_sha256':hashlib.sha256((dest/'payload.zip').read_bytes()).hexdigest()}
(dest/'payload_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
(base/'outputs'/('视频隔离教师完成保存辅助源冻结_'+stamp+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
