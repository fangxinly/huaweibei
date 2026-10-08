from pathlib import Path
import datetime,hashlib,json,shutil
base=Path(__file__).resolve().parent.parent;parent=Path('D:/CodexBackups/selective_flow_20261003_1105');assert shutil.disk_usage(parent).free>4*1024**3
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');dest=parent/('group_teacher_completion_research_'+stamp);dest.mkdir()
names=['outputs/视频隔离教师到学生的标签路径与校准修订设计.md','outputs/原消息相对保守接受的输出尺度解析核验.md','outputs/原消息相对保守接受输出解析核验.json','work/verify_conservative_output_prox_v1.py','work/audit_group_teacher_completed_preservation_snapshot_v1.py','work/audit_group_teacher_cpu_receipts_local_v1.py','work/group_teacher_completed_local_transport_v1.py','outputs/视频隔离教师0253轮原始监管与完成门控.json']
files={}
for name in names:
 src=base/name;out=dest/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,out);raw=src.read_bytes();assert raw==out.read_bytes();files[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
report={'status':'NEW_LOCAL_LABEL_PATH_DESIGN_AND_OUTPUT_PROX_MATH_RESEARCH_PERMANENT_SHA_VERIFIED_NOT_NEW_GPU_TRAINING','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'directory':str(dest),'files':files}
(dest/'preservation_record.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');(base/'outputs'/('视频隔离教师完成轮新研究永久保存_'+stamp+'.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'directory':str(dest),'files':len(files)}))
