from pathlib import Path
import hashlib,json,zipfile,datetime,ast,shutil
w=Path(__file__).resolve().parent
root=w/'group_teacher_plan_20261005T1650Z'
source=(w/'capture_soft_vector_v16.py').read_text(encoding='utf-8').replace('capture_soft_vector_v16.py','capture_soft_vector_v17.py')
insert="""
 teacher=Path('/data/coding/group_teacher_v1_20261005T1650Z')
 tree(teacher,'group_teacher')
 if teacher.exists():
  for path in sorted(teacher.rglob('*')):
   if path.is_file() and path.suffix in ['.log','.txt']:
    add(path,'group_teacher/'+path.relative_to(teacher).as_posix())
  matches=[]
  for candidate in sorted(Path('/proc').iterdir()):
   if not candidate.name.isdigit():continue
   try:raw=(candidate/'cmdline').read_bytes().decode().split('\\0')[:-1]
   except (OSError,UnicodeError):continue
   if any(Path(arg).name in ['check_group_teacher_v1.py','train_group_teacher_v1.py'] for arg in raw) and str(teacher) in raw:
    matches.append({'pid':int(candidate.name),'argv':raw})
  ti=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(teacher),gpu=inv['gpu'],compute=inv['compute'],matching_processes=matches,data_disk_free=shutil.disk_usage('/data').free,precheck_receipt_present=(teacher/'precheck_v1/receipt.json').is_file(),completion_present=(teacher/'run_v1/completion.json').is_file(),scope='Capture only; precheck receipt is not 100 epochs or OOF completion.')
  encoded=json.dumps(ti,indent=2).encode();z.writestr('group_teacher_inventory.json',encoded);members['group_teacher_inventory.json']={'bytes':len(encoded),'sha256':sha(encoded)}
"""
needle=" z.writestr('large_file_manifest.json'"
source=source.replace(needle,insert+'\n'+needle,1)
ast.parse(source)
cap=w/'capture_soft_vector_v17.py';assert not cap.exists();cap.write_text(source,encoding='utf-8')
aud=(w/'audit_soft_snapshot_v16.py').read_text(encoding='utf-8').replace('capture_soft_vector_v16.py','capture_soft_vector_v17.py')
(w/'audit_soft_snapshot_v17.py').write_text(aud,encoding='utf-8')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={p.name:p for p in sorted(root.iterdir()) if p.suffix in ['.py','.json','.npy']}
files['audit_group_teacher_mechanism_receipts_v1.py']=w/'audit_group_teacher_mechanism_receipts_v1.py'
plan=json.loads((root/'group_teacher_plan_v2.json').read_text())
for n,h in plan['source_sha256'].items():assert sha(files[n])==h
manifest={n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in files.items()}
bundle=w/'group_teacher_precheck_deployment_v1.zip';assert not bundle.exists()
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED) as z:
 for n,p in files.items():z.write(p,n)
 z.writestr('deployment_manifest.json',json.dumps(manifest,indent=2))
proof=dict(status='NEW_CAPTURE17_TEACHER_PRECHECK_ROOT_COVERAGE_AND_ORIGINAL_PINNED_PAYLOAD_PREPARED_NOT_GPU_EXECUTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),capture_sha256=sha(cap),payload_sha256=sha(bundle),payload_bytes=bundle.stat().st_size,files=manifest,old_scientific_sources_modified=False)
out=w.parent/'outputs/视频隔离教师部署与capture17准备核验.json';out.write_text(json.dumps(proof,indent=2),encoding='utf-8')
backup=Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_precheck_20261006T0210Z');backup.mkdir(exist_ok=False)
for p in [cap,w/'audit_soft_snapshot_v17.py',bundle,out,Path(__file__).resolve()]:
 shutil.copy2(p,backup/p.name);assert sha(backup/p.name)==sha(p)
print(json.dumps({k:proof[k] for k in ['status','capture_sha256','payload_sha256','payload_bytes']}))
