from pathlib import Path
import ast,hashlib,json,datetime
w=Path(__file__).resolve().parent
s=(w/'capture_soft_vector_v18.py').read_text(encoding='utf-8').replace('capture_soft_vector_v18.py','capture_soft_vector_v19.py').replace('zipfile,shutil','zipfile,shutil,os')
s=s.replace("   if not path.is_file() or path.suffix", "   if '.pending' in path.name:continue\n   if not path.is_file() or path.suffix")
old="   if path.stat().st_size>8*1024**2:large[name]={'path':str(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'sha256':sha(path.read_bytes())}"
new="""   if path.stat().st_size>8*1024**2:
    with path.open('rb') as handle:
     meta=os.fstat(handle.fileno());digest=hashlib.sha256()
     for chunk in iter(lambda:handle.read(1024*1024),b''):digest.update(chunk)
    large[name]={'path':str(path),'bytes':meta.st_size,'mtime_ns':meta.st_mtime_ns,'sha256':digest.hexdigest(),'scope':'Stable opened inode SHA reference; no fullweight transfer or CPU preservation asserted.'}"""
assert old in s;s=s.replace(old,new)
s=s.replace("  encoded=json.dumps(ti,indent=2).encode();z.writestr('group_teacher_inventory.json'", "  history=json.loads((teacher/'run_v1/history.json').read_text()) if (teacher/'run_v1/history.json').exists() else []\n  ti.update(history_epochs=len(history),last_history_epoch=history[-1] if history else None,formal_protocol_present=(teacher/'run_v1/protocol.json').exists(),formal_exit=json.loads((teacher/'formal_exit.json').read_text()) if (teacher/'formal_exit.json').exists() else None)\n  encoded=json.dumps(ti,indent=2).encode();z.writestr('group_teacher_inventory.json'")
s=s.replace('Large fullweights fresh SHA referenced, not included in ZIP. Already permanent copies verified separately; snapshot is not itself a new fullweight download.','Large fullweights fresh stable-inode SHA referenced, not included in ZIP. Earlier completed weights have separate permanent copies. New teacher initial or selected weights are not declared downloaded or CPU preserved by this snapshot. Pending files excluded.')
ast.parse(s);t=w/'capture_soft_vector_v19.py';assert not t.exists();t.write_text(s,encoding='utf-8')
(w/'audit_soft_snapshot_v19.py').write_text((w/'audit_soft_snapshot_v18.py').read_text(encoding='utf-8').replace('capture_soft_vector_v18.py','capture_soft_vector_v19.py'),encoding='utf-8')
proof=dict(status='NEW_TEACHER_LIVE_CAPTURE19_STABLE_INODE_REFERENCES_PENDING_EXCLUSION_SCOPE_FROZEN',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),capture_sha256=hashlib.sha256(t.read_bytes()).hexdigest(),prior_capture_versions_retained=True)
(w.parent/'outputs/视频隔离教师live捕获19冻结.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))
