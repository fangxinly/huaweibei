from pathlib import Path
import hashlib,json,zipfile,sys,datetime
r=Path(sys.argv[1]);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
receipt=json.loads((r/'transport_receipt.json').read_text())
assert sha(r/'completed_payload.zip')==receipt['sha256'] and (r/'completed_payload.zip').stat().st_size==receipt['bytes']
dest=r/'original_C';assert not dest.exists();dest.mkdir()
with zipfile.ZipFile(r/'completed_payload.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==receipt['members']
    assert all(not p.startswith('/') and '..' not in Path(p).parts for p in z.namelist())
    z.extractall(dest)
manifest=json.loads((dest/'preservation_manifest.json').read_text())
for name,meta in manifest['members'].items():
    p=dest/name;assert p.stat().st_size==meta['bytes'] and sha(p)==meta['sha256'],name
assert set(p.relative_to(dest).as_posix() for p in dest.rglob('*') if p.is_file())==set(manifest['members'])|{'preservation_manifest.json'}
proof=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),local=str(dest),zip_sha256=receipt['sha256'],member_sha_all_passed=True,zip_crc_passed=True,members=receipt['members'])
(r/'local_transport_audit.json').write_text(json.dumps(proof,indent=2),encoding='utf-8');print(json.dumps(proof))
