from pathlib import Path
import json,zipfile,hashlib
root=Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_completed_20261006')
for node in ['a','b','c']:
 d=root/node;r=json.loads((d/'completed_small_transport_receipt.json').read_text(encoding='utf-8'));raw=(d/'completed_small_originals.zip').read_bytes()
 assert hashlib.sha256(raw).hexdigest()==r['sha256'] and len(raw)==r['bytes']
 with zipfile.ZipFile(d/'completed_small_originals.zip') as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for name in z.namelist():
   dest=(d/name).resolve();assert dest.is_relative_to(d.resolve()) and not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
 assert hashlib.sha256((d/'preservation_manifest.json').read_bytes()).hexdigest()==r['manifest_sha256']
 print('LOCAL_SMALL_ORIGINALS_SHA_CRC_EXTRACTED',node)
