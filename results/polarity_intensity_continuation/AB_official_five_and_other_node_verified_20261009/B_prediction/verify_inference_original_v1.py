import base64,datetime,hashlib,json,pathlib,zipfile
r=pathlib.Path(__file__).resolve().parent
b=base64.b64decode(''.join((r/'inference_completed_read_original.b64').read_text().split()));v=json.loads(b)
print('READ_KEYS',list(v));d=r/'inference_completed_read_original';d.mkdir(exist_ok=True)
for n,s in v['files'].items():
 p=d/n;assert p.resolve().is_relative_to(d.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(s))
p=r/'B_complete_official_inference_original.zip';h=hashlib.sha256(p.read_bytes()).hexdigest();expected=json.loads((d/'capture_receipt.json').read_bytes());assert h==expected['archive_SHA'] and p.stat().st_size==expected['archive_bytes']==390867
dest=r/'B_inference_original';dest.mkdir(exist_ok=True)
with zipfile.ZipFile(p) as z:
 rows=json.loads(z.read('member_manifest.json'));names=z.namelist();assert z.testzip() is None and len(names)==len(set(names)) and set(names)=={a['name'] for a in rows}|{'member_manifest.json'}
 for a in rows:
  raw=z.read(a['name']);assert len(raw)==a['bytes'] and hashlib.sha256(raw).hexdigest()==a['sha256'];q=dest/a['name'];assert q.resolve().is_relative_to(dest.resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(raw)
results=list(dest.rglob('*result*.json'));print('RESULT_PATHS',[str(a.relative_to(dest)) for a in results]);print('DECODE_PATHS',[str(a.relative_to(d)) for a in d.rglob('*.json')])
cap=json.loads((d/'capture_receipt.json').read_bytes());assert cap['natural_exit']==0 and cap['archive_SHA']==h
receipt=dict(status='B_N2_OFFICIAL_INFERENCE_ORIGINAL_D_SHA_CRC_UNIQUE_VERIFIED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(p),archive_SHA=h,archive_bytes=p.stat().st_size,member_count=len(rows),all_member_SHA_CRC_unique_passed=True,original_capture_receipt=cap,SCP_29510_natural_exit=0,official_B_five_not_yet_scored=True)
(r/'B_inference_D_preservation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt))
