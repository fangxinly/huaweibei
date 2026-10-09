import base64,datetime,hashlib,json,pathlib,sys,zipfile
r=pathlib.Path(__file__).resolve().parent;stage=sys.argv[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
v=json.loads(base64.b64decode(''.join((r/(stage+'_wrapped_read_original.b64')).read_text().split())));cap=json.loads(base64.b64decode(v['capture_receipt.json']));p=r/(stage+'_complete_original.zip');assert sha(p)==cap['archive_SHA'] and p.stat().st_size==cap['archive_bytes'] and cap['natural_exit']==0
d=r/(stage+'_original');d.mkdir(exist_ok=True)
with zipfile.ZipFile(p) as z:
 rows=json.loads(z.read('member_manifest.json'));assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())=={v['name'] for v in rows}|{'member_manifest.json'}
 for v in rows:
  b=z.read(v['name']);assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256'];q=d/v['name'];assert q.resolve().is_relative_to(d.resolve());q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(b)
result=json.loads((d/'result.json').read_bytes());assert json.loads((d/'natural_exit.json').read_bytes())['natural_exit']==0
receipt=dict(status=stage+'_ORIGINAL_D_SHA_CRC_UNIQUE_VERIFIED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive=str(p),archive_SHA=sha(p),archive_bytes=p.stat().st_size,member_count=len(rows),all_member_SHA_CRC_unique_passed=True,original_capture=cap,result=result)
(r/(stage+'_D_preservation_receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt))
