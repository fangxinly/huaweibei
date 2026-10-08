import pathlib, hashlib, json, zipfile, datetime, sys
root=pathlib.Path(sys.argv[1]);out=root/'target_verification.json'
def sha(b):return hashlib.sha256(b).hexdigest()
m=json.loads((root/'manifest.json').read_text(encoding='utf-8'));p=root/'metadata.zip'
assert p.stat().st_size==m['archive_bytes'] and sha(p.read_bytes())==m['archive_sha256']
with zipfile.ZipFile(p) as z:
 assert len(z.namelist())==len(m['members'])
 for f in m['members']:
  b=z.read(f['name']);assert len(b)==f['bytes'] and sha(b)==f['sha256']
result={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'INDEPENDENT_REMOTE_ZIP_AND_ALL_MEMBERS_VERIFIED','directory':str(root),'archive_bytes':p.stat().st_size,'archive_sha256':m['archive_sha256'],'manifest_sha256':sha((root/'manifest.json').read_bytes()),'members':len(m['members'])}
with out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2)
print(json.dumps(result))
