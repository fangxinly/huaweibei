import pathlib, hashlib, json, zipfile, datetime, tomllib
HERE=pathlib.Path(__file__).resolve().parents[1]
DEST=pathlib.Path('D:/CodexBackups/selective_flow_20261003_1105/continuation_20261005T0253Z')
DEST.mkdir(exist_ok=False)
def sha(b):return hashlib.sha256(b).hexdigest()
automation=tomllib.loads(pathlib.Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'))
assert automation['status']=='ACTIVE' and automation['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58'
receipt={k:automation[k] for k in ['id','name','status','rrule','target_thread_id','updated_at']}
receipt.update(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),prompt_chars=len(automation['prompt']),prompt_sha256=sha(automation['prompt'].encode()),verification='updated tool result and actual UTF8 TOML target/status/rrule; automation-2 untouched')
(HERE/'outputs/研究监管迁移核验.json').write_text(json.dumps(receipt,indent=2,ensure_ascii=False),encoding='utf-8')
files=[]
for folder in [HERE/'outputs',HERE/'work']:
 for p in sorted(folder.rglob('*')):
  if p.is_file() and p.suffix not in ('.pt','.tmp','.pyc') and '__pycache__' not in p.parts:
   files.append((str(p.relative_to(HERE)).replace('\\','/'),p))
old=pathlib.Path('D:/CodexBackups/selective_flow_20261003_1105/inflow_conditions_20261005022433Z')
for p in sorted(old.rglob('*')):
 if p.is_file() and p.suffix in ('.json','.zip','.npy','.npz'):
  files.append(('v2_preservation/'+p.relative_to(old).as_posix(),p))
members=[]
archive=DEST/'metadata.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for name,p in files:
  b=p.read_bytes();z.writestr(name,b);members.append({'name':name,'source':str(p),'bytes':len(b),'sha256':sha(b)})
manifest={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive.read_bytes()),'members':members,'limits':'Small metadata and source only. Excludes every complete .pt. Existing checkpoint preservation proofs are included, not re-transfers.'}
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(members)
 for m in members:
  b=z.read(m['name']);assert len(b)==m['bytes'] and sha(b)==m['sha256']
  assert sha(pathlib.Path(m['source']).read_bytes())==m['sha256']
(DEST/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps({'destination':str(DEST),'members':len(members),'bytes':manifest['archive_bytes'],'sha256':manifest['archive_sha256'],'status':'LOCAL_ORIGINALS_AND_D_ZIP_ALL_MEMBERS_VERIFIED'},indent=2))
