from pathlib import Path
import datetime,hashlib,json,tomllib,zipfile
ROOT=Path(__file__).resolve().parents[1]
DEST=Path('D:/CodexBackups/selective_flow_20261003_1105/gated_metadata_20261005T0341Z');DEST.mkdir(exist_ok=False)
def sha(b):return hashlib.sha256(b).hexdigest()
automation=tomllib.loads(Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'))
assert automation['status']=='ACTIVE' and automation['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58'
r={k:automation[k] for k in ('id','name','status','rrule','target_thread_id','updated_at')}
r.update(verified_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),prompt_chars=len(automation['prompt']),prompt_sha256=sha(automation['prompt'].encode()),phase='Gated v3 completed; next utility candidate implementation and lease saves pending')
(ROOT/'outputs/研究监管阶段更新核验.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
files={}
folders=[ROOT/'outputs',ROOT/'work/live_20261005032625Z',ROOT/'work/live_20261005032825Z',ROOT/'work/gated_followup_202610050335Z',ROOT/'work/gated_v3_completed_20261005',ROOT/'work/inflow_gated_v3_20261005T0241Z']
for folder in folders:
 for p in sorted(folder.rglob('*')):
  if p.is_file() and p.suffix not in ('.pt','.tmp','.pyc') and '__pycache__' not in p.parts:files[p.relative_to(ROOT).as_posix()]=p
for p in sorted((ROOT/'work').glob('*.py')):files[p.relative_to(ROOT).as_posix()]=p
oldstate=ROOT/'work/state_before_gated_completion_20261005T0340Z.md';files[oldstate.relative_to(ROOT).as_posix()]=oldstate
broot=Path('D:/CodexBackups/selective_flow_20261003_1105/gated_v3_completed_20261005/b')
for p in sorted(broot.rglob('*')):
 if p.is_file() and p.suffix!='.pt':files['gated_b/'+p.relative_to(broot).as_posix()]=p
members=[]
with zipfile.ZipFile(DEST/'metadata.zip','x',zipfile.ZIP_DEFLATED) as z:
 for name,p in files.items():
  b=p.read_bytes();z.writestr(name,b);members.append({'name':name,'source':str(p),'bytes':len(b),'sha256':sha(b)})
with zipfile.ZipFile(DEST/'metadata.zip') as z:
 assert len(z.namelist())==len(members)
 for m in members:
  b=z.read(m['name']);assert len(b)==m['bytes'] and sha(b)==m['sha256'] and sha(Path(m['source']).read_bytes())==m['sha256']
p=DEST/'metadata.zip'
m={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_bytes':p.stat().st_size,'archive_sha256':sha(p.read_bytes()),'members':members,'limits':'Small evidence, source and metadata only; excludes all .pt. Source weights and independent full copies already verified separately. Later connection-close receipts are appended separately.'}
(DEST/'manifest.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
receipt={k:m[k] for k in ('created_at','archive_bytes','archive_sha256','limits')};receipt.update(destination=str(DEST),members=len(members),status='LOCAL_ORIGINALS_AND_D_METADATA_ZIP_ALL_MEMBERS_SHA_VERIFIED')
(ROOT/'outputs/门控资料本地保存核验.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print(json.dumps(receipt))
