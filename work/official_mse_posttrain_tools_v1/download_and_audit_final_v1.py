"""B downloads new completed original once, verifies all bytes and complete optimizer state."""
import argparse,datetime,hashlib,json,os,subprocess,sys,urllib.request,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--publication',type=Path,required=True);p.add_argument('--capture',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
for n in ('plan','bundle','assets'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--received-archive',type=Path);p.add_argument('--plan-sha',required=True);a=p.parse_args()
def sha(p):return digest(p)
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def blockhash(f):
 h=hashlib.sha256()
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
r=json.loads(a.publication.read_text());cap=json.loads(a.capture.read_text());assert r['status']=='REMOTE_RELEASE_ALL_DIGESTS_VERIFIED' and cap['natural_exit']==0 and cap['ZIP_CRC_unique_all_members']
assert subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()=='GPU-609b23d6-282d-8a2b-23f5-9433b824d212'
a.out.mkdir(exist_ok=False);parts=sorted([x for x in r['assets'] if x['source_sha256']==cap['archive_SHA']],key=lambda x:x['source_offset']);assert parts
archive=a.out/'A_complete_original.zip'
if a.received_archive:
 plan=json.loads(a.plan.read_text());transport=plan['received_original_transport'];assert transport['method']=='D_verified_original_SFTP' and str(a.received_archive)==transport['remote_archive']
 assert a.received_archive.stat().st_size==cap['archive_bytes'] and digest(a.received_archive)==cap['archive_SHA']
 os.link(a.received_archive,archive)
 (a.out/'transport_receipt.json').write_text(json.dumps(dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),method=transport['method'],archive_SHA=digest(archive),bytes=archive.stat().st_size,public_download=False,local_D_verification=transport['D_verification'])))
else:
 with archive.open('xb') as target:
  for row in parts:
   assert target.tell()==row['source_offset'];h=hashlib.sha256();n=0
   with urllib.request.urlopen(row['url'],timeout=180) as resp:
    for b in iter(lambda:resp.read(8*1024**2),b''):target.write(b);h.update(b);n+=len(b)
   assert n==row['bytes'] and h.hexdigest()==row['sha256']
assert archive.stat().st_size==cap['archive_bytes'] and digest(archive)==cap['archive_SHA']
original=a.out/'original';original.mkdir()
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for name in z.namelist():assert (original/name).resolve().is_relative_to(original.resolve())
 members=json.loads(z.read('member_SHA.json'))
 for name,h in members.items():
  with z.open(name) as f:assert blockhash(f)==h
 z.extractall(original)
sys.path.insert(0,str(a.bundle))
from audit_official import run
from types import SimpleNamespace
run(SimpleNamespace(plan=a.plan,plan_sha=a.plan_sha,bundle=a.bundle,assets=a.assets,input_root=original,out=a.out/'cpu_result'))
print((a.out/'cpu_result/audit_result.json').read_text(),flush=True)
