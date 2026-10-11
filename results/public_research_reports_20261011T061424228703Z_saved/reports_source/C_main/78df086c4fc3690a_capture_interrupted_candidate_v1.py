"""Capture a genuinely absent interrupted process; never invent natural exit."""
import argparse,datetime,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--prior',type=Path,required=True);p.add_argument('--bundle',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--uuid',required=True);a=p.parse_args()
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as z:
  for b in iter(lambda:z.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(f,v):f.write_text(json.dumps(v,indent=2),encoding='utf8')
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
dispatch=json.loads((a.prior/'actual_dispatch.json').read_text());uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==a.uuid
assert not Path('/proc/'+str(dispatch['pid'])).exists();assert not (a.prior/'natural_exit.json').exists() and not (a.prior/'capture_receipt.json').exists()
a.out.mkdir()
actual=dict(actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,UUID=uuid,prior_dispatch=dispatch,last_progress=json.loads((a.prior/'out/progress.json').read_text()),prior_process_absent=True,prior_natural_exit_unknown=True,interruption_cause_not_established=True,compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True))
write(a.out/'interruption_observation.json',actual)
warm=a.prior/'out/complete_resume_step400.pt';ready=json.loads((a.prior/'out/warmup400_preservation_ready.json').read_text());assert sha(warm)==ready['checkpoint_SHA'] and warm.stat().st_size==ready['checkpoint_bytes']
files={}
for root,pr in [(a.prior,'prior'),(a.bundle,'original_source'),(a.out,'observation')]:
 for f in root.rglob('*'):
  if f.is_file():files[pr+'/'+f.relative_to(root).as_posix()]=f
members={n:sha(f) for n,f in files.items()};write(a.out/'member_SHA.json',members)
archive=a.out/'complete_interrupted_original.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,1) as z:
 for n,f in files.items():z.write(f,n)
 z.write(a.out/'member_SHA.json','member_SHA.json')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in members.items():
  q=hashlib.sha256()
  with z.open(n) as f:
   for b in iter(lambda:f.read(8*1024**2),b''):q.update(b)
  assert q.hexdigest()==h
r=dict(status='INTERRUPTED_CANDIDATE_ORIGINAL_COMPLETE_SHA_CRC_VERIFIED',actual_UTC=utc(),capture_process_natural_exit=0,prior_training_natural_exit_unknown=True,prior_training_process_absent=True,prior_complete100=False,last_epoch=actual['last_progress']['epoch'],archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,ZIP_CRC_unique_all_members=True,warm400_checkpoint_SHA=ready['checkpoint_SHA'],warm400_checkpoint_bytes=ready['checkpoint_bytes'])
write(a.out/'capture_receipt.json',r);print(json.dumps(r),flush=True)
