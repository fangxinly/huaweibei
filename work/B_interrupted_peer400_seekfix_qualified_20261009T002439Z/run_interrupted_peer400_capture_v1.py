"""Fresh peer CPU audit, preserve natural child exit and every small original."""
import argparse,datetime,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as z:
  for b in iter(lambda:z.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(f,r):Path(f).write_text(json.dumps(r,indent=2),encoding='utf8')
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
p=argparse.ArgumentParser()
for n in ('plan','assets','root'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--plan-sha',required=True);a=p.parse_args()
assert sha(a.plan)==a.plan_sha;q=json.loads(a.plan.read_text());bundle=a.plan.parent
for n,h in q['source_sha256'].items():assert sha(bundle/n)==h,n
for n,h in q['qualification_sha256'].items():assert sha(bundle/n)==h,n
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert uuid==q['GPU_UUID'][q['execution_node']] and not compute.strip()
assert sys.executable==str(a.assets/'.venv/bin/python')
assert {n:importlib.metadata.version(n) for n in q['runtime_versions']}==q['runtime_versions']
for n,h in q['asset_sha256'].items():assert sha(a.assets/n)==h,n
assert shutil.disk_usage(a.assets).free>q['remote_free_floor_bytes']
remaining=(datetime.datetime.fromisoformat(q['conservative_lease_end_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert remaining>7200+1800
assert not a.root.exists();a.root.mkdir()
write(a.root/'fresh_preflight.json',dict(actual_UTC=utc(),UUID=uuid,compute=compute,fullargv=[sys.executable]+sys.argv,pid=os.getpid(),source_SHA=sha(__file__),space=shutil.disk_usage(a.root)._asdict(),remaining_seconds=remaining,source_assets_runtime_passed=True,processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)))
cmd=[sys.executable,str(bundle/'audit_interrupted_candidate400_v1.py'),'--publication',str(bundle/'publication.json'),'--capture',str(bundle/'interrupted_capture.json'),'--out',str(a.root/'out'),'--uuid',uuid,'--peer',q['source_peer'],'--retained-archive',q['retained_archive']]
with (a.root/'stdout.log').open('wb') as out,(a.root/'stderr.log').open('wb') as err:
 child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));write(a.root/'actual_dispatch.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd));code=child.wait()
write(a.root/'natural_exit.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd,natural_exit=code))
src=a.root/'original_wrapper_source';src.mkdir()
for n in list(q['source_sha256'])+list(q['qualification_sha256'])+[a.plan.name]:shutil.copy2(bundle/n,src/n)
files={f.relative_to(a.root).as_posix():f for f in a.root.rglob('*') if f.is_file() and f.name!='peer_interrupted_original.zip'}
members={n:sha(f) for n,f in files.items()};write(a.root/'member_SHA.json',members)
archive=a.root/'complete_peer400_small_original.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for n,f in files.items():z.write(f,n)
 z.write(a.root/'member_SHA.json','member_SHA.json')
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
r=dict(status='INTERRUPTED_PEER400_CPU_COMPLETE' if code==0 else 'INTERRUPTED_PEER400_CPU_FAILED',actual_UTC=utc(),node=q['execution_node'],source_peer=q['source_peer'],child_PID=child.pid,natural_exit=code,archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,all_small_original_member_SHA_CRC_unique=True,complete_large_original_retained_on_peer=True)
write(a.root/'capture_receipt.json',r);print(json.dumps(r),flush=True);raise SystemExit(code)
