"""Actual two-node synthetic qualification; no true datasets or task weights."""
import argparse,datetime,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf8')

def run(a):
 assert sha(a.plan)==a.plan_sha;p=json.loads(a.plan.read_text());bundle=a.plan.parent
 for n,h in p['source_sha256'].items():assert sha(bundle/n)==h,n
 assert not a.root.exists();a.root.mkdir()
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 assert uuid==p['GPU_UUID'][a.node] and not compute.strip()
 assert sys.executable==str(a.assets/'.venv/bin/python')
 assert {n:importlib.metadata.version(n) for n in p['runtime_versions']}==p['runtime_versions']
 assert shutil.disk_usage(a.root).free>=p['remote_free_floor_bytes']
 remaining=(datetime.datetime.fromisoformat(p['conservative_lease_end_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert remaining>7200+1200
 for n,h in p['asset_sha256'].items():assert sha(a.assets/n)==h,n
 write(a.root/'fresh_preflight.json',dict(actual_UTC=utc(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,wrapper_SHA=sha(__file__),plan_SHA=a.plan_sha,UUID=uuid,compute=compute,remaining_seconds=remaining,space=shutil.disk_usage(a.root)._asdict(),source_assets_versions_passed=True,processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)))
 cmd=[sys.executable,str(bundle/'qualify_resume_official_v1.py'),'--bundle',str(bundle),'--out',str(a.root/'out')]
 with (a.root/'stdout.log').open('wb') as out,(a.root/'stderr.log').open('wb') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));write(a.root/'actual_dispatch.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd));code=child.wait()
 write(a.root/'natural_exit.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd,natural_exit=code))
 src=a.root/'original_source';src.mkdir()
 for n in list(p['source_sha256'])+[a.plan.name]:shutil.copy2(bundle/n,src/n)
 members={f.relative_to(a.root).as_posix():sha(f) for f in a.root.rglob('*') if f.is_file()};write(a.root/'member_SHA.json',members)
 archive=a.root/'complete_native_original.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n in list(members)+['member_SHA.json']:z.write(a.root/n,n)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 r=dict(status='CANDIDATE_NATIVE_COMPLETE' if code==0 else 'CANDIDATE_NATIVE_FAILED',actual_UTC=utc(),node=a.node,child_PID=child.pid,natural_exit=code,archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,ZIP_CRC_unique_all_members=True,real_TRAIN=False)
 write(a.root/'capture_receipt.json',r);print(json.dumps(r),flush=True);raise SystemExit(code)

if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ('plan','assets','root'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=['A','B'],required=True);run(p.parse_args())
