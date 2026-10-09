"""Fresh other-node environment gate and natural-exit small original CPU audit."""
import argparse,datetime,hashlib,importlib.metadata,os,shutil,subprocess,sys,zipfile
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def run(a):
 sys.path.insert(0,str(a.plan.parent))
 from common import read,write,utc,verify
 p=read(a.plan);assert sha(a.plan)==a.plan_sha;verify(p,a.plan.parent,a.assets)
 helper=a.plan.parent/'audit_public_small_stage_v1.py';assert sha(helper)==p['source_sha256'][helper.name]
 assert sys.executable==str(a.assets/'.venv/bin/python')
 assert {n:importlib.metadata.version(n) for n in p['runtime_versions']}==p['runtime_versions']
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 assert uuid==p['GPU_UUID']['C'] and not compute.strip()
 remaining=(datetime.datetime.fromisoformat(p['conservative_lease_end_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert remaining>1200+7200
 space=shutil.disk_usage(a.root.parent);assert space.free>=p['remote_free_floor_bytes'] and not a.root.exists()
 a.root.mkdir()
 cmd=[sys.executable,str(helper),'--stage',a.stage,'--uuid',uuid,'--publication',str(a.publication),'--capture',str(a.capture),'--received-archive',str(a.received_archive),'--out',str(a.root/'out')]
 if a.stage=='score':
  assert a.prediction_original
  cmd+=['--prediction-original',str(a.prediction_original)]
 write(a.root/'fresh_preflight.json',dict(actual_UTC=utc(),UUID=uuid,compute=compute,space=space._asdict(),remaining_seconds=remaining,fullargv=[sys.executable]+sys.argv,wrapper_SHA=sha(__file__),helper_SHA=sha(helper),qualification_plan_SHA=a.plan_sha,processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)))
 with (a.root/'stdout.log').open('wb') as out,(a.root/'stderr.log').open('wb') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));write(a.root/'dispatch.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd));code=child.wait()
 write(a.root/'natural_exit.json',dict(actual_UTC=utc(),pid=child.pid,natural_exit=code,fullargv=cmd))
 source=a.root/'original_source';source.mkdir()
 for n in list(p['source_sha256'])+list(p['posttrain_native_CPU_qualification'])+[a.plan.name]:shutil.copy2(a.plan.parent/n,source/n)
 shutil.copy2(__file__,source/Path(__file__).name)
 members={f.relative_to(a.root).as_posix():sha(f) for f in a.root.rglob('*') if f.is_file()};write(a.root/'member_SHA.json',members)
 archive=a.root/'complete_small_original.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n in list(members)+['member_SHA.json']:z.write(a.root/n,n)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
 assert archive.stat().st_size<=40_000_000
 receipt=dict(status='ACTUAL_C_SMALL_CPU_'+a.stage.upper()+('_COMPLETE' if code==0 else '_FAILED'),actual_UTC=utc(),child_PID=child.pid,natural_exit=code,archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,ZIP_CRC_unique_all_members=True,original_transport='verified_C_original_SFTP')
 write(a.root/'capture_receipt.json',receipt);print(receipt,flush=True);raise SystemExit(code)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prediction','score'],required=True)
 for n in ('plan','assets','root','publication','capture','received-archive'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--prediction-original',type=Path)
 p.add_argument('--plan-sha',required=True);run(p.parse_args())
