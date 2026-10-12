"""Fresh native preflight, natural exit, full original prefix-state capture."""
import argparse,datetime,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,r):Path(p).write_text(json.dumps(r,indent=2),encoding='utf8')

def run(a):
 p=json.loads(a.plan.read_text());assert sha(a.plan)==a.plan_sha
 assert p['storage_transport_qualified'] and p['health_v2_native_qualified'] and p['precheck_steps']==16
 bundle=a.plan.parent;assert not a.root.exists();a.root.mkdir()
 assert sys.executable==str(a.assets/'.venv/bin/python')
 for n,h in p['source_sha256'].items():assert sha(bundle/n)==h,n
 for n,h in p['qualification_sha256'].items():assert sha(bundle/n)==h,n
 for node in ('A','B'):
  r=json.loads((bundle/(node+'_native_result.json')).read_text());e=json.loads((bundle/(node+'_native_exit.json')).read_text())
  assert r['UUID']==p['GPU_UUID'][node] and e['natural_exit']==0 and r['first16_same_parameter_trajectory'] and r['rounded_pure_decay_control_passed']
  assert r['helper_SHA']==p['source_sha256']['training_health_components_v2.py']
 remote=json.loads((bundle/'B_release_transport_result.json').read_text());assert remote['status']=='OTHER_NODE_PUBLIC_RELEASE_ORIGINAL_SHA_CRC_MEMBERS_VERIFIED'
 assert not p['local_D_capacity_claimed'] and p['latest_human_autonomous_storage_choice_recorded']
 uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
 assert uuid==p['GPU_UUID'][p['execution_node']] and not compute.strip()
 versions={n:importlib.metadata.version(n) for n in p['runtime_versions']};assert versions==p['runtime_versions']
 assert shutil.disk_usage(a.root).free>=p['remote_free_floor_bytes']
 remaining=(datetime.datetime.fromisoformat(p['conservative_lease_end_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert remaining>p['precheck_budget_seconds']+7200
 for n,h in p['asset_sha256'].items():assert sha(a.assets/n)==h,n
 cmd=[sys.executable,str(bundle/'precheck16.py'),'--plan',str(a.plan),'--plan-sha',a.plan_sha,'--bundle',str(bundle),'--assets',str(a.assets),'--out',str(a.root/'out')]
 write(a.root/'fresh_preflight.json',dict(actual_UTC=utc(),UUID=uuid,compute=compute,fullargv=[sys.executable]+sys.argv,pid=os.getpid(),source_SHA=sha(__file__),space=shutil.disk_usage(a.root)._asdict(),runtime_versions=versions,remaining_seconds=remaining,processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)))
 with (a.root/'stdout.log').open('wb') as out,(a.root/'stderr.log').open('wb') as err:
  child=subprocess.Popen(cmd,stdout=out,stderr=err,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
  write(a.root/'actual_dispatch.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd));code=child.wait()
 write(a.root/'natural_exit.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd,natural_exit=code))
 src=a.root/'original_source';src.mkdir()
 for n in list(p['source_sha256'])+list(p['qualification_sha256'])+[a.plan.name]:shutil.copy2(bundle/n,src/n)
 members={f.relative_to(a.root).as_posix():sha(f) for f in a.root.rglob('*') if f.is_file()};write(a.root/'member_SHA.json',members)
 archive=a.root/'complete_actual_prefix16.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED,1) as z:
  for n in list(members)+['member_SHA.json']:z.write(a.root/n,n)
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
  for n,h in members.items():
   d=hashlib.sha256()
   with z.open(n) as f:
    for block in iter(lambda:f.read(8*1024**2),b''):d.update(block)
   assert d.hexdigest()==h
 r=dict(status='ACTUAL_PREFIX16_COMPLETE' if code==0 else 'ACTUAL_PREFIX16_FAILED',actual_UTC=utc(),child_PID=child.pid,natural_exit=code,archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,ZIP_CRC_unique_all_members=True,no_D_capacity_claimed=True)
 write(a.root/'capture_receipt.json',r);print(json.dumps(r),flush=True);raise SystemExit(code)

if __name__=='__main__':
 parser=argparse.ArgumentParser()
 for n in ('plan','assets','root'):parser.add_argument('--'+n,type=Path,required=True)
 parser.add_argument('--plan-sha',required=True);run(parser.parse_args())
