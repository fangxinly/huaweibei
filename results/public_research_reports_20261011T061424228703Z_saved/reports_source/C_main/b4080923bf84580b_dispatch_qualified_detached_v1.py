"""Launch a pinned capture wrapper independently of the SSH controlling session."""
import argparse,datetime,hashlib,json,os,subprocess,sys,zipfile
from pathlib import Path
def sha(f):
 h=hashlib.sha256()
 with Path(f).open('rb') as z:
  for b in iter(lambda:z.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
p=argparse.ArgumentParser()
for n in ('zip','source','assets','root'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--zip-sha',required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--resume',type=Path);p.add_argument('--kind',choices=['peer400','recovery400'],required=True);a=p.parse_args()
assert sha(a.zip)==a.zip_sha and not a.source.exists() and not a.root.exists()
a.source.mkdir()
with zipfile.ZipFile(a.zip) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n in z.namelist():
  target=a.source/n;assert target.resolve().is_relative_to(a.source.resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
plan=a.source/('plan.json' if a.kind=='peer400' else 'qualified_resume_plan.json');q=json.loads(plan.read_text());assert sha(plan)==a.plan_sha
for n,h in q['source_sha256'].items():assert sha(a.source/n)==h,n
for n,h in q['qualification_sha256'].items():assert sha(a.source/n)==h,n
executable=a.assets/'.venv/bin/python';wrapper='run_interrupted_peer400_capture_v1.py' if a.kind=='peer400' else 'run_resumed100_capture_v1.py'
cmd=[str(executable),str(a.source/wrapper),'--plan',str(plan),'--plan-sha',a.plan_sha,'--assets',str(a.assets),'--root',str(a.root)]
if a.kind=='recovery400':
 assert a.resume is not None and sha(a.resume)==q['resume_checkpoint_SHA'];cmd+=['--resume',str(a.resume)]
with Path(str(a.root)+'_launcher.log').open('xb') as out,Path(str(a.root)+'_launcher.err').open('xb') as err:
 child=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
r=dict(status='DETACHED_CAPTURE_WRAPPER_DISPATCHED_NOT_COMPLETED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),wrapper_pid=child.pid,fullargv=cmd,launcher_pid=os.getpid(),launcher_fullargv=[sys.executable]+sys.argv,launcher_SHA=sha(Path(__file__)),source_ZIP_SHA=a.zip_sha,plan_SHA=a.plan_sha,detached_session=True,stdin_DEVNULL=True,kind=a.kind)
Path(str(a.root)+'_detached_dispatch.json').write_text(json.dumps(r,indent=2),encoding='utf8');print(json.dumps(r),flush=True)
