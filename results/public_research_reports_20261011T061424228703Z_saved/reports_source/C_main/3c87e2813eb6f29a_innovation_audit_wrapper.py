"""Actual fresh CPU audit process and immutable original capture."""
import argparse,datetime,hashlib,importlib.metadata,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from contract import read,write,sha
def utc():return datetime.datetime.now(datetime.timezone.utc)
def run(a):
    assert sha(a.plan)==a.plan_sha;p=read(a.plan);assert not a.root.exists();a.root.mkdir()
    for name,h in p['source_sha256'].items():assert sha(a.plan.parent/name)==h
    assert sys.executable==str(a.assets/'.venv/bin/python');versions={n:importlib.metadata.version(n) for n in p['runtime_versions']};assert versions==p['runtime_versions']
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)
    assert uuid==p['GPU_UUID']['B'] and not compute.strip();assert shutil.disk_usage(a.root).free>p['remote_space_floor_bytes']
    assert (datetime.datetime.fromisoformat(p['conservative_lease_end_UTC'])-utc()).total_seconds()>300+7200
    assert sha(a.train)==p['train_archive_SHA'] and sha(a.cache)==p['cache_archive_SHA']
    write(a.root/'actual_preflight.json',dict(actual_UTC=utc().isoformat(),UUID=uuid,compute=compute,processes=processes,fullargv=[sys.executable]+sys.argv,pid=os.getpid(),runtime_versions=versions,remote_space=shutil.disk_usage(a.root)._asdict()))
    cmd=[sys.executable,str(a.plan.parent/'innovation_cpu_audit.py'),'--plan',str(a.plan),'--plan-sha',a.plan_sha,'--train',str(a.train),'--cache',str(a.cache),'--out',str(a.root/'out')]
    write(a.root/'actual_dispatch.json',dict(actual_UTC=utc().isoformat(),fullargv=cmd))
    with (a.root/'stdout.log').open('wb') as o,(a.root/'stderr.log').open('wb') as e:
        child=subprocess.Popen(cmd,stdout=o,stderr=e,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));code=child.wait()
    write(a.root/'natural_exit.json',dict(actual_UTC=utc().isoformat(),pid=child.pid,fullargv=cmd,natural_exit=code,plan_SHA=a.plan_sha))
    source=a.root/'original_source';source.mkdir()
    for name in list(p['source_sha256'])+[a.plan.name]:shutil.copy2(a.plan.parent/name,source/name)
    members={f.relative_to(a.root).as_posix():sha(f) for f in a.root.rglob('*') if f.is_file()};write(a.root/'member_SHA.json',members);zp=a.root/'complete_actual_capture.zip'
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
        for name in list(members)+['member_SHA.json']:z.write(a.root/name,name)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for name,h in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    r=dict(status='ACTUAL_COMPLETE' if code==0 else 'ACTUAL_FAILED',actual_UTC=utc().isoformat(),node='B',stage='audit',child_natural_exit=code,child_PID=child.pid,archive=str(zp),archive_SHA=sha(zp),archive_bytes=zp.stat().st_size,CRC_pass=True,all_member_SHA=True,unique=True)
    write(a.root/'actual_capture_receipt.json',r);print(r);raise SystemExit(code)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('plan','assets','train','cache','root'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);run(p.parse_args())
