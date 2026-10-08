"""Fresh original process/capture wrapper; source, GPU, budget and space guards."""
import argparse,datetime,hashlib,importlib.metadata,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from contract import read,write,sha
def utc():return datetime.datetime.now(datetime.timezone.utc)
def run(a):
    assert sha(a.plan)==a.plan_sha;p=read(a.plan);assert a.stage in p['allowed_stages'];assert not a.root.exists();a.root.mkdir()
    bundle=a.plan.parent
    for name,h in p['source_sha256'].items():assert sha(bundle/name)==h,name
    assert sys.executable==str(a.assets/'.venv/bin/python');versions={n:importlib.metadata.version(n) for n in p['runtime_versions']};assert versions==p['runtime_versions']
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)
    assert uuid==p['GPU_UUID'][a.node] and not compute.strip();assert shutil.disk_usage(a.root).free>=p['remote_space_floor_bytes']
    assert (datetime.datetime.fromisoformat(p['conservative_lease_end_UTC'])-utc()).total_seconds()>p['stage_budget_seconds'][a.stage]+7200
    write(a.root/'actual_preflight.json',dict(actual_UTC=utc().isoformat(),UUID=uuid,compute=compute,processes=processes,pid=os.getpid(),fullargv=[sys.executable]+sys.argv,runtime_versions=versions,remote_space=shutil.disk_usage(a.root)._asdict(),lease_end_is_conservative_not_platform_confirmed=True))
    for name,h in p['asset_sha256'].items():assert sha(a.assets/name)==h,name
    if a.stage=='native':cmd=[sys.executable,str(bundle/'native_check.py'),'--out',str(a.root/'out')]
    elif a.stage=='cache':cmd=[sys.executable,str(bundle/'cache_features.py'),'--plan',str(a.plan),'--plan-sha',a.plan_sha,'--bundle',str(bundle),'--assets',str(a.assets),'--out',str(a.root/'out')]
    elif a.stage=='train':
        assert p['real_train_enabled'] and a.cache;assert p['native_D_and_other_CPU_reference'] and p['cache_D_reference'];assert sha(a.cache)==p['cache_reference']['SHA']
        cmd=[sys.executable,str(bundle/'train_pilot.py'),'--plan',str(a.plan),'--plan-sha',a.plan_sha,'--bundle',str(bundle),'--assets',str(a.assets),'--out',str(a.root/'out'),'--cache',str(a.cache)]
    write(a.root/'actual_dispatch.json',dict(actual_UTC=utc().isoformat(),fullargv=cmd))
    with (a.root/'stdout.log').open('wb') as o,(a.root/'stderr.log').open('wb') as e:
        child=subprocess.Popen(cmd,stdout=o,stderr=e,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));code=child.wait()
    write(a.root/'natural_exit.json',dict(actual_UTC=utc().isoformat(),pid=child.pid,fullargv=cmd,natural_exit=code,plan_SHA=a.plan_sha))
    source=a.root/'original_source';source.mkdir()
    for name in list(p['source_sha256'])+[a.plan.name]:shutil.copy2(bundle/name,source/name)
    members={f.relative_to(a.root).as_posix():sha(f) for f in a.root.rglob('*') if f.is_file()};write(a.root/'member_SHA.json',members);zp=a.root/'complete_actual_capture.zip'
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
        for name in list(members)+['member_SHA.json']:z.write(a.root/name,name)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for name,h in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    receipt=dict(status='ACTUAL_COMPLETE' if code==0 else 'ACTUAL_FAILED',actual_UTC=utc().isoformat(),stage=a.stage,node=a.node,child_natural_exit=code,child_PID=child.pid,archive=str(zp),archive_SHA=sha(zp),archive_bytes=zp.stat().st_size,CRC_pass=True,all_member_SHA=True,unique=True)
    write(a.root/'actual_capture_receipt.json',receipt);print(receipt);raise SystemExit(code)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['native','cache','train'],required=True)
    for name in ('plan','assets','root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=['A','B','C'],required=True);p.add_argument('--cache',type=Path);run(p.parse_args())
