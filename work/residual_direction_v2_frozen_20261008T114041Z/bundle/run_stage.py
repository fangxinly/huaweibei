"""Fresh genuine node execution, natural child exit, complete split archive."""
import argparse,datetime,hashlib,importlib.metadata,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from common import sha,read,write,utc,verify

def run(a):
    p=read(a.plan);assert sha(a.plan)==a.plan_sha and a.stage in p['allowed_stages'];bundle=a.plan.parent
    assert not a.root.exists();a.root.mkdir();assert sys.executable==str(a.assets/'.venv/bin/python')
    versions={n:importlib.metadata.version(n) for n in p['runtime_versions']};assert versions==p['runtime_versions']
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)
    assert uuid==p['GPU_UUID'][a.node] and not compute.strip();assert shutil.disk_usage(a.root).free>=p['remote_free_floor_bytes']
    remaining=(datetime.datetime.fromisoformat(p['conservative_lease_end_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds();assert remaining>p['stage_budget_seconds'][a.stage]+7200
    write(a.root/'actual_preflight.json',dict(actual_UTC=utc(),UUID=uuid,compute=compute,processes=processes,pid=os.getpid(),fullargv=[sys.executable]+sys.argv,runtime_versions=versions,remote_space=shutil.disk_usage(a.root)._asdict(),remaining_seconds=remaining,lease_end_is_conservative_not_platform_confirmed=True))
    verify(p,bundle,a.assets)
    script=dict(native='native_contract.py',train='train_official.py',audit='audit_official.py',infer='infer_official.py',score='score_official.py')[a.stage]
    cmd=[sys.executable,str(bundle/script)]
    if a.stage!='native':cmd+=['--plan',str(a.plan),'--plan-sha',a.plan_sha,'--bundle',str(bundle),'--assets',str(a.assets),'--out',str(a.root/'out')]
    if a.stage in ('audit','infer'):assert a.input_root;cmd+=['--input-root',str(a.input_root)]
    if a.stage=='score':assert a.prediction;cmd+=['--prediction',str(a.prediction)]
    if a.stage=='train':
        assert p['native_CPU_D_qualification'];fd=os.open(p['train_once_token'],os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,a.plan_sha.encode());os.close(fd)
    write(a.root/'actual_dispatch.json',dict(actual_UTC=utc(),fullargv=cmd))
    with (a.root/'stdout.log').open('wb') as o,(a.root/'stderr.log').open('wb') as e:
        child=subprocess.Popen(cmd,stdout=o,stderr=e,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));code=child.wait()
    write(a.root/'natural_exit.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd,natural_exit=code,plan_SHA=a.plan_sha))
    source=a.root/'original_source';source.mkdir()
    for n in list(p['source_sha256'])+[a.plan.name]:shutil.copy2(bundle/n,source/n)
    members={f.relative_to(a.root).as_posix():sha(f) for f in a.root.rglob('*') if f.is_file()};write(a.root/'member_SHA.json',members)
    zp=a.root/'complete_actual_capture.zip'
    with zipfile.ZipFile(zp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1,allowZip64=True) as z:
        for n in list(members)+['member_SHA.json']:z.write(a.root/n,n)
    assert zp.stat().st_size<=p['complete_archive_bytes_ceiling']
    with zipfile.ZipFile(zp) as z:
        assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
        for n,h in members.items():
            digest=hashlib.sha256()
            with z.open(n) as f:
                for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
            assert digest.hexdigest()==h
    parts=[]
    with zp.open('rb') as f:
        i=0
        while True:
            data=f.read(p['archive_part_bytes'])
            if not data:break
            part=a.root/('capture.part%02d'%i);part.write_bytes(data);parts.append(dict(name=part.name,bytes=len(data),SHA=sha(part)));i+=1
    r=dict(status='ACTUAL_COMPLETE' if code==0 else 'ACTUAL_FAILED',actual_UTC=utc(),stage=a.stage,node=a.node,child_natural_exit=code,child_PID=child.pid,archive=str(zp),archive_SHA=sha(zp),archive_bytes=zp.stat().st_size,parts=parts,CRC_pass=True,all_member_SHA=True,unique=True)
    write(a.root/'actual_capture_receipt.json',r);print(r,flush=True);raise SystemExit(code)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['native','train','audit','infer','score'],required=True)
    for n in ('plan','assets','root'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('input-root','prediction'):p.add_argument('--'+n,type=Path)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=['A','B'],required=True);run(p.parse_args())
