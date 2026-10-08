"""Fresh physical guard and natural-exit complete original capture."""
import argparse,datetime,hashlib,importlib.metadata,os,shutil,subprocess,sys,zipfile
from pathlib import Path
from common import read,write,sha,utc
def run(a):
    assert sha(a.plan)==a.plan_sha;p=read(a.plan);bundle=a.plan.parent;assert a.stage in p['allowed_stages'];assert not a.root.exists();a.root.mkdir()
    for n,h in p['source_sha256'].items():assert sha(bundle/n)==h,n
    assert sys.executable==str(a.assets/'.venv/bin/python');versions={n:importlib.metadata.version(n) for n in p['runtime_versions']};assert versions==p['runtime_versions']
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True);processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)
    assert uuid==p['GPU_UUID'][a.node] and not compute.strip();assert shutil.disk_usage(a.root).free>=p['remote_space_floor_bytes'];assert (datetime.datetime.fromisoformat(p['conservative_lease_end_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()>p['stage_budget_seconds'][a.stage]+7200
    write(a.root/'actual_preflight.json',dict(actual_UTC=utc(),UUID=uuid,compute=compute,processes=processes,pid=os.getpid(),fullargv=[sys.executable]+sys.argv,runtime_versions=versions,remote_space=shutil.disk_usage(a.root)._asdict(),lease_end_is_conservative_not_platform_confirmed=True))
    for n,h in p['asset_sha256'].items():assert sha(a.assets/n)==h,n
    cmd=[sys.executable]
    if a.stage=='native':cmd+=[str(bundle/'native_check.py'),'--out',str(a.root/'out')]+(['--gpu'] if a.node=='A' else [])
    else:
        cmd+=[str(bundle/({'cache':'cache_original.py','train':'train_increment.py','audit':'audit_increment.py'}[a.stage])),'--plan',str(a.plan),'--plan-sha',a.plan_sha,'--bundle',str(bundle),'--assets',str(a.assets),'--out',str(a.root/'out')]
        if a.stage=='cache':
            assert p['parent_D_CPU_joint_reference'];cmd+=['--parent',p['parent']['remote_path']]
        else:
            assert a.cache and a.tail;cmd+=['--cache',str(a.cache),'--tail',str(a.tail)]
            if a.stage=='train':
                assert p['real_train_enabled'] and p['native_D_other_CPU_reference'] and p['cache_D_reference'];assert sha(a.cache)==p['cache_reference']['SHA'] and sha(a.tail)==p['tail_reference']['SHA']
                fd=os.open(p['single_training_launch_token'],os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,a.plan_sha.encode());os.close(fd)
            else:assert a.state and a.prediction;cmd+=['--state',str(a.state),'--prediction',str(a.prediction)]
    write(a.root/'actual_dispatch.json',dict(actual_UTC=utc(),fullargv=cmd))
    with (a.root/'stdout.log').open('wb') as o,(a.root/'stderr.log').open('wb') as e:
        child=subprocess.Popen(cmd,stdout=o,stderr=e,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'));code=child.wait()
    write(a.root/'natural_exit.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd,natural_exit=code,plan_SHA=a.plan_sha))
    source=a.root/'original_source';source.mkdir()
    for name in list(p['source_sha256'])+[a.plan.name]:shutil.copy2(bundle/name,source/name)
    members={f.relative_to(a.root).as_posix():sha(f) for f in a.root.rglob('*') if f.is_file()};write(a.root/'member_SHA.json',members);zp=a.root/'complete_actual_capture.zip'
    with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
        for name in list(members)+['member_SHA.json']:z.write(a.root/name,name)
    with zipfile.ZipFile(zp) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for name,h in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
    receipt=dict(status='ACTUAL_COMPLETE' if code==0 else 'ACTUAL_FAILED',actual_UTC=utc(),stage=a.stage,node=a.node,child_natural_exit=code,child_PID=child.pid,archive=str(zp),archive_SHA=sha(zp),archive_bytes=zp.stat().st_size,CRC_pass=True,all_member_SHA=True,unique=True)
    write(a.root/'actual_capture_receipt.json',receipt);print(receipt,flush=True);raise SystemExit(code)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['native','cache','train','audit'],required=True)
    for n in ('plan','assets','root'):p.add_argument('--'+n,type=Path,required=True)
    for n in ('cache','tail','state','prediction'):p.add_argument('--'+n,type=Path)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=['A','B'],required=True);run(p.parse_args())
