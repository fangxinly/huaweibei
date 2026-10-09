"""Capture CPU-only synthetic posttrain qualification; healthy GPU jobs continue."""
import argparse,datetime,hashlib,importlib.metadata,shutil,subprocess,sys,zipfile
from pathlib import Path
from common import sha,read,write,utc,verify

def run(a):
    p=read(a.plan); assert sha(a.plan)==a.plan_sha and p['native_only']
    bundle=a.plan.parent; verify(p,bundle,a.assets)
    assert sys.executable==str(a.assets/'.venv/bin/python')
    versions={n:importlib.metadata.version(n) for n in p['runtime_versions']}
    assert versions==p['runtime_versions']
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    assert uuid==p['GPU_UUID'][a.node]
    assert shutil.disk_usage(a.root.parent).free>500_000_000
    remaining=(datetime.datetime.fromisoformat(p['conservative_lease_end_UTC'])-datetime.datetime.now(datetime.timezone.utc)).total_seconds()
    assert remaining>7500
    a.root.mkdir()
    write(a.root/'preflight.json',dict(actual_UTC=utc(),UUID=uuid,fullargv=[sys.executable]+sys.argv,versions=versions,free_bytes=shutil.disk_usage(a.root).free,remaining_seconds=remaining,CPU_only=True,healthy_GPU_compute_recorded_not_stopped=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)))
    cmd=[sys.executable,str(bundle/'qualify_candidate_posttrain_v1.py'),'--out',str(a.root/'native_result.json')]
    with (a.root/'stdout.log').open('wb') as out,(a.root/'stderr.log').open('wb') as err:
        child=subprocess.Popen(cmd,stdout=out,stderr=err)
        write(a.root/'dispatch.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd))
        code=child.wait()
    write(a.root/'natural_exit.json',dict(actual_UTC=utc(),pid=child.pid,fullargv=cmd,natural_exit=code))
    source=a.root/'original_source'; source.mkdir()
    for name in list(p['source_sha256'])+[a.plan.name]:shutil.copy2(bundle/name,source/name)
    members={f.relative_to(a.root).as_posix():sha(f) for f in a.root.rglob('*') if f.is_file()}
    write(a.root/'member_SHA.json',members)
    archive=a.root/'complete_small_original.zip'
    with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
        for name in list(members)+['member_SHA.json']:z.write(a.root/name,name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        for name,digest in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    write(a.root/'capture_receipt.json',dict(status='CANDIDATE_POSTTRAIN_NATIVE_COMPLETE' if code==0 else 'CANDIDATE_POSTTRAIN_NATIVE_FAILED',actual_UTC=utc(),natural_exit=code,child_PID=child.pid,archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,ZIP_CRC_unique_all_members=True,CPU_only=True,no_real_VAL_TEST_scores=True))
    print((a.root/'capture_receipt.json').read_text(),flush=True)
    raise SystemExit(code)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('plan','assets','root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--plan-sha',required=True);p.add_argument('--node',choices=['A','B'],required=True);run(p.parse_args())
