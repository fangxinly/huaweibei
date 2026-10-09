"""Capture each real local child, its natural exit and all original small artifacts."""
import argparse, datetime, hashlib, json, os, platform, shutil, subprocess, sys, zipfile
from pathlib import Path
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
p=argparse.ArgumentParser();p.add_argument('stage',choices=['qualify','score','peer']);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
base=Path(__file__).resolve().parent;ev=base.parent;runtime=base/'runtime/Scripts/python.exe';runner=base/'local_official_score_v1.py';plan=base/'local_official_protocol.json';old=ev/'A_score_qualified_20261009T082953Z/official_score_protocol.json'
assert shutil.disk_usage('D:/').free>40*1024**2+100*1024**2;assert shutil.disk_usage('C:/').free>200*1024**2
a.root.mkdir();out=a.root/'out';argv=[str(runtime),'-B',str(runner),a.stage,'--plan',str(plan)]
if a.stage=='qualify':argv+=['--old-plan',str(old),'--old-plan-sha',sha(old),'--dataset','C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen/work/careflow_data/mosi.pkl','--prediction',str(ev/'A100_fixed_official_VAL_TEST_prediction.npz'),'--metric-source',str(base/'sentiment_metrics_careflow_v1.py'),'--evidence',str(ev)]
else:argv+=['--plan-sha',sha(plan),'--out',str(out)]
if a.stage=='peer':argv+=['--result',str(ev/'A_local_official_score_actual_20261009T145600Z/out/official_aligned_five_result.json')]
before=utc();write(a.root/'dispatch_before.json',dict(actual_UTC=before,fullargv=argv,host=platform.node(),no_remote_execution=True,not_old_remote_stage_capture=True,runner_SHA=sha(runner),capture_source_SHA=sha(__file__)))
with (a.root/'stdout.log').open('wb') as stdout,(a.root/'stderr.log').open('wb') as stderr:
    child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,stdin=subprocess.DEVNULL);write(a.root/'dispatch_actual.json',dict(actual_UTC=utc(),child_PID=child.pid,fullargv=argv));code=child.wait()
write(a.root/'natural_exit.json',dict(actual_UTC=utc(),child_PID=child.pid,natural_exit=code,fullargv=argv))
original=a.root/'original_source';original.mkdir()
for f in [runner,Path(__file__),base/'sentiment_metrics_careflow_v1.py',old]+([plan] if plan.exists() else []):shutil.copyfile(f,original/f.name)
if code==0 and a.stage=='qualify':shutil.copyfile(plan,a.root/'qualification_result.json')
write(a.root/'runtime_receipt.json',dict(actual_UTC=utc(),python=str(runtime),python_binary_SHA=sha(runtime),qualification_numpy='1.26.4',host=platform.node(),only_CPU_arrays=True))
files=[f for f in a.root.rglob('*') if f.is_file()];records=[dict(path=f.relative_to(a.root).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in files];manifest=a.root/'member_manifest.json';write(manifest,dict(status='ACTUAL_LOCAL_STAGE_ORIGINAL_MEMBER_MANIFEST',stage=a.stage,natural_exit=code,records=records));archive=a.root/'complete_actual_local_original.zip'
with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for f in files+[manifest]:z.write(f,f.relative_to(a.root).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None;assert len(z.namelist())==len(set(z.namelist()))==len(records)+1
    for v in records:assert hashlib.sha256(z.read(v['path'])).hexdigest()==v['sha256']
write(a.root/'capture_receipt.json',dict(status='ACTUAL_LOCAL_'+a.stage.upper()+'_ORIGINALS_CAPTURED',actual_UTC=utc(),natural_exit=code,child_PID=child.pid,archive=str(archive),archive_bytes=archive.stat().st_size,archive_SHA=sha(archive),all_member_SHA_CRC_unique_passed=True,not_remote_capture=True,other_node_metric_recalculation_pending=True))
print(json.dumps(dict(stage=a.stage,natural_exit=code,archive_SHA=sha(archive),root=str(a.root))));sys.exit(code)
