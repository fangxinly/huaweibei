"""New isolated CPU dependency environment; no models/data/GPU forwards."""
import argparse,datetime,hashlib,importlib.metadata,json,os,shutil,subprocess,sys,zipfile
from pathlib import Path

def sha(f):
    h=hashlib.sha256()
    with Path(f).open('rb') as inp:
        for b in iter(lambda:inp.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(f,d):f.write_text(json.dumps(d,indent=2),encoding='utf8')
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--requirements',type=Path,required=True);p.add_argument('--requirements-sha',required=True);a=p.parse_args()
assert sha(a.requirements)==a.requirements_sha and not a.root.exists()
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
assert uuid=='GPU-daa2c09a-4ce5-26dd-b375-6b61242c6795' and not compute.strip()
assert shutil.disk_usage(a.root.parent).free>12_000_000_000
assert (datetime.datetime.fromisoformat('2026-10-09T15:00:00+00:00')-datetime.datetime.now(datetime.timezone.utc)).total_seconds()>7500
a.root.mkdir();shutil.copy2(a.requirements,a.root/'requirements.txt');shutil.copy2(__file__,a.root/'setup_source.py')
write(a.root/'preflight.json',dict(actual_UTC=utc(),UUID=uuid,compute=compute,fullargv=[sys.executable]+sys.argv,source_SHA=sha(__file__),requirements_SHA=a.requirements_sha,space=shutil.disk_usage(a.root)._asdict(),CPU_only=True,no_data_models_or_labels=True))
commands=[[sys.executable,'-m','venv','--system-site-packages',str(a.root/'.venv')],[str(a.root/'.venv/bin/python'),'-m','pip','install','--no-cache-dir','--disable-pip-version-check','-r',str(a.root/'requirements.txt')]]
code=0;children=[]
with (a.root/'stdout.log').open('wb') as out,(a.root/'stderr.log').open('wb') as err:
    for cmd in commands:
        child=subprocess.Popen(cmd,stdout=out,stderr=err);code=child.wait();children.append(dict(pid=child.pid,fullargv=cmd,natural_exit=code))
        if code:break
write(a.root/'natural_exit.json',dict(actual_UTC=utc(),natural_exit=code,children=children))
if code==0:
    expected=dict(line.strip().split('==',1) for line in (a.root/'requirements.txt').read_text().splitlines() if line.strip())
    cmd=[str(a.root/'.venv/bin/python'),'-c','import importlib.metadata,json,sys; names=json.loads(sys.argv[1]); print(json.dumps({n:importlib.metadata.version(n) for n in names}))',json.dumps(list(expected))]
    actual=json.loads(subprocess.check_output(cmd,text=True));assert actual==expected,(actual,expected)
    write(a.root/'runtime_result.json',dict(status='NEW_ISOLATED_CANDIDATE_PURE_CPU_RUNTIME_PINNED_PASSED',actual_UTC=utc(),runtime_python=str(a.root/'.venv/bin/python'),runtime_versions=actual,UUID=uuid,CPU_only=True,GPU_forward=False,model_data_download=False,old_environment_modified=False,free_bytes=shutil.disk_usage(a.root).free))
files=[f for f in a.root.iterdir() if f.is_file()];members={f.name:sha(f) for f in files};write(a.root/'member_SHA.json',members)
archive=a.root/'complete_small_original.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for f in files+[a.root/'member_SHA.json']:z.write(f,f.name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,digest in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
write(a.root/'capture_receipt.json',dict(actual_UTC=utc(),natural_exit=code,archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,ZIP_CRC_unique_all_members=True,CPU_only=True))
print((a.root/'capture_receipt.json').read_text(),flush=True)
raise SystemExit(code)
