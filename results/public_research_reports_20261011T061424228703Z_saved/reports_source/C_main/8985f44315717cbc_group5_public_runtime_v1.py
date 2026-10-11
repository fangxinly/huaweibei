"""Restore exact public assets and existing offline wheels in a fresh private venv.

No task weights or pickle values are loaded, no network package installation,
no global-environment modification, no training/inference/score dispatch.
"""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(p,o):Path(p).write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def utc():return dt.datetime.now(dt.timezone.utc)


def safe(root,name):
    p=(root/name).resolve()
    if '\\' in name or ':' in name or not p.is_relative_to(root.resolve()):raise ValueError('Unsafe archive member')
    return p


def fetch(spec,path):
    if path.exists():raise FileExistsError('Fresh public download path required')
    h=hashlib.sha256();n=0
    with path.open('xb') as out:
        for part in spec['parts']:
            if not part['url'].startswith('https://github.com/fangxinly/huaweibei/releases/download/'):
                raise PermissionError('Unexpected public archive location')
            ph=hashlib.sha256();size=0
            with urllib.request.urlopen(urllib.request.Request(part['url'],headers={'User-Agent':'Group5-public-runtime'}),timeout=180) as stream:
                for b in iter(lambda:stream.read(8*1024**2),b''):
                    size+=len(b);n+=len(b)
                    if size>part['bytes']:raise ValueError('Public range too large')
                    ph.update(b);h.update(b);out.write(b)
            if size!=part['bytes'] or ph.hexdigest()!=part['digest'].removeprefix('sha256:'):
                raise ValueError('Public range SHA/length differs')
    if n!=spec['bytes'] or h.hexdigest()!=spec['whole_SHA']:raise ValueError('Whole public archive SHA/length differs')


def run(a):
    if sha(a.plan)!=a.plan_sha:raise PermissionError('Frozen public-restore plan differs')
    p=json.loads(a.plan.read_bytes())
    if p['status']!='GROUP5_BATCH5_PUBLIC_RUNTIME_ONLY_FROZEN' or sha(__file__)!=p['source_SHA']:
        raise PermissionError('Preparation cannot restore a native runtime')
    if a.root.exists():raise FileExistsError('Fresh isolated runtime root required')
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True)
    meminfo=Path('/proc/meminfo').read_text();available=int(next(s.split()[1] for s in meminfo.splitlines() if s.startswith('MemAvailable:')))*1024
    if gpu!=p['GPU_UUID'] or compute.strip() or available<6*1024**3 or shutil.disk_usage(a.root.parent).free<8*1024**3:
        raise PermissionError('Fresh UUID/compute/RAM/space gate failed')
    if (dt.datetime.fromisoformat(p['lease_end_UTC'])-utc()).total_seconds()<p['stage_budget_seconds']+7200:
        raise PermissionError('Runtime setup plus two-hour saving margin exceeds lease')
    if p['local_C_free_bytes']<200*1024**2 or p['local_D_free_bytes']<40*1024**2 or not 0<=(utc()-dt.datetime.fromisoformat(p['local_space_capture_UTC'])).total_seconds()<=300:
        raise PermissionError('Fresh local preservation floors failed')
    a.root.mkdir();write(a.root/'fresh_physical.json',dict(actual_UTC=utc().isoformat(),GPU_UUID=gpu,compute=compute,meminfo=meminfo,
        fullargv=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True),remote_free=shutil.disk_usage(a.root).free,argv=[sys.executable]+sys.argv))
    assets=a.root/'public';assets.mkdir()
    common=a.root/'public_original_assets.zip';fetch(p['common'],common)
    with zipfile.ZipFile(common) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(p['common_member_SHA'])|{'asset_manifest.json'}:
            raise ValueError('Common original unique/exact member set differs')
        m=json.loads(z.read('asset_manifest.json'))
        if {r['archive']:r['sha256'] for r in m['files']}!=p['common_member_SHA']:raise ValueError('Original public manifest differs')
        for name,wanted in p['common_member_SHA'].items():
            dest=safe(assets,name);take=name in p['asset_SHA'];h=hashlib.sha256()
            if take:dest.parent.mkdir(parents=True,exist_ok=True)
            out=dest.open('xb') if take else None
            try:
                with z.open(name) as f:
                    for b in iter(lambda:f.read(8*1024**2),b''):
                        h.update(b)
                        if out:out.write(b)
            finally:
                if out:out.close()
            if h.hexdigest()!=wanted:raise ValueError('Original public member SHA/CRC differs: '+name)
    for name,h in p['asset_SHA'].items():
        if sha(assets/name)!=h:raise ValueError('Actual restored public assets differ')
    wheels_archive=a.root/'offline_original_wheels.zip';fetch(p['wheels'],wheels_archive)
    wheels=a.root/'wheels';wheels.mkdir()
    with zipfile.ZipFile(wheels_archive) as z:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip() is not None:raise ValueError('Offline wheel CRC/unique differs')
        for name in z.namelist():
            dest=safe(wheels,name);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
    manifest=json.loads((wheels/'wheel_manifest.json').read_bytes())
    if not json.loads((wheels/'linux_dependency_closure.json').read_bytes())['complete']:raise PermissionError('Offline wheel closure incomplete')
    for name,row in manifest['wheel_files'].items():
        if sha(wheels/name)!=row['sha256']:raise ValueError('Original wheel SHA differs')
    def child(name,argv):
        with (a.root/(name+'_stdout.log')).open('wb') as out,(a.root/(name+'_stderr.log')).open('wb') as err:
            c=subprocess.Popen(argv,stdout=out,stderr=err,stdin=subprocess.DEVNULL)
            write(a.root/(name+'_dispatch.json'),dict(actual_UTC=utc().isoformat(),pid=c.pid,argv=argv));code=c.wait()
        write(a.root/(name+'_natural_exit.json'),dict(actual_UTC=utc().isoformat(),pid=c.pid,argv=argv,natural_exit=code))
        if code:raise RuntimeError('Original child failed: '+name)
    child('venv',[sys.executable,'-m','venv','--system-site-packages',str(assets/'.venv')])
    python=str(assets/'.venv/bin/python')
    child('offline_install',[python,'-m','pip','install','--no-index','--find-links',str(wheels),'--no-deps','-r',str(wheels/'offline_requirements.txt')])
    code="import torch,numpy,transformers,sentencepiece,scipy,sklearn,importlib.metadata as im,json;print(json.dumps({n:im.version(n) for n in "+repr(list(p['runtime_versions']))+"}))"
    versions=json.loads(subprocess.check_output([python,'-B','-c',code],text=True))
    if versions!=p['runtime_versions']:raise ValueError('Actual runtime package versions differ')
    write(a.root/'runtime_original_receipt.json',dict(status='GROUP5_BATCH5_EXACT_PUBLIC_RUNTIME_RESTORED',actual_UTC=utc().isoformat(),
        GPU_UUID=gpu,python=python,versions=versions,source_SHA=sha(__file__),plan_SHA=a.plan_sha,assets_SHA=p['asset_SHA'],
        common_SHA=p['common']['whole_SHA'],wheel_SHA=p['wheels']['whole_SHA'],all_public_original_SHA_CRC_unique_exact_set_passed=True,
        global_environment_changed=False,package_index_contacted=False,task_weights_restored=False,original_pickle_values_decoded=False,
        models_created=0,new_training_inference_scoring=0,argv=[sys.executable]+sys.argv))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--plan-sha',required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a)
    except BaseException as e:
        if a.root.exists():write(a.root/'failed_natural_exit.json',dict(exit=1,actual_UTC=utc().isoformat(),error_type=type(e).__name__,message=str(e)))
        raise
    else:write(a.root/'natural_exit.json',dict(exit=0,actual_UTC=utc().isoformat()))
