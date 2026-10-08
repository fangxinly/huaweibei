"""Real read-only node snapshot. Running prefix is never a COMPLETE receipt."""
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def argv_sources(value):
    if isinstance(value,dict):
        for k,v in value.items():
            if k in ('argv','fullargv','expected_child_fullargv') and isinstance(v,list) and len(v)>1:
                p=Path(str(v[1]))
                if p.is_file() and str(p).startswith('/data/coding/') and p.suffix=='.py':yield p.parent
            yield from argv_sources(v)
    elif isinstance(value,list):
        for v in value:yield from argv_sources(v)

def main(a):
    output=Path('/data/coding')/('lease_dynamic_'+a.node+'_'+a.stamp)
    if output.exists():raise ValueError('Fresh dynamic root required')
    output.mkdir();payload=output/'payload';payload.mkdir()
    plan=json.loads(a.plan.read_text());assets=Path('/data/coding/multimodal_flow_public_20261006T1341Z')
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if uuid!=plan['assigned_gpu_uuid'][a.node]:raise ValueError('Physical UUID mismatch')
    start=dt.datetime.now(dt.timezone.utc).isoformat()
    versions={n:importlib.metadata.version(n) for n in plan['runtime_exact_versions']}
    if versions!=plan['runtime_exact_versions']:raise ValueError('Native runtime differs')
    for name,h in plan['asset_sha256'].items():
        if sha(assets/name)!=h:raise ValueError('Asset differs '+name)
    write(payload/'physical.json',dict(actual_capture_start_utc=start,node=a.node,uuid=uuid,
        compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True),
        full_process_table=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True),
        fullargv=[sys.executable]+sys.argv,pid=os.getpid(),native_runtime=versions,public_asset_sha256=plan['asset_sha256'],
        free_bytes=shutil.disk_usage('/data/coding').free,operator_source_sha256=sha(__file__)))
    large=[];source_roots={a.plan.parent} if a.plan.parent.name.startswith('anchored_flow_pilot40_source_') else set();copied=[];root_records=[]
    roots=[Path(p) for p in json.loads(a.roots.read_text())]
    for root in roots:
        if not root.is_dir() or not str(root.resolve()).startswith('/data/coding/'):
            raise ValueError('Declared original root absent/outside data coding: '+str(root))
        root_records.append({'original_root':str(root),'snapshot_is_not_natural_completion':True})
        for p in sorted(root.rglob('*')):
            if not p.is_file() or p.suffix=='.tmp' or '__pycache__' in p.parts:continue
            rel=p.relative_to(root);before=p.stat()
            if p.suffix in ('.pt','.bin') or before.st_size>16*1024**2:
                if False: # v2: completed qualified originals hashed as refs, never duplicated
                    dest=output/('frozen_'+p.name);shutil.copyfile(p,dest)
                    after=p.stat()
                    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                        raise RuntimeError('Live state changed during snapshot; refuse frozen-prefix claim')
                    h=sha(dest)
                    with zipfile.ZipFile(dest) as z:
                        if z.testzip() or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Snapshot complete state CRC/unique differs')
                    large.append({'original_path':str(p),'frozen_copy_path':str(dest),'bytes':dest.stat().st_size,'sha256':h,
                        'new_large_original_requires_D_transfer':True,'complete_training_claimed':False})
                else:
                    h=sha(p);after=p.stat()
                    if (before.st_ino,before.st_size,before.st_mtime_ns)!=(after.st_ino,after.st_size,after.st_mtime_ns):
                        raise RuntimeError('Original changed during hashing')
                    large.append({'original_path':str(p),'bytes':before.st_size,'sha256':h,'new_large_original_requires_D_transfer':False})
                continue
            q=payload/'originals'/root.name/rel;q.parent.mkdir(parents=True,exist_ok=True)
            data=p.read_bytes()
            if p.suffix=='.jsonl' and data and not data.endswith(b'\n'):data=data[:data.rfind(b'\n')+1]
            q.write_bytes(data);copied.append(q)
            if p.suffix=='.json':
                try:source_roots.update(argv_sources(json.loads(data)))
                except json.JSONDecodeError:raise RuntimeError('Atomic original JSON incomplete '+str(p))
    source_files={}
    for src in sorted(source_roots):
        for p in sorted(src.glob('*.py')):
            q=payload/'sources'/src.name/p.name;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
            source_files[str(p)]=sha(p)
    shutil.copy2(a.plan,payload/'pinned_asset_runtime_plan.json')
    shutil.copy2(__file__,payload/'dynamic_capture_operator.py')
    write(payload/'dynamic_manifest.json',dict(status='REAL_DYNAMIC_NODE_SNAPSHOT_WITH_RUNNING_PREFIX',
        actual_capture_start_utc=start,actual_capture_finish_utc=dt.datetime.now(dt.timezone.utc).isoformat(),node=a.node,
        original_roots=root_records,source_file_sha256=source_files,large_original_refs=large,
        prefix_is_not_COMPLETE=True,source_roots_from_original_fullargv=True,
        member_sha256={p.relative_to(payload).as_posix():sha(p) for p in sorted(payload.rglob('*')) if p.is_file()}))
    archive=output/'dynamic_capture.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(payload.rglob('*')):
            if p.is_file():z.write(p,p.relative_to(payload).as_posix())
    with zipfile.ZipFile(archive) as z:
        if z.testzip() or len(z.namelist())!=len(set(z.namelist())):raise ValueError('Dynamic capture ZIP failed')
    result=dict(status='REAL_DYNAMIC_SNAPSHOT_ZIP_CREATED_D_TRANSFER_PENDING',node=a.node,
        actual_utc=dt.datetime.now(dt.timezone.utc).isoformat(),sha256=sha(archive),bytes=archive.stat().st_size,
        members=len(z.namelist()),large_refs=large,not_completed_training=True)
    write(output/'dynamic_receipt.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--node',choices=['A','B','C'],required=True)
    p.add_argument('--stamp',required=True);p.add_argument('--plan',type=Path,required=True);p.add_argument('--roots',type=Path,required=True)
    main(p.parse_args())
