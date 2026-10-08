"""Small original-environment CPU/CUDA fixture; never loads data or task weights."""
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


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def physical(plan):
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    processes=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True)
    versions={name:importlib.metadata.version(name) for name in plan['runtime_exact_versions']}
    assets=Path(plan['assets'])
    asset_sha={rel:sha(assets/rel) for rel in plan['asset_sha256']}
    source=Path(__file__).resolve().parent
    source_sha={rel:sha(source/rel) for rel in plan['source_sha256']}
    if uuid!=plan['assigned_gpu_uuid']:raise PermissionError('Fresh physical UUID differs')
    if versions!=plan['runtime_exact_versions']:raise PermissionError('Original native dependencies differ')
    if source_sha!=plan['source_sha256'] or asset_sha!=plan['asset_sha256']:
        raise PermissionError('Frozen source/public asset bytes differ')
    if sys.executable!=plan['python']:raise PermissionError('Native interpreter differs')
    end=dt.datetime.fromisoformat(plan['conservative_lease_end_UTC'])
    remaining=(end-dt.datetime.now(dt.timezone.utc)).total_seconds()
    if remaining<1800+7200:raise PermissionError('Fixture execution plus two-hour saving reserve absent')
    free=shutil.disk_usage(source).free
    if free<2*1024**3:raise PermissionError('Remote fixture saving space absent')
    return {'actual_utc':utc(),'uuid':uuid,'compute':compute,'full_process_table':processes,
            'runtime_exact_versions':versions,'source_sha256':source_sha,'asset_sha256':asset_sha,
            'remaining_seconds':remaining,'remote_free_bytes':free,
            'pid':os.getpid(),'fullargv':[sys.executable]+sys.argv,
            'proc_self_cmdline':Path('/proc/self/cmdline').read_bytes().decode().split('\0')[:-1]}


def run(a):
    root=a.protocol.resolve().parent
    if sha(a.protocol)!=a.protocol_sha:raise ValueError('Protocol SHA differs')
    plan=json.loads(a.protocol.read_text())
    if plan['status']!='NATIVE_SYNTHETIC_CPU_CUDA_ONLY_FROZEN':raise PermissionError('Scope differs')
    if a.mode=='worker':
        before=physical(plan);write(root/'actual_preflight.json',before)
        if before['compute'].strip():raise PermissionError('Existing GPU compute; healthy task must continue')
        tests=subprocess.run([sys.executable,str(root/'test_fold_contract.py')],capture_output=True,text=True)
        (root/'native_contract_tests.stdout.log').write_text(tests.stdout)
        (root/'native_contract_tests.stderr.log').write_text(tests.stderr)
        if tests.returncode:raise RuntimeError('Native contract regression tests failed')
        import torch
        from anchored_flow import synthetic_check
        if not torch.cuda.is_available():raise RuntimeError('Native CUDA unavailable')
        cpu=synthetic_check('cpu')
        cuda=synthetic_check('cuda')
        torch.cuda.synchronize()
        peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
        if max(peak.values())>6*1024**3:raise RuntimeError('Cumulative GPU budget exceeded')
        result={'status':'NATIVE_CPU_AND_P4_CUDA_SYNTHETIC_FLOW_PASSED','actual_utc':utc(),
                'cpu':cpu,'cuda':cuda,'cumulative_peak_bytes':peak,
                'real_data_or_pretrained_forward':False,'real_labels_read':False,
                'full_model_precheck_completed':False,'fivefold_training_started':False,
                'native_contract_test_natural_exit':tests.returncode,'native_contract_tests':27,
                'pid':os.getpid(),'fullargv':[sys.executable]+sys.argv,'protocol_sha256':a.protocol_sha}
        write(root/'actual_fixture_result.json',result)
        print(json.dumps(result),flush=True)
        return
    for name in ('actual_child.json','natural_exit.json','complete_capture.zip'):
        if (root/name).exists():raise FileExistsError('Fresh fixture root required')
    command=[sys.executable,str(Path(__file__).resolve()),'--protocol',str(a.protocol.resolve()),
             '--protocol-sha',a.protocol_sha,'--mode','worker']
    with (root/'worker.stdout.log').open('wb') as out,(root/'worker.stderr.log').open('wb') as err:
        child=subprocess.Popen(command,stdout=out,stderr=err)
        write(root/'actual_child.json',{'actual_utc':utc(),'pid':child.pid,'fullargv':command,
               'source_sha256':sha(__file__),'protocol_sha256':a.protocol_sha})
        code=child.wait()
    write(root/'natural_exit.json',{'actual_utc':utc(),'pid':child.pid,'fullargv':command,'natural_exit':code})
    after=physical(plan);write(root/'actual_complete_physical.json',after)
    files=sorted(p for p in root.rglob('*') if p.is_file())
    members={p.relative_to(root).as_posix():sha(p) for p in files}
    capture={'status':'COMPLETE' if code==0 else 'FAILED','actual_utc':utc(),'natural_exit':code,
             'child_pid':child.pid,'child_fullargv':command,'member_sha256':members,
             'large_task_weights_included':False,'fixture_only':True}
    write(root/'complete_capture_receipt.json',capture)
    with zipfile.ZipFile(root/'complete_capture.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,p.relative_to(root).as_posix())
        z.write(root/'complete_capture_receipt.json','complete_capture_receipt.json')
    with zipfile.ZipFile(root/'complete_capture.zip') as z:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip() is not None:
            raise RuntimeError('Capture ZIP CRC/unique failed')
    print(json.dumps({'status':capture['status'],'natural_exit':code,'child_pid':child.pid,
          'actual_utc':utc(),'capture_sha256':sha(root/'complete_capture.zip'),
          'capture_bytes':(root/'complete_capture.zip').stat().st_size}),flush=True)
    if code:print((root/'worker.stderr.log').read_text())
    raise SystemExit(code)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--protocol-sha',required=True);p.add_argument('--mode',choices=['worker','wrapper'],required=True)
    run(p.parse_args())
