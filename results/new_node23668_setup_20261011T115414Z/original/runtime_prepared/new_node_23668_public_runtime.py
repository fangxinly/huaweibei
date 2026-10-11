"""New node: exact isolated public runtime only; no task decoding or model work."""
import argparse, datetime as dt, hashlib, json, os, pathlib, shutil
import subprocess, sys, time, urllib.parse, urllib.request, zipfile
P=pathlib.Path
def now():return dt.datetime.now(dt.timezone.utc)
def write(p,o):
    p=P(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(o,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):
    h=hashlib.sha256()
    with P(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(argv):return subprocess.check_output(argv,text=True).strip()
def safe(base,name):
    q=(base/name).resolve()
    if '\\' in name or ':' in name or not q.is_relative_to(base.resolve()):raise ValueError('Unsafe original member')
    return q
def main(a):
    if sha(a.plan)!=a.plan_sha:raise PermissionError('Exact plan differs')
    p=json.loads(a.plan.read_bytes())
    if p['status']!='NEW_NODE23668_ISOLATED_PUBLIC_RUNTIME_FROZEN' or sha(__file__)!=p['source_SHA']:raise PermissionError('Source/plan not frozen')
    if a.root.exists():raise FileExistsError('Fresh runtime root required')
    start=now()
    gpu=run(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'])
    compute=run(['nvidia-smi','--query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'])
    memmax=int(P('/sys/fs/cgroup/memory.max').read_text());memcurrent=int(P('/sys/fs/cgroup/memory.current').read_text())
    hostmem=int(next(x.split()[1] for x in P('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    available=min(hostmem,memmax-memcurrent)
    if gpu!=p['GPU_UUID'] or compute or available<6*1024**3 or shutil.disk_usage(a.root.parent).free<15*1024**3:raise PermissionError('New-node UUID/empty compute/cgroup RAM/space failed')
    if not 0<=(start-dt.datetime.fromisoformat(p['local_space_capture_UTC'])).total_seconds()<=300:raise PermissionError('Local space observation stale')
    if p['local_C_free_bytes']<200*1024**2 or p['local_D_free_bytes']<40*1024**2:raise PermissionError('Local preservation floors failed')
    if (dt.datetime.fromisoformat(p['lease_end_UTC'])-start).total_seconds()<p['stage_budget_seconds']+7200:raise PermissionError('Human lease plus saving reserve failed')
    a.root.mkdir();meta=a.root/'metadata';meta.mkdir()
    write(meta/'fresh_physical.json',{'actual_UTC':start.isoformat(),'GPU_UUID':gpu,'compute':compute,'cgroup_memory_max':memmax,'cgroup_memory_current':memcurrent,'available_RAM_bytes':available,'remote_free_bytes':shutil.disk_usage(a.root).free,'fullargv':run(['ps','-eo','pid,ppid,args','--width','10000']),'argv':[sys.executable]+sys.argv})
    shutil.copyfile(a.plan,meta/'plan.json');shutil.copyfile(__file__,meta/P(__file__).name)
    def child(name,argv):
        with (meta/(name+'_stdout.log')).open('wb') as out,(meta/(name+'_stderr.log')).open('wb') as err:
            c=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err)
            write(meta/(name+'_dispatch.json'),{'actual_UTC':now().isoformat(),'pid':c.pid,'argv':argv})
            code=c.wait()
        write(meta/(name+'_natural_exit.json'),{'actual_UTC':now().isoformat(),'pid':c.pid,'argv':argv,'natural_exit':code})
        if code:raise RuntimeError(name+' natural nonzero; original logs preserved')
    # Public unchanged archives were already preserved in Release. Verify every member.
    common=P(p['common']['pretransport_path']);wheelszip=P(p['wheels']['pretransport_path'])
    for path,key in [(common,'common'),(wheelszip,'wheels')]:
        if path.stat().st_size!=p[key]['bytes'] or sha(path)!=p[key]['whole_SHA']:raise ValueError('Transport original changed: '+key)
    assets=a.root/'public';assets.mkdir();wheels=a.root/'wheels';wheels.mkdir()
    with zipfile.ZipFile(common) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or set(names)!=set(p['common_member_SHA'])|{'asset_manifest.json'}:raise ValueError('Common unique/exactset differs')
        manifest=json.loads(z.read('asset_manifest.json'))
        if {r['archive']:r['sha256'] for r in manifest['files']}!=p['common_member_SHA']:raise ValueError('Common original manifest differs')
        for n,h in p['common_member_SHA'].items():
            target=safe(assets,n);take=n in p['asset_SHA'];digest=hashlib.sha256()
            if take:target.parent.mkdir(parents=True,exist_ok=True)
            out=target.open('xb') if take else None
            try:
                with z.open(n) as f:
                    for b in iter(lambda:f.read(8*1024**2),b''):
                        digest.update(b)
                        if out:out.write(b)
            finally:
                if out:out.close()
            if digest.hexdigest()!=h:raise ValueError('Original common SHA/CRC differs: '+n)
    for n,h in p['asset_SHA'].items():
        if sha(assets/n)!=h:raise ValueError('Restored asset differs')
    with zipfile.ZipFile(wheelszip) as z:
        names=z.namelist()
        if len(names)!=len(set(names)) or z.testzip() is not None:raise ValueError('Original wheel archive CRC/unique failed')
        for n in names:
            target=safe(wheels,n);target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(n))
    manifest=json.loads((wheels/'wheel_manifest.json').read_bytes())
    if not json.loads((wheels/'linux_dependency_closure.json').read_bytes())['complete']:raise PermissionError('Original 24-package dependency closure incomplete')
    for n,row in manifest['wheel_files'].items():
        if sha(wheels/n)!=row['sha256']:raise ValueError('Original individual wheel differs')
    write(meta/'public_original_archive_verification.json',{'actual_UTC':now().isoformat(),'common_SHA':sha(common),'wheels_SHA':sha(wheelszip),'common_all_member_SHA_CRC_unique_exactset':True,'all_offline_wheel_SHA_CRC_unique':True,'assets_SHA':p['asset_SHA'],'task_pickle_numerically_decoded':False})
    # Base torch differs: download the exact official CUDA 12.1 wheel into this new root.
    spec=p['torch_wheel'];url,fragment=urllib.parse.urldefrag(spec['wheel_link_with_original_SHA'])
    if urllib.parse.urlparse(url).hostname not in ['download.pytorch.org','download-r2.pytorch.org']:raise PermissionError('Unexpected Torch source')
    wanted=urllib.parse.parse_qs(fragment)['sha256'][0]
    wheel=a.root/urllib.parse.unquote(urllib.parse.urlparse(url).path.rsplit('/',1)[1])
    h=hashlib.sha256();size=0;downloadstart=now()
    with urllib.request.urlopen(url,timeout=90) as response,wheel.open('xb') as out:
        length=response.headers.get('Content-Length')
        for b in iter(lambda:response.read(8*1024**2),b''):
            if (now()-start).total_seconds()>p['stage_budget_seconds']:raise TimeoutError('Frozen setup stage budget exceeded')
            out.write(b);h.update(b);size+=len(b)
    if h.hexdigest()!=wanted or (length is not None and int(length)!=size):raise ValueError('Official Torch original SHA/length differs')
    with zipfile.ZipFile(wheel) as z:
        if len(z.namelist())!=len(set(z.namelist())) or z.testzip() is not None:raise ValueError('Official Torch wheel CRC/unique failed')
        torchmeta=z.read('torch-2.1.0+cu121.dist-info/METADATA').decode()
    (meta/'official_torch_original_METADATA.txt').write_text(torchmeta,encoding='utf8')
    write(meta/'official_torch_download_receipt.json',{'start_UTC':downloadstart.isoformat(),'actual_UTC':now().isoformat(),'official_reference':spec,'bytes':size,'SHA':h.hexdigest(),'CRC_unique':True})
    child('venv',[sys.executable,'-m','venv','--system-site-packages',str(a.root/'.venv')])
    python=str(a.root/'.venv/bin/python')
    child('offline24_install',[python,'-m','pip','install','--no-cache-dir','--no-index','--find-links',str(wheels),'--no-deps','-r',str(wheels/'offline_requirements.txt')])
    child('exact_torch_install',[python,'-m','pip','install','--no-cache-dir','--no-index','--no-deps',str(wheel)])
    code="import importlib.metadata as im,json,torch,numpy,transformers,sentencepiece,scipy,sklearn; print(json.dumps({'versions':{n:im.version(n) for n in "+repr(list(p['runtime_versions']))+"},'CUDA_available':torch.cuda.is_available(),'CUDA_version':torch.version.cuda,'GPU_name':torch.cuda.get_device_name(0)},sort_keys=True))"
    child('runtime_import',[python,'-B','-c',code])
    result=json.loads((meta/'runtime_import_stdout.log').read_bytes())
    if result['versions']!=p['runtime_versions'] or not result['CUDA_available'] or result['CUDA_version']!='12.1':raise ValueError('Actual runtime or CUDA import differs')
    write(meta/'runtime_original_receipt.json',{'actual_UTC':now().isoformat(),'status':'NEW_NODE23668_EXACT_25_PUBLIC_RUNTIME_RESTORED','GPU_UUID':gpu,'python':python,'versions':result['versions'],'CUDA_available':result['CUDA_available'],'CUDA_version':result['CUDA_version'],'source_SHA':sha(__file__),'plan_SHA':a.plan_sha,'assets_SHA':p['asset_SHA'],'torch_official_wheel_SHA':wanted,'global_environment_changed':False,'task_arrays_models_training_inference_scoring_once':0,'native_training_or_recovery_qualification':False,'real_CPU_original_audit':False})
    write(meta/'natural_exit.json',{'actual_UTC':now().isoformat(),'exit':0})
    print(json.dumps({'status':'EXACT_PUBLIC_RUNTIME_RESTORED_NO_TASK_EXECUTION','metadata':str(meta),'python':python}),flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--plan',type=P,required=True);parser.add_argument('--plan-sha',required=True);parser.add_argument('--root',type=P,required=True);a=parser.parse_args()
    try:main(a)
    except BaseException as e:
        if a.root.exists():write(a.root/'metadata'/'failed_natural_exit.json',{'actual_UTC':now().isoformat(),'exit':1,'error_type':type(e).__name__,'message':str(e)})
        raise
