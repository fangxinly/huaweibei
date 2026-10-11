"""One fresh samefold training, separate original CPU audit, immutable capture.

No resume, no outer scores, no retries, no process termination, no GitHub secrets.
"""
import argparse,datetime as dt,hashlib,importlib.metadata,json,os,pathlib,shutil,subprocess,sys,urllib.request
P=pathlib.Path
def now():return dt.datetime.now(dt.timezone.utc)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,o):
    temp=p.with_name(p.name+'.tmp');temp.write_text(json.dumps(o,indent=2)+'\n');temp.replace(p)
def physical(p,root,compute_empty=True):
    uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    cmax=int(P('/sys/fs/cgroup/memory.max').read_text());current=int(P('/sys/fs/cgroup/memory.current').read_text())
    available=min(cmax-current,int(next(x.split()[1] for x in P('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024)
    if uuid!=p['GPU_UUID'] or (compute_empty and compute) or available<6*1024**3:raise PermissionError('Fresh UUID/compute/container RAM failed')
    return dict(actual_UTC=now().isoformat(),GPU_UUID=uuid,compute=compute,cgroup_memory_max=cmax,cgroup_memory_current=current,effective_available_RAM_bytes=available,remote_free_bytes=shutil.disk_usage(root).free,fullargv=subprocess.check_output(['ps','-eo','pid,ppid,args','--width','10000'],text=True))
def run(a):
    root=a.root;planfile=root/'plan.json';p=json.loads(planfile.read_bytes())
    if sha(planfile)!=a.plan_sha or sha(__file__)!=p['supervisor_SHA'] or p['status']!='GROUP5_DIRECT_TRAIN_FROZEN':raise PermissionError('Exact formal supervisor/plan differs')
    if p['fold']!=1 or p['authorized_methods']!=['careflow'] or p.get('resume') or not p['fresh_resume_disabled']:raise PermissionError('Only fresh CaReFlow fold1 authorized')
    if (root/'out').exists() or (root/'dispatch.json').exists():raise FileExistsError('Formal stage already attempted')
    if sys.executable!=p['python'] or {n:importlib.metadata.version(n) for n in p['runtime_versions']}!=p['runtime_versions']:raise PermissionError('Exact public runtime differs')
    for n,h in p['source_SHA'].items():
        if sha(root/'source'/n)!=h:raise ValueError('Source differs: '+n)
    for n,h in p['asset_SHA'].items():
        if sha(P(p['assets_root'])/n)!=h:raise ValueError('Public asset differs: '+n)
    if sha(P(p['native_precheck_receipt']))!=p['native_precheck_receipt_SHA']:raise PermissionError('Native original receipt differs')
    if sha(P(p['runtime_original_receipt']))!=p['runtime_original_receipt_SHA']:raise PermissionError('Actual runtime original receipt differs')
    route=p['preservation_route']
    if route['mode']!='SERVER_ORIGINAL_LOCAL_STREAM_RELAY_SERVER_WHOLE_RESTORE':raise PermissionError('Unknown preservation route')
    routefile=root/'stream_route_original_receipt.json'
    if sha(routefile)!=route['qualification_receipt_SHA']:raise PermissionError('Original route qualification differs')
    qualified=json.loads(routefile.read_bytes())
    if qualified['status']!='FULL_ORIGINAL_RELEASE_STREAM_RESTORED_VERIFIED' or not qualified['all_member_SHA_CRC_unique_exact_set_passed']:raise PermissionError('Actual small whole route not verified')
    with urllib.request.urlopen(route['remote_loopback_relay']+'/health',timeout=15) as response:health=json.load(response)
    if health['status']!='LOCAL_ONLY_RELEASE_RELAY_READY' or health['repository']!='fangxinly/huaweibei':raise PermissionError('Fresh stream route unavailable')
    fresh=physical(p,root)
    if fresh['remote_free_bytes']<p['remote_required_bytes'] or p['local_C_free_bytes']<p['original_capture_local_C_required_bytes'] or p['local_D_free_bytes']<40*1024**2:raise PermissionError('One complete original plus remote restoration space absent')
    if not 0<=(now()-dt.datetime.fromisoformat(p['local_space_capture_UTC'])).total_seconds()<=300:raise PermissionError('Local reserved space observation stale')
    if (dt.datetime.fromisoformat(p['lease_end_UTC'])-now()).total_seconds()<p['stage_budget_seconds']+max(7200,p['saving_budget_seconds']):raise PermissionError('Conservative stage and saving lease absent')
    write(root/'supervisor_fresh_physical.json',fresh)
    env=dict(os.environ,PYTHONPATH=str(root/'source'))
    argv=[p['python'],'-B',str(root/'source/group5_direct_runtime_v2.py'),'--plan',str(planfile),'--plan-sha',a.plan_sha,'--bundle',str(root/'source'),'--assets',p['assets_root'],'--out',str(root/'out'),'--method','careflow','--fold','1','--stage','train']
    with (root/'stdout.log').open('xb') as out:
        c=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=env)
        write(root/'dispatch.json',dict(actual_UTC=now().isoformat(),child=c.pid,argv=argv,plan_SHA=a.plan_sha))
        code=c.wait()
    write(root/'wrapper_exit.json',dict(actual_UTC=now().isoformat(),child=c.pid,natural_exit=code))
    if code==0:
        if (dt.datetime.fromisoformat(p['lease_end_UTC'])-now()).total_seconds()<max(7200,p['saving_budget_seconds']):raise PermissionError('Original CPU and saving reserve absent')
        fresh=physical(p,root);write(root/'CPU_fresh_physical.json',fresh)
        with (root/'original_CPU_audit.once').open('x') as f:json.dump(dict(actual_UTC=now().isoformat(),plan_SHA=a.plan_sha),f)
        argv=[p['python'],'-B',str(root/'source/group5_direct_CPU_audit_v2.py'),'--root',str(root),'--plan-sha',a.plan_sha,'--out',str(root/'independent_CPU_audit.json')]
        with (root/'CPU_stdout.log').open('xb') as out:
            c=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=subprocess.STDOUT,env=dict(env,CUDA_VISIBLE_DEVICES=''))
            write(root/'CPU_dispatch.json',dict(actual_UTC=now().isoformat(),child=c.pid,argv=argv));cpu_code=c.wait()
        write(root/'CPU_wrapper_exit.json',dict(actual_UTC=now().isoformat(),child=c.pid,natural_exit=cpu_code))
        if cpu_code:raise RuntimeError('Original CPU audit natural nonzero; original preserved, no retry')
    # Capture only after observing the original process exit; a failed run stays failed.
    sys.path.insert(0,str(root/'source'));from group5_release_transport_v1 import seal,plan
    archive=root/'complete_actual_careflow_fold1_newnode_original.zip'
    if shutil.disk_usage(root).free<sum(x.stat().st_size for x in root.rglob('*') if x.is_file())+2*1024**3:raise PermissionError('Complete immutable original capture space absent')
    proof=seal(root,archive);write(root/'capture_receipt.json',dict(**proof,training_natural_exit=code,outer_scoring_performed=False))
    ranges=plan(archive,'group5-careflow-fold1-newnode-'+proof['archive_SHA'][:12]+'.zip')
    ranges.update(repository='fangxinly/huaweibei',tag='autonomous-flow-health-20261008',publication_authorized=True)
    write(root/'original_range_manifest.json',ranges)
    write(root/'preservation_ready.json',dict(actual_UTC=now().isoformat(),status='ORIGINAL_CAPTURE_READY_FOR_STREAM_UPLOAD_AND_REMOTE_FULL_RESTORATION',capture=proof,training_natural_exit=code,CPU_original_audit_passed=code==0,source_original_retained=True,actual_Release_restoration_passed=False))
    return code
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=P,required=True);parser.add_argument('--plan-sha',required=True);a=parser.parse_args()
    try:code=run(a)
    except Exception as e:
        write(a.root/'supervisor_failure.json',dict(actual_UTC=now().isoformat(),error_type=type(e).__name__,message=str(e),originals_retained=True));raise
    else:write(a.root/'supervisor_natural_exit.json',dict(actual_UTC=now().isoformat(),natural_exit=code));raise SystemExit(code)
