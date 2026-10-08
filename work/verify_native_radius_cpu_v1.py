from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,zipfile,subprocess,sys,time,os
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();q=lambda c:subprocess.check_output(c,text=True).strip()
gpu=q(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader']);assert gpu.split(',')[0].strip()==a.expected_uuid
compute=q(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']);assert not compute
assert sha(a.root/'snapshot.zip')==a.expected_sha;orig=a.root/'originals';orig.mkdir(exist_ok=False)
with zipfile.ZipFile(a.root/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    manifest=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(manifest)|{'member_manifest.json'}
    for name,item in manifest.items():
        pp=PurePosixPath(name);assert not pp.is_absolute() and '..' not in pp.parts
        b=z.read(name);assert len(b)==item['bytes'] and hashlib.sha256(b).hexdigest()==item['sha256']
        target=orig/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
cmd=[sys.executable,str(orig/'audit_native_radius_cal_v1.py'),'--root',str(orig),'--out',str(a.root/'cpu_array_audit.json')]
with (a.root/'cpu_array_audit.log').open('x') as f:
    child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);t=time.monotonic();actual=[]
    while time.monotonic()-t<2:
        actual=(Path('/proc')/str(child.pid)/'cmdline').read_bytes().decode().split('\0')[:-1]
        if actual==cmd:break
        time.sleep(.01)
    assert actual==cmd;code=child.wait()
exit={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'wrapper_pid':os.getpid(),'child_pid':child.pid,'argv':cmd,'actual_proc_argv':actual,'exit_code':code}
(a.root/'cpu_array_exit.json').write_text(json.dumps(exit,indent=2));assert code==0
receipt={'status':'OTHER_NODE_CAL_NATIVE_RADIUS_ORIGINALS_NUMPY_VERIFIED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'gpu_uuid':a.expected_uuid,'compute':compute,
    'original_packet_sha256':a.expected_sha,'files':len(manifest),'array_audit_sha256':sha(a.root/'cpu_array_audit.json'),'exit_sha256':sha(a.root/'cpu_array_exit.json'),
    'original_GPU_array_sha256':sha(orig/'mechanism/cal_candidate_arrays.npz'),'no_CPU_model_forward':True,'no_torch_import':True,'wrapper_source_sha256':sha(__file__),
    'full_process_argv':q(['ps','-ww','-eo','pid,ppid,args'])}
(a.root/'cpu_preservation_receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt['status'])
