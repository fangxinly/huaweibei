from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,zipfile,subprocess,sys,os,time
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-sha',required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();q=lambda x:subprocess.check_output(x,text=True).strip();utc=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
gpu=q(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader']);assert gpu.split(',')[0].strip()==a.expected_uuid
assert sha(a.root/'snapshot.zip')==a.expected_sha;orig=a.root/'originals';orig.mkdir(exist_ok=False)
with zipfile.ZipFile(a.root/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    manifest=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(manifest)|{'member_manifest.json'}
    for name,item in manifest.items():
        pp=PurePosixPath(name);assert not pp.is_absolute() and '..' not in pp.parts
        data=z.read(name);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        target=orig/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
audits={};exits={}
for phase in ['precheck','execute','evaluation']:
    cmd=[sys.executable,str(orig/('audit_matched_pool_eval_v1.py' if phase=='evaluation' else 'audit_matched_message_pools_v1.py')),'--root',str(orig)]
    if phase!='evaluation':cmd+=['--phase',phase]
    cmd+=['--out',str(a.root/('cpu_'+phase+'_audit.json'))]
    with (a.root/('cpu_'+phase+'.log')).open('x') as f:
        child=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT);t=time.monotonic();actual=[]
        while time.monotonic()-t<2:
            actual=(Path('/proc')/str(child.pid)/'cmdline').read_bytes().decode().split('\0')[:-1]
            if actual==cmd:break
            time.sleep(.01)
        assert actual==cmd;code=child.wait()
    exit=dict(utc=utc(),wrapper_pid=os.getpid(),child_pid=child.pid,argv=cmd,actual_proc_argv=actual,exit_code=code)
    ep=a.root/('cpu_'+phase+'_exit.json');ep.write_text(json.dumps(exit,indent=2));assert code==0
    audits[phase]=sha(a.root/('cpu_'+phase+'_audit.json'));exits[phase]=sha(ep)
receipt=dict(status='OTHER_NODE_MATCHED_MESSAGE_POOLS_ORIGINALS_AND_EVAL_NUMPY_VERIFIED',utc=utc(),gpu_uuid=a.expected_uuid,original_packet_sha256=a.expected_sha,files=len(manifest),audits_sha256=audits,exits_sha256=exits,no_CPU_model_forward=True,no_torch_import=True,original_GPU_prediction_sha256=sha(orig/'execute/predictions_frozen.npz'),original_EVAL_arrays_sha256=sha(orig/'evaluation/eval_arrays.npz'),wrapper_source_sha256=sha(__file__),full_process_argv=q(['ps','-ww','-eo','pid,ppid,args']))
(a.root/'cpu_preservation_receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt['status'])
