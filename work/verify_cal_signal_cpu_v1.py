from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,os,subprocess,sys,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--expected-sha',required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=lambda cmd:subprocess.check_output(cmd,text=True).strip()
gpu=q(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader']);assert gpu.split(',')[0]=='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'
assert sha(a.root/'snapshot.zip')==a.expected_sha
orig=a.root/'originals';orig.mkdir(exist_ok=False)
with zipfile.ZipFile(a.root/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    members=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(members)|{'member_manifest.json'}
    assert 'metric_labels.npz' not in members
    for name,item in members.items():
        pp=PurePosixPath(name);assert not pp.is_absolute() and '..' not in pp.parts
        data=z.read(name);assert len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256']
        target=orig/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
cmd=[sys.executable,str(orig/'audit_cal_video_equal_signal_v1.py'),'--root',str(orig),'--out',str(a.root/'cpu_array_audit.json')]
with (a.root/'cpu_audit.log').open('x') as handle:
    child=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT)
    argv=(Path('/proc')/str(child.pid)/'cmdline').read_bytes().decode().split('\0')[:-1]
    code=child.wait()
ex=dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wrapper_pid=os.getpid(),child_pid=child.pid,argv=cmd,actual_proc_argv=argv,exit_code=code)
(a.root/'cpu_exit.json').write_text(json.dumps(ex,indent=2));assert code==0 and argv==cmd
audit=json.loads((a.root/'cpu_array_audit.json').read_text());assert audit['status']=='CAL_ONLY_VIDEO_EQUAL_INDEPENDENT_CPU_AUDIT_PASSED'
receipt=dict(status='OTHER_NODE_CAL_SIGNAL_ORIGINALS_AND_NUMPY_VERIFIED',actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_uuid=gpu.split(',')[0],original_packet_sha256=sha(a.root/'snapshot.zip'),original_scientific_files=len(members),cpu_array_audit_sha256=sha(a.root/'cpu_array_audit.json'),cpu_exit_sha256=sha(a.root/'cpu_exit.json'),no_cpu_model_forward=True,no_EVAL_labels_in_packet=True,no_fullTRAIN_label_archive_in_packet=True,wrapper_source_sha256=sha(__file__),full_process_argv=q(['ps','-ww','-eo','pid,ppid,args']),audit=audit)
(a.root/'cpu_preservation_receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt['status'])
