"""Other-node original-byte preservation and NumPy verification, no torch."""
from pathlib import Path,PurePosixPath
import argparse,datetime,hashlib,json,subprocess,sys,zipfile,os
p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--expected-sha",required=True);p.add_argument("--expected-uuid",required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=lambda c:subprocess.check_output(c,text=True).strip()
gpu=q(["nvidia-smi","--query-gpu=uuid,name","--format=csv,noheader"])
assert gpu.split(",")[0]==a.expected_uuid
packet=a.root/"snapshot.zip";assert sha(packet)==a.expected_sha
orig=a.root/"originals";orig.mkdir(exist_ok=False)
with zipfile.ZipFile(packet) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    manifest=json.loads(z.read("member_manifest.json"))
    assert set(z.namelist())==set(manifest)|{"member_manifest.json"}
    for name,item in manifest.items():
        pp=PurePosixPath(name);assert not pp.is_absolute() and ".." not in pp.parts
        data=z.read(name);assert len(data)==item["bytes"] and hashlib.sha256(data).hexdigest()==item["sha256"]
        target=orig/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
cmd=[sys.executable,str(orig/"audit_scalar_geometry_v1.py"),"--root",str(orig),"--phase","execute","--out",str(a.root/"cpu_array_audit.json")]
with (a.root/"cpu_audit.log").open("x") as handle:
    child=subprocess.Popen(cmd,stdout=handle,stderr=subprocess.STDOUT)
    launch=dict(wrapper_pid=os.getpid(),child_pid=child.pid,argv=cmd,actual_proc_argv=(Path("/proc")/str(child.pid)/"cmdline").read_bytes().decode().split("\0")[:-1])
    code=child.wait()
(a.root/"cpu_exit.json").write_text(json.dumps(dict(actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exit_code=code,**launch),indent=2))
assert code==0
audit=json.loads((a.root/"cpu_array_audit.json").read_text())
assert audit["status"]=="ORIGINAL_SCALAR_GEOMETRY_EXECUTE_INDEPENDENT_AUDIT_PASSED"
r=dict(status="OTHER_NODE_SCALAR_GEOMETRY_ORIGINALS_AND_NUMPY_VERIFIED",actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    gpu_uuid=gpu.split(",")[0],original_packet_sha256=sha(packet),files=len(manifest),no_cpu_model_forward=True,no_torch_import=True,
    original_gpu_receipt_sha256=sha(orig/"execute/receipt.json"),cpu_array_audit_sha256=sha(a.root/"cpu_array_audit.json"),
    cpu_exit_sha256=sha(a.root/"cpu_exit.json"),wrapper_source_sha256=sha(__file__),
    full_process_argv=q(["ps","-ww","-eo","pid,ppid,args"]),audit=audit)
(a.root/"cpu_preservation_receipt.json").write_text(json.dumps(r,indent=2));print(r["status"])

