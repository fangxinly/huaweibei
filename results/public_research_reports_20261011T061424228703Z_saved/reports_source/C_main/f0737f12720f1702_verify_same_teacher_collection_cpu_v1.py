"""Other-node NumPy original receipt/array verification, no model CPU forward."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys, zipfile
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
p.add_argument('--original-teacher-directory',type=Path,required=True);p.add_argument('--fold',type=int,required=True)
p.add_argument('--target-node',choices=['a','b','c'],required=True);a=p.parse_args()
sha=lambda data:hashlib.sha256(data).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
expected={'a':'GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7','b':'GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067','c':'GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa'}
training=['a','b','c'][a.fold];assert training!=a.target_node
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert gpu==expected[a.target_node]
r=read(a.directory/'receipt.json');raw=(a.directory/'snapshot.zip').read_bytes()
assert r['status']=='ACTUAL_COLLECTION_ORIGINAL_SMALL_FILES_ATOMIC_PACKAGE_COMPLETE'
assert len(raw)==r['bytes'] and sha(raw)==r['sha256']
out=a.directory/'originals';out.mkdir(exist_ok=False)
with zipfile.ZipFile(a.directory/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    m=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
    for name,meta in m.items():
        target=out/name;assert target.resolve().is_relative_to(out.resolve())
        data=z.read(name);assert len(data)==meta['bytes'] and sha(data)==meta['sha256']
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
script=out/'audit_same_teacher_collection_v2.py'
assert sha(script.read_bytes())=='dcff3ad8965214682a9f13c6988f6123e04bc7cb6a79517c6092f680c1c188c3'
audit=a.directory/'cpu_array_audit.json'
command=[sys.executable,str(script),str(out),'--original-teacher-directory',str(a.original_teacher_directory),
         '--fold',str(a.fold),'--phase','execute','--output',str(audit)]
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat();start=now()
with (a.directory/'cpu_audit.log').open('x') as log:
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);code=child.wait()
exit_proof={'started_utc':start,'finished_utc':now(),'pid':child.pid,'argv':command,'exit_code':code}
(a.directory/'cpu_exit.json').write_text(json.dumps(exit_proof,indent=2))
assert code==0 and read(audit)['status']=='ACTUAL_COLLECTION_FIT_INNER_SCALAR_ARRAYS_INDEPENDENT_AUDIT_PASSED'
assert 'torch' not in sys.modules
receipt={'status':'INDEPENDENT_OTHER_NODE_COLLECTION_ORIGINAL_SHA_CRC_PROCESS_RECEIPTS_AND_NUMPY_ARRAYS_VERIFIED',
         'utc':now(),'training_node':training,'target_cpu_node':a.target_node,'fold':a.fold,'gpu_uuid':gpu,
         'package_sha256':r['sha256'],'original_transport_receipt_sha256':sha((a.directory/'receipt.json').read_bytes()),
         'original_members':m,'cpu_array_audit_sha256':sha(audit.read_bytes()),
         'cpu_exit_sha256':sha((a.directory/'cpu_exit.json').read_bytes()),'source_sha256':sha(Path(__file__).read_bytes()),
         'torch_imported':False,'cpu_full_model_forward':False,'old_full_weights_retransferred':False}
(a.directory/'cpu_preservation_receipt.json').write_text(json.dumps(receipt,indent=2))
print(receipt['status'],flush=True)
