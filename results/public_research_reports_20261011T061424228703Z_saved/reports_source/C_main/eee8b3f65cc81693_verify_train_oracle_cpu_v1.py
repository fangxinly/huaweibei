"""Verify original diagnostic files/arrays on an independent CPU target, no Torch."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8388608),b''):h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
p.add_argument('--expected-uuid',required=True);a=p.parse_args();d=a.directory
manifest=json.loads((d/'preservation_manifest.json').read_text(encoding='utf-8'))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name','--format=csv,noheader'],text=True).strip()
assert gpu.split(',')[0]==a.expected_uuid==manifest['target_cpu_uuid']
assert manifest['diagnostic_node']=='c' and manifest['target_cpu_node']=='b'
assert manifest['cpu_source_sha256']==sha(__file__)
for name,info in manifest['files'].items():
    assert (d/name).stat().st_size==info['bytes'] and sha(d/name)==info['sha256'],name
assert (d/'exit_code.txt').read_text().strip()=='0'
command=[sys.executable,str(d/'audit_train_oracle_v1.py'),'--plan',str(d/'train_oracle_plan_v2.json'),
         '--directory',str(d/'out'),'--report',str(d/'independent_cpu_array_audit.json')]
subprocess.run(command,check=True)
receipt=dict(status='ORIGINAL_TRAIN_ORACLE_CPU_FILES_AND_ARRAYS_VERIFIED',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu=gpu,
    diagnostic_node='c',target_cpu_node='b',target_uuid=a.expected_uuid,
    cpu_source_sha256=sha(__file__),manifest_sha256=sha(d/'preservation_manifest.json'),
    files=manifest['files'],array_audit_sha256=sha(d/'independent_cpu_array_audit.json'),
    array_audit_subprocess_exit_code=0,torch_imported=False,cuda_training_executed=False,
    scope='Original C diagnostic preserved and independently audited with NumPy on B; not new fitting or C result regenerated.')
(d/'cpu_preservation_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
print('ORACLE_CPU_PRESERVATION_COMPLETE',flush=True)
