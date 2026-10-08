import hashlib,json,sys,zipfile
from pathlib import Path
base=Path('D:/CodexBackups/selective_flow_20261003_1105/weak_retention_pilot40_20261007T143015Z')
bundle=Path('work/weak_retention_pilot40_20261007T143015Z/bundle').resolve();sys.path.insert(0,str(bundle))
from fold_evidence import verify_capsule,join
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
cap=base/'B_precheck_complete_capture.zip';cr=json.loads((base/'B_precheck_capture_receipt.json').read_text())
assert cap.stat().st_size==cr['bytes']
manifest,r=verify_capsule(cap,cr['sha256'])
with zipfile.ZipFile(cap) as z:
    for name in z.namelist():
        if not name.startswith('run/'):continue
        path=(base/'B_CPU_original'/name[4:]).resolve();assert path.is_relative_to((base/'B_CPU_original').resolve())
        path.parent.mkdir(parents=True,exist_ok=True);assert not path.exists();path.write_bytes(z.read(name))
names=dict(original_capsule='A_precheck_complete_capture.zip',CPU_capsule='B_precheck_complete_capture.zip',
    complete_checkpoint='A_original/out/precheck_full.pt',prediction='A_original/out/precheck_dummy_predictions.npz',
    original_receipt='A_original/out/actual_stage_receipt.json',CPU_receipt='B_CPU_original/out/actual_stage_receipt.json')
binding={k:dict(path=v,sha256=sha(base/v)) for k,v in names.items()}
(base/'precheck_joint_binding.json').write_text(json.dumps(binding,indent=2)+'\n')
join(binding,base,base/'precheck_D_B_CPU_joint.json')
print(json.dumps(dict(status=json.loads((base/'precheck_D_B_CPU_joint.json').read_text())['status'],
    joint_SHA=sha(base/'precheck_D_B_CPU_joint.json'),CPU=r)))
