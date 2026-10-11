"""Audit the actual downloaded precheck; extract only authenticated original bytes."""
import datetime, hashlib, json, sys, zipfile
from pathlib import Path

base=Path('D:/CodexBackups/selective_flow_20261003_1105/weak_retention_pilot40_20261007T143015Z')
bundle=Path('work/weak_retention_pilot40_20261007T143015Z/bundle').resolve()
sys.path.insert(0,str(bundle))
from fold_evidence import verify_capsule
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
cap=base/'A_precheck_complete_capture.zip'
cr=json.loads((base/'A_precheck_capture_receipt.json').read_text())
assert cap.stat().st_size==cr['bytes']
manifest,receipt=verify_capsule(cap,cr['sha256'])
assert receipt['protocol_sha256']=='2cab1dfb9ad00075f7a4fb6609b9bad1f6b2241c422a170c86503830089e3c68'
cp=base/'A_original/out/precheck_full.pt'
assert cp.stat().st_size==receipt['complete_checkpoint']['bytes']
assert sha(cp)==receipt['complete_checkpoint']['sha256']
with zipfile.ZipFile(cp) as z:
    assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
    cp_members=len(z.namelist())
with zipfile.ZipFile(cap) as z:
    for name in z.namelist():
        if not name.startswith('run/'):continue
        target=(base/'A_original'/name[4:]).resolve()
        assert target.is_relative_to((base/'A_original').resolve())
        data=z.read(name)
        if target.exists():assert target.read_bytes()==data
        else:
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
result=dict(status='ACTUAL_A_PRECHECK_COMPLETE_D_BYTES_CAPTURE_PASSED_CPU_PENDING',
    actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    capsule_SHA=cr['sha256'],capsule_members=cr['member_count'],
    whole_checkpoint_SHA=receipt['complete_checkpoint']['sha256'],
    whole_checkpoint_bytes=cp.stat().st_size,checkpoint_crc_unique_members=cp_members,
    natural_exit=0,child=receipt['pid'],protocol_SHA=receipt['protocol_sha256'],
    peak=receipt['peak'],
    failed_direct_A_to_B_transfer='SCP connection timeout; no authentication prompt, no transferred checkpoint',
    next_action='Relay these D original bytes via authenticated B SFTP, then original native CPU audit.')
(base/'actual_A_precheck_D_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
