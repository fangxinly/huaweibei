"""Authenticate completed training originals downloaded from A; never load model objects."""
import datetime, hashlib, json, sys, zipfile
from pathlib import Path
base=Path('D:/CodexBackups/selective_flow_20261003_1105/weak_retention_train40_20261007T145201Z')
bundle=Path('work/weak_retention_train40_20261007T145201Z/bundle').resolve()
sys.path.insert(0,str(bundle))
from fold_evidence import verify_capsule
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
cap=base/'A_complete_capture.zip'
cr=json.loads((base/'A_capture_receipt.json').read_text(encoding='utf8'))
assert cap.stat().st_size==cr['bytes']
manifest,receipt=verify_capsule(cap,cr['sha256'])
assert receipt['protocol_sha256']=='cff9da9c18a4ac10a0d53b585a24126beb492735b4310aa1b6fd113f0f268092'
ref=receipt['complete_resume_and_selected']
cp=base/'resume_and_selected_full.pt'
assert cp.stat().st_size==ref['bytes'] and sha(cp)==ref['sha256']
with zipfile.ZipFile(cp) as z:
 assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
 members=len(z.namelist())
with zipfile.ZipFile(cap) as z:
 for name in z.namelist():
  if not name.startswith('run/'):continue
  dest=(base/'A_original'/name[4:]).resolve()
  assert dest.is_relative_to((base/'A_original').resolve())
  data=z.read(name)
  if dest.exists():assert dest.read_bytes()==data
  else:dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(data)
result=dict(status='ACTUAL_TRAIN40_A_COMPLETE_D_BYTES_CAPTURE_PASSED_CPU_PENDING',
 actual_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 capsule_SHA=cr['sha256'],capsule_members=cr['member_count'],
 whole_checkpoint_SHA=ref['sha256'],whole_checkpoint_bytes=cp.stat().st_size,
 checkpoint_crc_unique_members=members,child=receipt['pid'],natural_exit=0,
 protocol_SHA=receipt['protocol_sha256'],best_epoch=receipt['best_epoch'],
 selected_INNER_metrics=receipt['selected_INNER_five_development_only'])
(base/'actual_A_train40_D_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result))
