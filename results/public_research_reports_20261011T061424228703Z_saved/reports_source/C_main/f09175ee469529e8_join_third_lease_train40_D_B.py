import hashlib,json,sys,zipfile
from pathlib import Path
base=Path('D:/CodexBackups/selective_flow_20261003_1105/weak_retention_train40_20261007T145201Z')
bundle=Path('work/weak_retention_train40_20261007T145201Z/bundle').resolve();sys.path.insert(0,str(bundle))
from fold_evidence import verify_capsule,join
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
cap=base/'B_CPU_complete_capture.zip';cr=json.loads((base/'B_CPU_capture_receipt.json').read_text(encoding='utf8'))
assert cap.stat().st_size==cr['bytes'];manifest,r=verify_capsule(cap,cr['sha256'])
with zipfile.ZipFile(cap) as z:
 for name in z.namelist():
  if not name.startswith('run/'):continue
  path=(base/'B_CPU_original'/name[4:]).resolve();assert path.is_relative_to((base/'B_CPU_original').resolve())
  data=z.read(name)
  if path.exists():assert path.read_bytes()==data
  else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
names=dict(original_capsule='A_complete_capture.zip',CPU_capsule='B_CPU_complete_capture.zip',
 complete_checkpoint='resume_and_selected_full.pt',prediction='A_original/out/OUTER_prediction_only.npz',
 original_receipt='A_original/out/actual_stage_receipt.json',CPU_receipt='B_CPU_original/out/actual_stage_receipt.json')
binding={k:dict(path=v,sha256=sha(base/v)) for k,v in names.items()}
(base/'training_joint_binding.json').write_text(json.dumps(binding,indent=2)+'\n',encoding='utf8')
join(binding,base,base/'training_D_B_CPU_joint.json')
print(json.dumps(dict(status=json.loads((base/'training_D_B_CPU_joint.json').read_text(encoding='utf8'))['status'],joint_SHA=sha(base/'training_D_B_CPU_joint.json'),CPU=r)))
