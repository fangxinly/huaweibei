import sys,json,shutil,zipfile,hashlib,ast
from pathlib import Path
root=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(root/'work/anchored_increment_candidate'))
from common import sha,read,write
ref=read(root/'work/anchored_increment_pointer.json');old=Path(ref['source']);d=Path(ref['D']);s=root/'work'/('anchored_increment_inference_'+sys.argv[1]);s.mkdir();p=read(old/'increment_audit_protocol.json')
for n in p['source_sha256']:shutil.copy2(old/n,s/n)
shutil.copy2(root/'work/anchored_increment_candidate/inference_upgrade.py',s/'inference_upgrade.py');shutil.copy2(root/'work/anchored_increment_candidate/integration_check.py',s/'cache_original.py')
for f in s.glob('*.py'):ast.parse(f.read_text(encoding='utf8'))
p.update(status='ORIGINAL_INCREMENT_DROPIN_INFERENCE_NATIVE_INTEGRATION_FROZEN',allowed_stages=['cache'],real_train_enabled=False,source_sha256={f.name:sha(f) for f in s.iterdir()},check_device='cuda',changed_state_remote_path='/data/coding/anchored_increment_train_A_20261008T024739Z/out/complete_changed_training_state.pt',tail_remote_path='/data/coding/anchored_increment_cache_A_20261008T024739Z/out/original_tail.pt',actualclock_integration_freeze_UTC=sys.argv[2]);write(s/'integration_protocol.json',p)
zp=d/'frozen_dropin_inference_source.zip'
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
    for f in s.iterdir():z.write(f,f.name)
with zipfile.ZipFile(zp) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for f in s.iterdir():assert hashlib.sha256(z.read(f.name)).hexdigest()==sha(f)
r=dict(source=str(s),ZIP=str(zp),ZIP_SHA=sha(zp),plan_SHA=sha(s/'integration_protocol.json'),actualclock=sys.argv[2],no_new_training_or_real_labels=True);write(d/'integration_source_reference.json',r);print(json.dumps(r))
