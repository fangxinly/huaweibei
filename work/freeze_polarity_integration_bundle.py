import ast,hashlib,json,shutil,sys,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]
stamp=sys.argv[1];out=base/'work'/('polarity_intensity_integration_frozen_'+stamp);out.mkdir(exist_ok=False)
bundle=out/'bundle';bundle.mkdir()
prep=base/'work/polarity_intensity_official_pipeline_preparation'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in prep.iterdir():
    if p.suffix in ('.py','.npy'):shutil.copyfile(p,bundle/p.name)
wrapper=(bundle/'budgeted_native.py').read_text()
assert wrapper.count("root/'native_contract.py'")==1
(bundle/'budgeted_native.py').write_text(wrapper.replace("root/'native_contract.py'","root/'integration_contract.py'"),encoding='utf-8')
for p in bundle.glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
plan=json.loads((base/'work/polarity_intensity_frozen_20261008T152344Z/bundle/plan.json').read_text())
plan.update(actualclock_freeze_UTC=sys.argv[2],sources={p.name:sha(p) for p in sorted(bundle.iterdir())},
    qualification='CPU synthetic training optimizer and saved tail integration; full encoder/real labels absent',
    fixed_candidate_mode='factorized_aux',matched_control_mode='regression_aux',
    candidate_full_encoder_precheck_D_B_qualification=False)
(bundle/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
archive=out/'source.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(bundle.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(archive) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
receipt=dict(status='INTEGRATION_SOURCE_FROZEN_BEFORE_NATIVE_EXECUTION',actualclock_UTC=sys.argv[2],
    local_root=str(out),source_ZIP_SHA256=sha(archive),plan_SHA256=sha(bundle/'plan.json'),source_ZIP_bytes=archive.stat().st_size,
    full_encoder_or_real_data=False,new_VAL_TEST_scores=False)
(out/'freeze_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('polarity_intensity_integration_'+stamp);dest.mkdir(exist_ok=False)
for p in (archive,out/'freeze_receipt.json'):
    shutil.copyfile(p,dest/p.name);assert sha(p)==sha(dest/p.name)
print(json.dumps(dict(**receipt,D=str(dest))))
