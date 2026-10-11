import hashlib,json,shutil,sys,zipfile
from pathlib import Path
base=Path(__file__).resolve().parents[1]
stamp=sys.argv[1]
src=base/'work/polarity_intensity_candidate_v1'
root=base/'work'/('polarity_intensity_frozen_'+stamp)
root.mkdir(exist_ok=False)
bundle=root/'bundle';bundle.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
bootstrap=base/'outputs/polarity_intensity_native_20261008T151107Z'
nodes={}
for node in ('A','B','C'):
    data=json.loads((bootstrap/node/'bootstrap.json').read_text())
    assert data['status']=='READ_ONLY_NEW_NODE_BOOTSTRAP_COMPLETE' and data['compute']==''
    assert sha(bootstrap/node/'preflight.py')==data['source_SHA256']
    nodes[node]=data['GPU'].split(',')[0]
for p in sorted(src.glob('*.py')):shutil.copyfile(p,bundle/p.name)
sources={p.name:sha(p) for p in sorted(bundle.glob('*.py'))}
plan=dict(status='FROZEN_CPU_SYNTHETIC_QUALIFICATION_ONLY',actualclock_freeze_UTC=sys.argv[2],
    human_lease_source='Human reply: 24h后; duration only; no platform expiry confirmation',
    conservative_cutoff_UTC='2026-10-09T15:00:00+00:00',timeout_seconds=180,
    save_reserve_seconds=7200,allowed_UUIDs=[nodes['A'],nodes['B']],
    python='/data/miniconda/envs/torch/bin/python',torch='2.1.0+cu121',numpy='1.26.4',
    sources=sources,real_data_or_weights=False,new_VAL_TEST_scores=False,
    full_training_authorized_resource_gate_passed=False,permanent_20GB_backup_path=None)
(bundle/'plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
archive=root/'source.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(bundle.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(archive) as z: assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
receipt=dict(status='SOURCE_FROZEN_BEFORE_NATIVE_EXECUTION',actualclock_UTC=sys.argv[2],source_ZIP_SHA256=sha(archive),
    source_ZIP_bytes=archive.stat().st_size,plan_SHA256=sha(bundle/'plan.json'),nodes=nodes,
    local_root=str(root),bootstrap_source_SHA_verified=True)
(root/'freeze_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('polarity_intensity_native_'+stamp)
assert shutil.disk_usage(dest.parent).free>32*1024*1024
dest.mkdir(exist_ok=False)
for p in (archive,root/'freeze_receipt.json'):
    shutil.copyfile(p,dest/p.name);assert sha(p)==sha(dest/p.name)
for node in ('A','B','C'):
    (dest/node).mkdir()
    for name in ('preflight.py','bootstrap.json'):
        p=bootstrap/node/name;shutil.copyfile(p,dest/node/name);assert sha(p)==sha(dest/node/name)
print(json.dumps(dict(**receipt,D=str(dest)),ensure_ascii=False))
