from pathlib import Path
import datetime,hashlib,json,shutil,zipfile
work=Path(__file__).resolve().parent
d=Path('D:/CodexBackups/selective_flow_20261003_1105/train_oracle_completed_20261006T0158Z/c')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert shutil.disk_usage(d).free>1073741824
shutil.copy2(work/'verify_train_oracle_cpu_v1.py',d/'verify_train_oracle_cpu_v1.py')
files=['diagnose_train_oracle_v2.py','train_oracle_plan_v2.json','audit_train_oracle_v1.py',
       'capture_soft_vector_v15.py','verify_train_oracle_cpu_v1.py','execution.log','exit_code.txt',
       'out/receipt.json','out/diagnostics.npz']
assert (d/'exit_code.txt').read_text().strip()=='0'
r=json.loads((d/'out/receipt.json').read_text());assert sha(d/'out/diagnostics.npz')==r['diagnostics_sha256']
manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),diagnostic_node='c',
    target_cpu_node='b',target_cpu_uuid='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067',
    cpu_source_sha256=sha(d/'verify_train_oracle_cpu_v1.py'),
    scope='Original C diagnostic, local D preservation, CPU B pending original receipt; no new fullweights.',
    files={n:dict(bytes=(d/n).stat().st_size,sha256=sha(d/n)) for n in files})
(d/'preservation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
bundle=d/'cpu_bundle.zip';assert not bundle.exists()
with zipfile.ZipFile(bundle,'x',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for n in files+['preservation_manifest.json']:z.write(d/n,n)
with zipfile.ZipFile(bundle) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
print(json.dumps(dict(bundle=str(bundle),bytes=bundle.stat().st_size,sha256=sha(bundle),files=len(files))))
