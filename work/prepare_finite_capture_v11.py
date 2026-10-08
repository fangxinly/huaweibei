from pathlib import Path
import hashlib
s=Path('work/capture_soft_vector_v10.py').read_text(encoding='utf-8').replace('capture_soft_vector_v10.py','capture_soft_vector_v11.py')
needle="  b=path.read_bytes();z.writestr(name,b);members[name]={'bytes':len(b),'sha256':sha(b)}"
assert needle in s
s=s.replace(needle,"  b=path.read_bytes();meta={'bytes':len(b),'sha256':sha(b)}\n  if name in members:\n   assert members[name]==meta;return\n  z.writestr(name,b);members[name]=meta")
p=Path('work/capture_soft_vector_v11.py');p.write_text(s,encoding='utf-8')
aud=Path('work/audit_soft_snapshot_v10.py').read_text(encoding='utf-8').replace('capture_soft_vector_v10.py','capture_soft_vector_v11.py').replace("assert set(z.namelist())==set(manifest)|{'member_manifest.json'}","assert len(z.namelist())==len(set(z.namelist())) and set(z.namelist())==set(manifest)|{'member_manifest.json'}")
Path('work/audit_soft_snapshot_v11.py').write_text(aud,encoding='utf-8')
aud=Path('work/audit_finite_snapshot_full_extras_v2.py').read_text(encoding='utf-8').replace('capture_soft_vector_v10.py','capture_soft_vector_v11.py').replace('finite_risk_snapshots_20261005T1538Z','finite_risk_snapshots_20261005T1541Z');Path('work/audit_finite_snapshot_full_extras_v3.py').write_text(aud,encoding='utf-8')
aud=Path('work/audit_latest_finite_all_v2.py').read_text(encoding='utf-8').replace('audit_soft_snapshot_v10.py','audit_soft_snapshot_v11.py').replace('finite_risk_snapshots_20261005T1538Z','finite_risk_snapshots_20261005T1541Z').replace('finite_snapshot10_prior','finite_snapshot11_prior').replace('audit_finite_snapshot_full_extras_v2.py','audit_finite_snapshot_full_extras_v3.py');Path('work/audit_latest_finite_all_v3.py').write_text(aud,encoding='utf-8')
print(hashlib.sha256(p.read_bytes()).hexdigest())
