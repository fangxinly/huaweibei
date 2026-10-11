from pathlib import Path
import hashlib
s=Path('work/capture_soft_vector_v9.py').read_text(encoding='utf-8').replace('capture_soft_vector_v9.py','capture_soft_vector_v10.py')
assert "['.json','.npz','.npy','.pt']" in s;s=s.replace("['.json','.npz','.npy','.pt']","['.json','.npz','.npy','.pt','.py']")
p=Path('work/capture_soft_vector_v10.py');p.write_text(s,encoding='utf-8')
aud=Path('work/audit_soft_snapshot_v9.py').read_text(encoding='utf-8').replace('capture_soft_vector_v9.py','capture_soft_vector_v10.py');Path('work/audit_soft_snapshot_v10.py').write_text(aud,encoding='utf-8')
aud=Path('work/audit_finite_snapshot_full_extras_v1.py').read_text(encoding='utf-8').replace("finite_risk_snapshots_20261005T1534Z","finite_risk_snapshots_20261005T1538Z").replace('capture_soft_vector_v9.py','capture_soft_vector_v10.py');Path('work/audit_finite_snapshot_full_extras_v2.py').write_text(aud,encoding='utf-8')
aud=Path('work/audit_latest_finite_all_v1.py').read_text(encoding='utf-8').replace('audit_soft_snapshot_v9.py','audit_soft_snapshot_v10.py').replace('finite_risk_snapshots_20261005T1534Z','finite_risk_snapshots_20261005T1538Z').replace('finite_snapshot9_prior','finite_snapshot10_prior').replace('audit_finite_snapshot_full_extras_v1.py','audit_finite_snapshot_full_extras_v2.py');Path('work/audit_latest_finite_all_v2.py').write_text(aud,encoding='utf-8')
print(hashlib.sha256(p.read_bytes()).hexdigest())
