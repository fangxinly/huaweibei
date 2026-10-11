from pathlib import Path
import hashlib
p=Path('work/capture_soft_vector_v6.py');s=p.read_text(encoding='utf-8')
s=s.replace('capture_soft_vector_v6.py','capture_soft_vector_v7.py')
needle=" tree(formal/'run','finite_run')"
add=""" tree(formal/'run','finite_run')
 tree(formal/'full_checkpoints','finite_full_checkpoints')
 tree(Path('/data/coding/finite_preservation_20261005T1515Z'),'finite_preservation')
 failed=Path('/data/coding/failed_finite_vector_diagnostics_20261005T1503Z')
 tree(failed,'finite_failed_diagnostics')
 if (failed/'anomaly_trace.txt').exists():add(failed/'anomaly_trace.txt','finite_failed_diagnostics/anomaly_trace.txt')
 for name in ['training.log','wrapper.log','assembly_a.log','assembly_b.log']:
  if (formal/name).exists():add(formal/name,'finite_formal_logs/'+name)
 if (finite_root/'failed_vector_diagnostic.log').exists():add(finite_root/'failed_vector_diagnostic.log','finite_failed_diagnostics/diagnostic.log')
"""
assert needle in s;s=s.replace(needle,add)
Path('work/capture_soft_vector_v7.py').write_text(s,encoding='utf-8')
audit=Path('work/audit_soft_snapshot_v6.py').read_text(encoding='utf-8').replace('capture_soft_vector_v6.py','capture_soft_vector_v7.py');Path('work/audit_soft_snapshot_v7.py').write_text(audit,encoding='utf-8')
print(hashlib.sha256(s.encode()).hexdigest())
