from pathlib import Path
import hashlib
s=Path('work/capture_soft_vector_v7.py').read_text(encoding='utf-8').replace('capture_soft_vector_v7.py','capture_soft_vector_v8.py')
needle=" tree(Path('/data/coding/finite_preservation_20261005T1515Z'),'finite_preservation')"
extra="""
 singleton=Path('/data/coding/finite_single_token_precheck_20261005T1525Z')
 tree(singleton,'finite_single_token_precheck')
 for f in sorted(singleton.glob('*.py')):add(f,'finite_single_token_precheck/'+f.name)
 if (singleton/'precheck.log').exists():add(singleton/'precheck.log','finite_single_token_precheck/precheck.log')
"""
assert needle in s;s=s.replace(needle,needle+extra)
for suffix in ['zero_norm_probe.log','zero_norm_probe_v2.log','diagnose_finite_zero_norm_v2.py']:
 s=s.replace(" tree(failed,'finite_failed_diagnostics')", " tree(failed,'finite_failed_diagnostics')\n if (failed/'"+suffix+"').exists():add(failed/'"+suffix+"','finite_failed_diagnostics/"+suffix+"')")
p=Path('work/capture_soft_vector_v8.py');p.write_text(s,encoding='utf-8')
aud=Path('work/audit_soft_snapshot_v7.py').read_text(encoding='utf-8').replace('capture_soft_vector_v7.py','capture_soft_vector_v8.py');Path('work/audit_soft_snapshot_v8.py').write_text(aud,encoding='utf-8');print(hashlib.sha256(p.read_bytes()).hexdigest())
