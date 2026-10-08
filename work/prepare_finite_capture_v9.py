from pathlib import Path
import hashlib
s=Path('work/capture_soft_vector_v8.py').read_text(encoding='utf-8').replace('capture_soft_vector_v8.py','capture_soft_vector_v9.py')
needle=" if (singleton/'precheck.log').exists():add(singleton/'precheck.log','finite_single_token_precheck/precheck.log')"
s=s.replace(needle,needle+"\n for n in ['warmup.log','full_raw.log']:\n  if (singleton/n).exists():add(singleton/n,'finite_single_token_precheck/'+n)")
p=Path('work/capture_soft_vector_v9.py');p.write_text(s,encoding='utf-8')
aud=Path('work/audit_soft_snapshot_v8.py').read_text(encoding='utf-8').replace('capture_soft_vector_v8.py','capture_soft_vector_v9.py');Path('work/audit_soft_snapshot_v9.py').write_text(aud,encoding='utf-8');print(hashlib.sha256(p.read_bytes()).hexdigest())
