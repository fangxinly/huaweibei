from pathlib import Path
w=Path(__file__).parent
s=(w/'audit_finite_c2_full_local_v2.py').read_text(encoding='utf-8').replace('.read_text()',".read_text(encoding='utf-8')")
(w/'audit_finite_c2_full_local_v4.py').write_text(s,encoding='utf-8')
