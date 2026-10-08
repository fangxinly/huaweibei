"""Pin independently cloned Adam CPU step tensors for diagnostic restore."""
from pathlib import Path
p=Path(__file__).with_name('minimal_fixed_staged_training_v2.py')
s=p.read_text(encoding='utf-8')
s=s.replace('from resume_next_update_check_v1 import verify_next_update',
            'from resume_next_update_check_v2 import verify_next_update')
q=p.with_name('minimal_fixed_staged_training_v3.py')
if q.exists():raise FileExistsError('New version only')
q.write_text(s,encoding='utf-8');compile(s,str(q),'exec')
compile(q.with_name('resume_next_update_check_v2.py').read_text(),str(q.with_name('resume_next_update_check_v2.py')),'exec')
print('AST only, no Torch model or resume update executed')
