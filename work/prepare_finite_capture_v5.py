from pathlib import Path
r=Path(__file__).resolve().parent
code=(r/'capture_soft_vector_v4.py').read_text(encoding='utf-8').replace('capture_soft_vector_v4.py','capture_soft_vector_v5.py')
needle=" tree(Path('/data/coding/jacobian_preservation_20261005T1406Z'),'current_preservation')"
assert code.count(needle)==1
extra="""
 finite_root=Path('/data/coding/finite_task_risk_preflight_20261005T1428Z')
 for item in sorted(finite_root.glob('*.py')):add(item,'finite_source/'+item.name)
 for item in sorted(finite_root.glob('*.json')):add(item,'finite_source/'+item.name)
 for filename in ['preflight.log','head_holdout.log','head_holdout_v2.log','full_inference.log']:
  if (finite_root/filename).exists():add(finite_root/filename,'finite_logs/'+filename)
 tree(finite_root/'checks','finite_preflight')
"""
code=code.replace(needle,needle+extra)
(r/'capture_soft_vector_v5.py').write_text(code,encoding='utf-8')
audit=(r/'audit_soft_snapshot_v4.py').read_text(encoding='utf-8').replace('capture_soft_vector_v4.py','capture_soft_vector_v5.py')
# Current old experiments still audited with exactly their frozen protocol.
# Finite preflight gets a separate source/array audit, not old100-epoch criteria.
(r/'audit_soft_snapshot_v5.py').write_text(audit,encoding='utf-8')
print('CAPTURE5_NEW_FINITE_PREFLIGHT_ROOT_INCLUDED')
