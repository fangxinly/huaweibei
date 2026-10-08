from pathlib import Path
import hashlib,json
w=Path(__file__).parent;s=(w/'capture_soft_vector_v13.py').read_text(encoding='utf-8').replace('capture_soft_vector_v13.py','capture_soft_vector_v14.py');needle=" z.writestr('large_file_manifest.json',json.dumps(large,indent=2));";assert needle in s
s=s.replace(needle," tree(Path('/data/coding/train_group_mapping_20261005T1628Z'),'train_group_mapping')\n tree(Path('/data/coding/train_group_mapping_source_20261005T1628Z'),'train_group_mapping_source')\n"+needle)
p=w/'capture_soft_vector_v14.py';p.write_text(s,encoding='utf-8')
for origin,target in [('audit_soft_snapshot_v13.py','audit_soft_snapshot_v14.py'),('audit_finite_snapshot_full_extras_v5.py','audit_finite_snapshot_full_extras_v6.py'),('audit_finite_c2_diagnostics_completed_snapshot_v2.py','audit_finite_c2_diagnostics_completed_snapshot_v3.py'),('audit_finite_c2_full_cpu_snapshot_refs_v1.py','audit_finite_c2_full_cpu_snapshot_refs_v2.py')]:
 s=(w/origin).read_text(encoding='utf-8').replace('capture_soft_vector_v13.py','capture_soft_vector_v14.py');(w/target).write_text(s,encoding='utf-8')
print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
