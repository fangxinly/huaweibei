from pathlib import Path
import hashlib,json
w=Path(__file__).parent;s=(w/'capture_soft_vector_v12.py').read_text(encoding='utf-8').replace('capture_soft_vector_v12.py','capture_soft_vector_v13.py')
old=" z.writestr('large_file_manifest.json',json.dumps(large,indent=2));"
new=""" assembly=Path('/data/coding/finite_c2_assembly_20261005T1615Z')
 tree(assembly,'finite_c2_assembly')
 for folder,prefix in [(assembly,'finite_c2_assembly'),(c2,'finite_c2')]:
  for n in ['assembly.log','diagnostics_v4.log']:
   if (folder/n).exists():add(folder/n,prefix+'/'+n)
 tree(Path('/data/coding/finite_c2_preservation_20261005T1620Z'),'finite_c2_preservation')

"""+old
assert old in s;s=s.replace(old,new)
p=w/'capture_soft_vector_v13.py';p.write_text(s,encoding='utf-8')
s=(w/'audit_soft_snapshot_v12.py').read_text(encoding='utf-8').replace('capture_soft_vector_v12.py','capture_soft_vector_v13.py');(w/'audit_soft_snapshot_v13.py').write_text(s,encoding='utf-8')
print(json.dumps({'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
