import email.parser,json,zipfile
from pathlib import Path
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.markers import default_environment
from pip._vendor.packaging.version import Version
folder=Path('work/second_lease_linux_wheels_20261006T1401Z');metadata={}
canon=lambda n:n.lower().replace('_','-').replace('.','-')
for p in folder.glob('*.whl'):
 with zipfile.ZipFile(p) as z:
  n=[n for n in z.namelist() if n.endswith('.dist-info/METADATA')];m=email.parser.Parser().parsestr(z.read(n[0]).decode());metadata[canon(m['Name'])]=(m['Version'],m.get_all('Requires-Dist',[]))
env=default_environment();env.update({'os_name':'posix','sys_platform':'linux','platform_machine':'x86_64','platform_system':'Linux','python_version':'3.10','python_full_version':'3.10.14','implementation_name':'cpython','platform_python_implementation':'CPython','implementation_version':'3.10.14','extra':''})
missing=[];invalid=[];checked=[]
for name,(version,requirements) in metadata.items():
 for raw in requirements:
  req=Requirement(raw)
  if req.marker and not req.marker.evaluate(env):continue
  dest=canon(req.name);checked.append({'source':name,'requires':raw})
  if dest not in metadata:missing.append({'source':name,'requires':raw})
  elif req.specifier and Version(metadata[dest][0]) not in req.specifier:invalid.append({'source':name,'requires':raw,'got':metadata[dest][0]})
result={'scope':'Linux_x86_64_Python310_base_dependencies_only_no_extras','wheels':len(metadata),'missing':missing,'invalid':invalid,'checked_requirements':checked,'complete':not missing and not invalid}
(folder/'linux_dependency_closure.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
