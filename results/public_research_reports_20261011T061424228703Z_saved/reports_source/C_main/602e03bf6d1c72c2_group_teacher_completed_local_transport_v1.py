"""Audit D originals and make one preserved transport package for independent CPU."""
from pathlib import Path
import argparse,datetime,json,hashlib,shutil,zipfile
from audit_group_teacher_completed_files_v1 import audit,sha
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--node',choices=['a','b','c'],required=True);a=p.parse_args()
d=a.directory;m=json.loads((d/'preservation_manifest.json').read_text(encoding='utf-8'))
assert m['training_node']==m['assembly_node']==a.node
for group in ['required_seven_files','extra_files']:
 for n,meta in m[group].items():assert (d/n).stat().st_size==meta['bytes'] and sha(d/n)==meta['sha256']
r=audit(d,['a','b','c'].index(a.node))
out=d/'local_completed_preservation_audit.json';assert not out.exists()
out.write_text(json.dumps(r,indent=2),encoding='utf-8')
assert shutil.disk_usage(d).free>4*1024**3
dest=d/'independent_cpu_transport.zip';assert not dest.exists()
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_STORED) as z:
 for n in list(m['required_seven_files'])+list(m['extra_files'])+['preservation_manifest.json']:z.write(d/n,n)
with zipfile.ZipFile(dest) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==28
report={'status':'D_FULL_COMPLETED100_SEVEN_AND_TWENTY_EXTRA_SHA_CRC_SELECTION_ARRAYS_VERIFIED_CPU_TRANSPORT_READY_NOT_CPU_COMPLETE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':a.node,'directory':str(d),'manifest_sha256':sha(d/'preservation_manifest.json'),'transport_sha256':sha(dest),'transport_bytes':dest.stat().st_size,'array_audit':r}
(d/'independent_cpu_transport_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
