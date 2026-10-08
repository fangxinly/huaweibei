"""Preserve actual original collection files; contains no old full weights."""
from pathlib import Path
import argparse, datetime, hashlib, json, zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
assert read(a.root/'execute_exit.json')['exit_code']==0
assert read(a.root/'execute/independent_execute_audit.json')['status']=='ACTUAL_COLLECTION_FIT_INNER_SCALAR_ARRAYS_INDEPENDENT_AUDIT_PASSED'
out=a.root/'preservation_execute';out.mkdir(exist_ok=False)
members={};pending=out/'snapshot.pending'
with zipfile.ZipFile(pending,'x',zipfile.ZIP_DEFLATED) as z:
    for path in sorted(a.root.rglob('*')):
        if not path.is_file() or out in path.parents or '__pycache__' in path.parts or '.pending' in path.name:continue
        if path.suffix not in ['.json','.npz','.npy','.py','.log','.txt']:continue
        assert path.stat().st_size<8*1024**2
        name=path.relative_to(a.root).as_posix();data=path.read_bytes()
        z.writestr(name,data);members[name]={'bytes':len(data),'sha256':sha(data)}
    z.writestr('member_manifest.json',json.dumps(members,indent=2))
with zipfile.ZipFile(pending) as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
final=out/'snapshot.zip';pending.replace(final)
r={'status':'ACTUAL_COLLECTION_ORIGINAL_SMALL_FILES_ATOMIC_PACKAGE_COMPLETE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
   'sha256':sha(final.read_bytes()),'bytes':final.stat().st_size,'members':len(members),
   'source_sha256':sha(Path(__file__).read_bytes()),'full_weights_transferred':False,'phase':'execute'}
(out/'receipt.json').write_text(json.dumps(r,indent=2))
print(r['status'],flush=True)
