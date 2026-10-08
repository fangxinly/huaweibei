from pathlib import Path
import argparse, datetime, hashlib, json, zipfile
p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
sha=lambda b:hashlib.sha256(b).hexdigest()
d=a.directory;r=json.loads((d/'preservation_execute/receipt.json').read_text())
raw=(d/'preservation_execute/snapshot.zip').read_bytes()
assert r['status']=='ACTUAL_COLLECTION_ORIGINAL_SMALL_FILES_ATOMIC_PACKAGE_COMPLETE'
assert len(raw)==r['bytes'] and sha(raw)==r['sha256']
assert sha((Path(__file__).parent/'package_same_teacher_collection_v1.py').read_bytes())==r['source_sha256']
with zipfile.ZipFile(d/'preservation_execute/snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    m=json.loads(z.read('member_manifest.json'));assert set(z.namelist())==set(m)|{'member_manifest.json'}
    for name,meta in m.items():
        target=d/name;assert target.resolve().is_relative_to(d.resolve())
        data=z.read(name);assert len(data)==meta['bytes'] and sha(data)==meta['sha256']
        if target.is_file():assert target.read_bytes()==data,name
        else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
assert json.loads((d/'execute_exit.json').read_text())['exit_code']==0
assert json.loads((d/'execute/independent_execute_audit.json').read_text())['status']=='ACTUAL_COLLECTION_FIT_INNER_SCALAR_ARRAYS_INDEPENDENT_AUDIT_PASSED'
out=d/'preservation_execute/independent_package_audit.json';assert not out.exists()
out.write_text(json.dumps({'status':'ACTUAL_COLLECTION_ORIGINAL_PACKAGE_D_SHA_CRC_ALL_MEMBERS_VERIFIED',
    'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'package_sha256':r['sha256'],'members':m,
    'source_sha256':sha(Path(__file__).read_bytes()),'old_full_weights_transferred':False},indent=2))
print('ACTUAL_COLLECTION_ORIGINAL_PACKAGE_D_SHA_CRC_ALL_MEMBERS_VERIFIED',len(m))
