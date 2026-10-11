import argparse,datetime,hashlib,json,zipfile
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(a):
    r=json.loads(a.receipt.read_text());assert r['child_natural_exit']==0
    paths=[a.parts/n['name'] for n in r['parts']]
    for path,record in zip(paths,r['parts']):assert path.stat().st_size==record['bytes'] and sha(path)==record['SHA']
    assert not a.root.exists();a.root.mkdir();zp=a.root/'complete_original_training_capture.zip'
    with zp.open('xb') as out:
        for path in paths:
            with path.open('rb') as f:
                for block in iter(lambda:f.read(8*1024**2),b''):out.write(block)
    assert zp.stat().st_size==r['archive_bytes'] and sha(zp)==r['archive_SHA']
    target=a.root/'extracted';target.mkdir()
    with zipfile.ZipFile(zp) as z:
        assert len(z.namelist())==len(set(z.namelist()));manifest=json.loads(z.read('member_SHA.json'));assert set(z.namelist())==set(manifest)|{'member_SHA.json'}
        for n,h in manifest.items():
            dest=(target/n).resolve();assert target.resolve() in dest.parents;dest.parent.mkdir(parents=True,exist_ok=True);digest=hashlib.sha256()
            with z.open(n) as f,dest.open('xb') as out:
                for b in iter(lambda:f.read(8*1024**2),b''):digest.update(b);out.write(b)
            assert digest.hexdigest()==h
        (target/'member_SHA.json').write_bytes(z.read('member_SHA.json'))
    receipt=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),full_original_archive_SHA=r['archive_SHA'],all_member_SHA_CRC_unique=True,not_local_manufactured_training=True,original_PID=r['child_PID'],original_natural_exit=0,root=str(target))
    (a.root/'actual_restore_receipt.json').write_text(json.dumps(receipt,indent=2));print(receipt)
if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('receipt','parts','root'):p.add_argument('--'+n,type=Path,required=True)
    run(p.parse_args())
