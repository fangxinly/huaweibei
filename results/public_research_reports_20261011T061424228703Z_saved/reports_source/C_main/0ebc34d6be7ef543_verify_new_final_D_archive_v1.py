"""Read-only verification of a newly downloaded complete original archive."""
import argparse,datetime,hashlib,json,os,shutil,sys,zipfile
from pathlib import Path

def digest(f):
    h=hashlib.sha256()
    for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('capture','archive','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();r=json.loads(a.capture.read_text(encoding='utf8'))
    assert r['natural_exit']==0 and r['status']=='ACTUAL_RESUMED_TRAIN100_COMPLETE' and r['ZIP_CRC_unique_all_members']
    assert a.archive.stat().st_size==r['archive_bytes']
    with a.archive.open('rb') as f:assert digest(f)==r['archive_SHA']
    with zipfile.ZipFile(a.archive) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        members=json.loads(z.read('member_SHA.json'))
        for n,h in members.items():
            with z.open(n) as f:assert digest(f)==h
        assert set(z.namelist())==set(members)|{'member_SHA.json'}
    result=dict(status='NEW_D_COMPLETE_ORIGINAL_SHA_CRC_UNIQUE_ALL_MEMBERS_PASSED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),fullargv=[sys.executable]+sys.argv,archive=str(a.archive.resolve()),archive_SHA=r['archive_SHA'],archive_bytes=r['archive_bytes'],member_count=len(members),free_after=shutil.disk_usage(a.archive.parent).free,large_original=True,old_files_modified_or_deleted=False)
    a.out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
