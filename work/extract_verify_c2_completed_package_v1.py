from pathlib import Path
import argparse,json,tarfile,hashlib
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--receipt-name',default='receipt.json');a=p.parse_args();d=a.directory
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();r=json.loads((d/a.receipt_name).read_text());archive=d/'completed.tar.gz'
assert archive.stat().st_size==r['package_bytes'] and sha(archive)==r['package_sha256']
with tarfile.open(archive) as t:
 assert set(t.getnames())==set(r['members'])
 for x in t.getmembers():assert x.isfile() and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts
 t.extractall(d)
for n,x in r['members'].items():assert sha(d/n)==x['sha256'] and (d/n).stat().st_size==x['bytes']
print('C2_ACTUAL_COMPLETE_PACKAGE_SOURCE_SHA_VERIFIED')
