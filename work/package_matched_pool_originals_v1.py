from pathlib import Path
import argparse,datetime,hashlib,json,shutil,zipfile
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root
sha=lambda b:hashlib.sha256(b).hexdigest()
assert shutil.disk_usage('D:/').free>1024**3
assert json.loads((r/'evaluation/independent_audit.json').read_text())['status']=='EVAL_FIXED_ROLE_AND_RISK_METRICS_INDEPENDENTLY_RECONSTRUCTED'
out=r/'preservation';out.mkdir(exist_ok=False);manifest={}
files=[p for p in r.iterdir() if p.is_file() and p.suffix in ['.json','.py','.npz','.log','.txt']]
for phase in ['precheck','execute','evaluation']:files.extend(p for p in (r/phase).iterdir() if p.is_file())
with zipfile.ZipFile(out/'snapshot.zip','x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files:
        name=str(p.relative_to(r)).replace('\\','/');assert name not in manifest and name!='metric_labels.npz'
        b=p.read_bytes();manifest[name]=dict(bytes=len(b),sha256=sha(b));z.writestr(name,b)
    z.writestr('member_manifest.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(out/'snapshot.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
data=(out/'snapshot.zip').read_bytes();receipt=dict(status='NEW_MATCHED_POOL_ORIGINALS_PERMANENT_D_PACKAGED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),bytes=len(data),sha256=sha(data),files=len(manifest),full_TRAIN_label_archive_included=False,only_post_freeze_EVAL_y_in_metric_arrays=True)
(out/'receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
