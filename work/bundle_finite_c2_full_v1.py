from pathlib import Path
import argparse,json,hashlib,tarfile,datetime,shutil
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();r=a.directory;sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
m=json.loads((r/'preservation_manifest.json').read_text());assert m['training_node']=='c' and m['assembly_node']=='a'
names=list(m['required_seven_files'])+list(m['extra_files'])+['preservation_manifest.json']
for group in ['required_seven_files','extra_files']:
 for n,v in m[group].items():assert sha(r/n)==v['sha256'] and (r/n).stat().st_size==v['bytes']
archive=r.parent/'c2_full_preservation.tar';assert not archive.exists()
with tarfile.open(archive,'w') as t:
 for n in names:t.add(r/n,arcname=n,recursive=False)
receipt=dict(status='C2_FULL_SEVEN_PLUS_TWELVE_ARCHIVE_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256=sha(archive),bytes=archive.stat().st_size,manifest=m,source_sha256=sha(__file__),data_disk_free=shutil.disk_usage('/data').free)
(r.parent/'c2_full_archive_receipt.json').write_text(json.dumps(receipt,indent=2));print('C2_FULL_PRESERVATION_BUNDLE_READY',archive.stat().st_size,flush=True)
