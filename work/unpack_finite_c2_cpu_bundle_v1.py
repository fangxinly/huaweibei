from pathlib import Path
import argparse,tarfile,json,hashlib,subprocess,shutil,datetime
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--expected-uuid',required=True);a=p.parse_args();d=a.directory
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();r=json.loads((d/'c2_full_archive_receipt.json').read_text());archive=d/'c2_full_preservation.tar'
assert sha(archive)==r['sha256'] and archive.stat().st_size==r['bytes']
uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip();assert uuid==a.expected_uuid and uuid=='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'
with tarfile.open(archive) as t:
 assert set(t.getnames())==set(r['manifest']['required_seven_files'])|set(r['manifest']['extra_files'])|{'preservation_manifest.json'}
 for x in t.getmembers():assert x.isfile() and Path(x.name).name==x.name
 t.extractall(d)
print('C2_OUTSIDE_TRAINING_AND_ASSEMBLY_CPU_ARCHIVE_VERIFIED',datetime.datetime.now(datetime.timezone.utc).isoformat(),shutil.disk_usage('/data').free,flush=True)
