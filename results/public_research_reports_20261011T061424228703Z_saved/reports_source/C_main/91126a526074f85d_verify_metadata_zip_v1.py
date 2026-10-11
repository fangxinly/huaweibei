from pathlib import Path
import argparse,datetime,hashlib,json,shutil,zipfile
a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True);c=a.parse_args();d=c.directory
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
m=json.loads((d/'manifest.json').read_text());assert sha(d/'metadata.zip')==m['archive_sha256'] and (d/'metadata.zip').stat().st_size==m['archive_bytes']
with zipfile.ZipFile(d/'metadata.zip') as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(m['members'])
 assert set(z.namelist())=={e['name'] for e in m['members']}
 for e in m['members']:
  b=z.read(e['name']);assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']
r={'verified_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'status':'INDEPENDENT_CPU_METADATA_ZIP_AND_ALL_MEMBERS_SHA_VERIFIED','directory':str(d),'archive_sha256':m['archive_sha256'],'members':len(m['members']),'manifest_sha256':sha(d/'manifest.json'),'free_bytes':shutil.disk_usage(d).free,'verification_source_sha256':sha(Path(__file__))}
with (d/'destination_verification.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
