from pathlib import Path
import argparse,datetime,hashlib,json,shutil,zipfile
a=argparse.ArgumentParser();a.add_argument('--tag',required=True);a.add_argument('--folders',nargs='+',required=True);a.add_argument('--out',type=Path,required=True);c=a.parse_args()
assert c.tag.isalnum();root=Path(__file__).resolve().parents[1];dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('metadata_'+c.tag);assert shutil.disk_usage(dest.parent).free>100_000_000;dest.mkdir(exist_ok=False)
files={}
for folder in [root/'outputs']+[root/f for f in c.folders]:
 assert folder.is_dir()
 for p in sorted(folder.rglob('*')):
  if p.is_file() and p.suffix not in ['.pt','.tmp','.pyc'] and '__pycache__' not in p.parts:files[p.relative_to(root).as_posix()]=p
for p in (root/'work').glob('*.py'):files[p.relative_to(root).as_posix()]=p
members=[]
with zipfile.ZipFile(dest/'metadata.zip','x',zipfile.ZIP_DEFLATED) as z:
 for name,p in sorted(files.items()):
  b=p.read_bytes();z.writestr(name,b);members.append({'name':name,'source':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()})
with zipfile.ZipFile(dest/'metadata.zip') as z:
 assert len(z.namelist())==len(set(z.namelist()))==len(members)
 for e in members:
  b=z.read(e['name']);assert len(b)==e['bytes'] and hashlib.sha256(b).hexdigest()==e['sha256']==hashlib.sha256(Path(e['source']).read_bytes()).hexdigest()
p=dest/'metadata.zip';manifest={'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive_bytes':p.stat().st_size,'archive_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'members':members,'limits':'No full weights; original files retained. Reports updated after packing must be saved additively.'};(dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
receipt={'status':'LOCAL_ORIGINAL_AND_PERMANENT_D_ZIP_ALL_SHA_VERIFIED','created_at':manifest['created_at'],'destination':str(dest),'archive_bytes':manifest['archive_bytes'],'archive_sha256':manifest['archive_sha256'],'members':len(members),'manifest_sha256':hashlib.sha256((dest/'manifest.json').read_bytes()).hexdigest(),'drive_D_free_bytes':shutil.disk_usage(dest).free}
assert not c.out.exists();c.out.write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(receipt))
