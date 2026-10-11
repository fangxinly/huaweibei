from pathlib import Path
import argparse,datetime,hashlib,json,shutil
a=argparse.ArgumentParser();a.add_argument('--directory',type=Path,required=True);a.add_argument('--files',nargs='+',type=Path,required=True);a.add_argument('--out',type=Path,required=True);c=a.parse_args();c.directory.mkdir(parents=True,exist_ok=False);rows=[]
for p in c.files:
 b=p.read_bytes();dest=c.directory/p.name;assert not dest.exists();dest.write_bytes(b);digest=hashlib.sha256(b).hexdigest();assert hashlib.sha256(dest.read_bytes()).hexdigest()==digest
 rows.append({'source':str(p.resolve()),'destination':str(dest),'sha256':digest,'bytes':len(b)})
r={'status':'ADDITIVE_PERMANENT_RECORDS_ALL_SHA_VERIFIED','saved_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'rows':rows,'limits':'These records are additional files, not members of an earlier immutable ZIP.'}
with (c.directory/'late_manifest.json').open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
with c.out.open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2)
print(json.dumps({'status':r['status'],'files':len(rows)}))
