import hashlib,json,subprocess
from pathlib import Path
base=Path(__file__).resolve().parents[1];repo=base/'github_upload_20261008T133346Z'
manifest=json.loads((repo/'source_manifest.json').read_text(encoding='utf-8'))
rows=subprocess.check_output(['git','ls-files','--stage','-z'],cwd=repo).split(b'\0')
index={}
for row in rows:
    if row:
        meta,path=row.split(b'\t',1);index[path.decode()]=meta.split()[1].decode()
records={r['path']:r for r in manifest['records']}
paths=list(records);objects=''.join(index[p]+'\n' for p in paths).encode()
raw=subprocess.check_output(['git','cat-file','--batch'],input=objects,cwd=repo)
cursor=0;failures=[]
for path in paths:
    end=raw.index(b'\n',cursor);head=raw[cursor:end].split();size=int(head[2]);start=end+1
    blob=raw[start:start+size];cursor=start+size+1
    if hashlib.sha256(blob).hexdigest()!=records[path]['published_sha256']:failures.append(path)
assert cursor==len(raw)
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/main'],cwd=repo,text=True).split()[0]
status=subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True)
report=dict(commit=head,remote_main=remote,remote_matches=head==remote,index_blob_SHA_failures=failures,
            tracked_files=len(index),manifest_files=len(records),checkout_clean=not status)
out=base/'outputs/github_polarity_update_20261008T153225Z_v2'
(out/('blob_verification_'+head[:12]+'.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
assert not failures and head==remote and not status
