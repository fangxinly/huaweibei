"""Publish a small evidence tree from authoritative D bare state, no checkout reset."""
import datetime as dt,hashlib,json,pathlib,subprocess
from raw_TRAIN_capture_v1 import raw,sha
from prepare_github_source_upload import sanitize
from publish_test_selected_group5_preparation_v1 import DC,GIT
P=pathlib.Path

def publish_tree(folder,prefix,message):
    folder=P(folder);bare=DC/'publication.git'
    def git(args,data=None):return subprocess.check_output([GIT,'--git-dir='+str(bare)]+args,input=data,cwd=folder)
    parent=git(['rev-parse','HEAD']).decode().strip()
    assert parent==json.loads((DC/'GitHub_publication_receipt.json').read_bytes())['commit']
    assert git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==parent
    git(['read-tree','HEAD']);inventory=json.loads(git(['show','HEAD:source_manifest.json']));records={v['path']:v for v in inventory['records']};published=[]
    for p in sorted(folder.rglob('*')):
        if not p.is_file() or p.suffix=='.zip':continue
        if p.stat().st_size>10*1024**2:raise ValueError('Only small source/evidence enters Git')
        original=p.read_bytes();b,changes=sanitize(original)
        rel=prefix+'/'+p.relative_to(folder).as_posix();oid=git(['hash-object','-w','--stdin'],b).decode().strip()
        git(['update-index','--add','--cacheinfo','100644',oid,rel]);assert git(['show',':'+rel])==b
        records[rel]=dict(path=rel,origin=str(p),kind='group5_actual_evidence_and_source',bytes=len(b),original_sha256=sha(p),published_sha256=hashlib.sha256(b).hexdigest(),sanitizations=changes);published.append(rel)
    inventory['records']=[records[k] for k in sorted(records)];oid=git(['hash-object','-w','--stdin'],raw(inventory)).decode().strip()
    git(['update-index','--add','--cacheinfo','100644',oid,'source_manifest.json']);tree=git(['write-tree']).decode().strip()
    commit=git(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit-tree',tree,'-p',parent,'-m',message]).decode().strip()
    git(['update-ref','refs/heads/main',commit,parent]);git(['push','origin','refs/heads/main:refs/heads/main'])
    assert git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==commit
    for rel in published:assert hashlib.sha256(git(['show',commit+':'+rel])).hexdigest()==records[rel]['published_sha256']
    receipt=dict(status='D_BARE_GROUP5_EXACT_BLOB_GITHUB_VERIFIED',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),commit=commit,parent_commit=parent,files=len(published)+1,all_published_blobs_verified=True,C_checkout_not_modified=True)
    (DC/'GitHub_publication_receipt.json').write_bytes(raw(receipt));return receipt
