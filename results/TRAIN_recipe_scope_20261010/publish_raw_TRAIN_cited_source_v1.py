import datetime,hashlib,json,os,pathlib,shutil,subprocess,sys
P=pathlib.Path;sys.path.insert(0,str(P(__file__).parent))
from prepare_github_source_upload import sanitize
from raw_TRAIN_capture_v1 import raw,sha,seal
r=P(sys.argv[1]);dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
kind=sys.argv[2] if len(sys.argv)>2 else 'association'
assert kind in ['association','recipe']
receipt_names=['recipe_Release_receipt.json'] if kind=='recipe' else ['source_Release_receipt.json','decision_Release_receipt.json']
pubs=[json.loads((r/n).read_bytes()) for n in receipt_names]
for p in pubs:assert p['remote_digest_verified'] and sha(P(p['source']))==p['source_SHA']
git='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe';bare=dc/'publication.git'
def cmd(args,data=None):return subprocess.check_output([git,'--git-dir='+str(bare)]+args,input=data,cwd=r)
out=r/'public_source_and_summary';out.mkdir()
names=(['归一化配方适用范围.md','recipe_scope_decision.json','recipe_scope_D_receipt.json','recipe_Release_receipt.json','fetch_raw_TRAIN_recipe_sources_v1.py'] if kind=='recipe' else ['引用仓库输入来源关联.md','source_association_decision.json','fetch_ledger.json','D_preservation_receipt.json','decision_D_preservation_receipt.json','source_Release_receipt.json','decision_Release_receipt.json','fetch_raw_TRAIN_cited_source_v1.py'])
for n in names:
    shutil.copyfile(r/n,out/n)
helpers=['save_raw_TRAIN_recipe_scope_v1.py','fetch_MAG_recipe_PR10_v1.py'] if kind=='recipe' else ['save_raw_TRAIN_cited_source_decision_v1.py']
for n in helpers:shutil.copyfile(P(__file__).parent/n,out/n)
shutil.copyfile(__file__,out/P(__file__).name)
old=cmd(['rev-parse','HEAD']).decode().strip();assert old==cmd(['ls-remote','origin','refs/heads/main']).decode().split()[0]
cmd(['read-tree','HEAD']);manifest=json.loads(cmd(['show','HEAD:source_manifest.json']));records={v['path']:v for v in manifest['records']}
files=sorted(out.iterdir())
for f in files:
    rel=('results/TRAIN_recipe_scope_20261010/' if kind=='recipe' else 'results/TRAIN_cited_source_association_20261010/')+f.name;original=f.read_bytes();data,changes=sanitize(original)
    oid=cmd(['hash-object','-w','--stdin'],data).decode().strip();cmd(['update-index','--add','--cacheinfo','100644',oid,rel]);assert cmd(['show',':'+rel])==data
    records[rel]=dict(path=rel,origin=str(f),kind='cited_public_source_association',bytes=len(data),original_sha256=hashlib.sha256(original).hexdigest(),published_sha256=hashlib.sha256(data).hexdigest(),sanitizations=changes)
manifest['records']=[records[k] for k in sorted(records)];data=raw(manifest);oid=cmd(['hash-object','-w','--stdin'],data).decode().strip();cmd(['update-index','--add','--cacheinfo','100644',oid,'source_manifest.json']);assert cmd(['show',':source_manifest.json'])==data
tree=cmd(['write-tree']).decode().strip();commit=cmd(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit-tree',tree,'-p',old,'-m','Preserve cited dataset-source association without inferring preprocessing']).decode().strip()
cmd(['update-ref','refs/heads/main',commit,old]);cmd(['push','origin','refs/heads/main:refs/heads/main']);assert cmd(['ls-remote','origin','refs/heads/main']).decode().split()[0]==commit
receipt=dict(status='D_BARE_CITED_SOURCE_EXACT_BLOB_GITHUB_VERIFIED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=commit,parent_commit=old,C_checkout_not_modified=True,D_bare=str(bare),files=len(files)+1,all_published_blobs_verified=True,no_new_numeric_execution=True)
for p in [dc/'GitHub_publication_receipt.json',r/'GitHub_publication_receipt.json']:p.write_bytes(raw(receipt))
state=json.loads((dc/'D_current_research_state.json').read_bytes());state['github_source']=receipt;key='latest_raw_TRAIN_recipe_scope' if kind=='recipe' else 'latest_raw_TRAIN_cited_public_source';state[key]['GitHub_publication']=receipt;state[key]['Release_publications']=pubs;state['updated_at_utc']=receipt['actual_UTC']
for p in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
    tmp=p.with_name(p.name+'.citedpublished.tmp');tmp.write_bytes(raw(state));os.replace(tmp,p)
assert (dc/'D_current_research_state.json').read_bytes()==(c/'自主优化实际接续.json').read_bytes()
sync=dict(actual_UTC=receipt['actual_UTC'],D_C_same_SHA=sha(dc/'D_current_research_state.json'),GitHub_commit=commit,source_and_decision_Release_digest_verified=True)
(r/'publication_and_state_sync_receipt.json').write_bytes(raw(sync))
closure=r/'publication_closure';closure.mkdir()
for n in ['GitHub_publication_receipt.json','publication_and_state_sync_receipt.json']+receipt_names:shutil.copyfile(r/n,closure/n)
shutil.copyfile(__file__,closure/P(__file__).name);proof=seal(closure,'complete_actual_source_association_publication_original.zip');(r/'publication_closure_D_receipt.json').write_bytes(raw(proof))
with (c/'研究接续状态.md').open('a',encoding='utf-8') as f:f.write('\n'+receipt['actual_UTC']+' 新公开来源'+kind+'批Release远端digest与D bare exactblob GitHub'+commit+'真实过，公示闭合ZIP'+proof['archive_SHA']+'全member SHA/CRC/unique/exactset过，D/C同字节，C旧HEAD未动。不冒数据身份或中心化已证实，不重做旧批。\n')
print(json.dumps(dict(receipt=receipt,closure=proof)))
