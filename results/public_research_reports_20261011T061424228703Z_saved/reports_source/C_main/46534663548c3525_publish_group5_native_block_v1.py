"""Publish new local evidence and source via D exact blobs; no remote-node calls."""
import datetime as dt,hashlib,json,pathlib,shutil,subprocess,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'work'))
from raw_TRAIN_capture_v1 import raw,sha,seal
from raw_TRAIN_save_and_publish_v1 import publish
from prepare_github_source_upload import sanitize
from publish_test_selected_group5_preparation_v1 import DC,OUT,GIT,sync

def main():
 root=P(json.loads((BASE/'work/group5_native_block_pointer.json').read_bytes())['root'])
 proof=json.loads((root/'D_metadata_preservation_receipt.json').read_bytes())
 publish(P(proof['archive']),proof['archive_SHA'],'group5-native-review-block-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
 public=root/'public';public.mkdir()
 for p in list(root.iterdir()):
  if p.is_file() and p.suffix in ('.py','.json','.txt','.md'):
   b,changes=sanitize(p.read_bytes());(public/p.name).write_bytes(b)
 for name in ['group5_direct_runtime_v1.py','group5_fresh_direct_session_v1.py','group5_release_transport_v1.py','group5_teacher_optimizer_v1.py','assemble_group5_pipeline_v1.py','check_group5_transport_v1.py']:
  b,changes=sanitize((BASE/'work'/name).read_bytes());(public/name).write_bytes(b)
 for prefix,source,names in [
  ('transport',DC.parent/'g5transport_20261010T031621Z',['qualification.json','manifest.json','remote_upload_receipt.json','actual_remote_restoration.json','local_qualification.json']),
  ('runtime_source',DC.parent/'g5pipe_20261010T032228Z',['D_preservation_receipt.json','Release_receipt.json','range_upload_receipt.json'])]:
  dest=public/prefix;dest.mkdir()
  for name in names:
   b,changes=sanitize((source/name).read_bytes());(dest/name).write_bytes(b)
 bare=DC/'publication.git'
 def git(args,data=None):return subprocess.check_output([GIT,'--git-dir='+str(bare)]+args,input=data,cwd=root)
 parent=git(['rev-parse','HEAD']).decode().strip()
 assert parent==json.loads((DC/'GitHub_publication_receipt.json').read_bytes())['commit']
 assert git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==parent
 git(['read-tree','HEAD']);inventory=json.loads(git(['show','HEAD:source_manifest.json']));records={v['path']:v for v in inventory['records']}
 published=[];prefix='results/group5_native_precheck_and_transport_20261010'
 for p in sorted(public.rglob('*')):
  if not p.is_file():continue
  rel=prefix+'/'+p.relative_to(public).as_posix();b=p.read_bytes()
  oid=git(['hash-object','-w','--stdin'],b).decode().strip()
  git(['update-index','--add','--cacheinfo','100644',oid,rel]);assert git(['show',':'+rel])==b
  records[rel]=dict(path=rel,origin=str(p),kind='group5_native_observation_pending_CPU_and_full_capture_not_formal_training',bytes=len(b),original_sha256=sha(p),published_sha256=sha(p),sanitizations={})
  published.append(rel)
 inventory['records']=[records[k] for k in sorted(records)];b=raw(inventory)
 oid=git(['hash-object','-w','--stdin'],b).decode().strip();git(['update-index','--add','--cacheinfo','100644',oid,'source_manifest.json'])
 tree=git(['write-tree']).decode().strip()
 commit=git(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit-tree',tree,'-p',parent,'-m','Preserve fivefold native precheck observation and CPU input review block']).decode().strip()
 git(['update-ref','refs/heads/main',commit,parent]);git(['push','origin','refs/heads/main:refs/heads/main'])
 assert git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==commit
 for rel in published:assert hashlib.sha256(git(['show',commit+':'+rel])).hexdigest()==records[rel]['published_sha256']
 receipt=dict(status='D_BARE_GROUP5_NATIVE_OBSERVATION_AND_REVIEW_BLOCK_EXACT_BLOB_VERIFIED',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),commit=commit,parent_commit=parent,C_checkout_not_modified=True,all_published_blobs_verified=True,files=len(published)+1,real_large_checkpoint_Release_saved=False,independent_CPU_passed=False)
 for p in [root/'GitHub_publication_receipt.json',DC/'GitHub_publication_receipt.json']:p.write_bytes(raw(receipt))
 state=json.loads((DC/'D_current_research_state.json').read_bytes());entry=state['latest_human_TEST_selected_group5']
 entry['latest_native_metadata_publication']=dict(github=receipt,Release=json.loads((root/'Release_receipt.json').read_bytes()))
 state['github_source']=receipt;sync(state)
 closure=root/'publication_closure';closure.mkdir()
 for name in ['GitHub_publication_receipt.json','D_metadata_preservation_receipt.json','Release_receipt.json']:shutil.copyfile(root/name,closure/name)
 (closure/'state_sync_receipt.json').write_bytes(raw(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),D_C_SHA=sha(DC/'D_current_research_state.json'),current_remote_block_unchanged=True)))
 closed=seal(closure,'complete_actual_publication_closure.zip');(root/'closure_D_receipt.json').write_bytes(raw(closed))
 with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+receipt['actual_UTC']+' 新原生预检终端观察/输入审批拒绝元数据与源码D bare exactblob GitHub'+commit+'、新元数据ZIP Release digest过；完整模型2.22GB尚未CPU审核/上传，不能冒大状态保存成功。不连SSH重试，D/C同字节。\n')
 print(json.dumps(dict(GitHub=receipt,closure=closed),ensure_ascii=False))
if __name__=='__main__':main()
