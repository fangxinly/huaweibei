import datetime,hashlib,json,os,pathlib,shutil,subprocess,sys
P=pathlib.Path;sys.path.insert(0,str(P(__file__).parent))
from prepare_github_source_upload import sanitize
from raw_TRAIN_capture_v1 import raw,sha,seal
r=P(sys.argv[1]);dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
pub=json.loads((r/'Release_receipt.json').read_bytes());assert pub['remote_digest_verified'] and pub['source_SHA']==sha(r/'complete_actual_TRAIN_support_original.zip')
git='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe';bare=dc/'publication.git'
def cmd(args,data=None):return subprocess.check_output([git,'--git-dir='+str(bare)]+args,input=data,cwd=r)
out=r/'public_source_and_summary';out.mkdir()
for n in ['TRAIN原拟合状态支持诊断结果.md','saved_support_independent_summary.json','support_D_preservation_receipt.json','Release_receipt.json','completion_metadata_D_receipt.json','retry_session_observation.json','finalize_TRAIN_support_retry_v1.py']:shutil.copyfile(r/n,out/n)
for p in (r/'verified_original/payload').iterdir():
 if p.suffix in ['.py','.json']:shutil.copyfile(p,out/p.name)
shutil.copyfile(__file__,out/P(__file__).name)
old=cmd(['rev-parse','HEAD']).decode().strip();assert old==cmd(['ls-remote','origin','refs/heads/main']).decode().split()[0]
cmd(['read-tree','HEAD']);manifest=json.loads(cmd(['show','HEAD:source_manifest.json']));records={v['path']:v for v in manifest['records']}
files=sorted(out.iterdir())
for f in files:
 rel='results/TRAIN_saved_fit_support_20261010/'+f.name;original=f.read_bytes();data,changes=sanitize(original);oid=cmd(['hash-object','-w','--stdin'],data).decode().strip();cmd(['update-index','--add','--cacheinfo','100644',oid,rel]);assert cmd(['show',':'+rel])==data
 records[rel]=dict(path=rel,origin=str(f),kind='saved_support_source_or_analysis',bytes=len(data),original_sha256=hashlib.sha256(original).hexdigest(),published_sha256=hashlib.sha256(data).hexdigest(),sanitizations=changes)
manifest['records']=[records[k] for k in sorted(records)];data=raw(manifest);oid=cmd(['hash-object','-w','--stdin'],data).decode().strip();cmd(['update-index','--add','--cacheinfo','100644',oid,'source_manifest.json']);assert cmd(['show',':source_manifest.json'])==data
tree=cmd(['write-tree']).decode().strip();commit=cmd(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit-tree',tree,'-p',old,'-m','Preserve TRAIN saved-fit support diagnostic without changing original predictions']).decode().strip();cmd(['update-ref','refs/heads/main',commit,old]);cmd(['push','origin','refs/heads/main:refs/heads/main']);assert cmd(['ls-remote','origin','refs/heads/main']).decode().split()[0]==commit
receipt=dict(status='D_BARE_TRAIN_SUPPORT_EXACT_BLOB_GITHUB_VERIFIED',actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),commit=commit,parent_commit=old,C_checkout_not_modified=True,D_bare=str(bare),files=len(files)+1,all_published_blobs_verified=True,primary_results_unchanged=True,full_neural_video_fivefold_complete=False)
for p in [dc/'GitHub_publication_receipt.json',r/'GitHub_publication_receipt.json']:p.write_bytes(raw(receipt))
state=json.loads((dc/'D_current_research_state.json').read_bytes());state['github_source']=receipt;state['latest_TRAIN_support_diagnostic_complete']['GitHub_publication']=receipt
sent=dict(thread_ID='01a10fcb-6663-70a2-9a76-60e5634d0c03',input_original_archive_SHA=pub['source_SHA'],send_actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),send_once_confirmed=True,full_reply_received=False,no_same_batch_resend=True,scope='ANALYSIS_ONLY_NO_GPU_CREDENTIALS_OR_MAIN_SOURCE_CHANGES')
state['latest_TRAIN_support_analysis_sent']=sent;state['updated_at_utc']=receipt['actual_UTC'];state['next_gate']='Read a new complete independent support review once and decide within the saved-result scientific limits. Do not rerun the completed diagnostic or fit/score/rescue. Keep source-unit/upstream exposure uncertainty.'
for dest in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
 tmp=dest.with_name(dest.name+'.supportpublished.tmp');tmp.write_bytes(raw(state));os.replace(tmp,dest)
assert (dc/'D_current_research_state.json').read_bytes()==(c/'自主优化实际接续.json').read_bytes()
adv=json.loads((c/'研究建议交流接续.json').read_bytes());adv['latest_TRAIN_support_analysis_sent']=sent;(c/'研究建议交流接续.json').write_bytes(raw(adv))
(r/'advisor_actual_send_receipt.json').write_bytes(raw(sent));(r/'publication_and_state_sync_receipt.json').write_bytes(raw(dict(actual_UTC=state['updated_at_utc'],D_C_same_SHA=sha(dc/'D_current_research_state.json'),GitHub_commit=commit,Release_remote_digest_verified=True)))
with (c/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+receipt['actual_UTC']+' 新TRAIN支持诊断源码/报告/原保存回执从D bare exact blob GitHub '+commit+' 实际过，C旧HEAD未动。新SHA0f45efd1批已向原建议聊天发送一次仅分析，完整结果待；旧批不轮询重发。D/C同字节。\n')
closure=r/'publication_closure';closure.mkdir()
for n in ['GitHub_publication_receipt.json','advisor_actual_send_receipt.json','publication_and_state_sync_receipt.json']:shutil.copyfile(r/n,closure/n)
shutil.copyfile(__file__,closure/P(__file__).name);proof=seal(closure,'complete_actual_publication_closure_original.zip');(r/'publication_closure_D_receipt.json').write_bytes(raw(proof))
print(json.dumps({'receipt':receipt,'closure':proof}))
