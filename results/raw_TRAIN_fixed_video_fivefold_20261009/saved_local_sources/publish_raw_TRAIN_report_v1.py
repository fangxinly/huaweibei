import argparse,hashlib,json,pathlib,subprocess,sys
P=pathlib.Path;p=argparse.ArgumentParser();p.add_argument('root',type=P);p.add_argument('--stamp',required=True);a=p.parse_args();r=a.root
sys.path.insert(0,str(P(__file__).parent));from prepare_github_source_upload import sanitize
dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z';bare=dc/'publication.git';git='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
def cmd(args,data=None):return subprocess.check_output([git,'--git-dir='+str(bare)]+args,input=data,cwd=r)
old=cmd(['rev-parse','HEAD']).decode().strip();assert old==cmd(['ls-remote','origin','refs/heads/main']).decode().split()[0]
cmd(['read-tree','HEAD']);manifest=json.loads(cmd(['show','HEAD:source_manifest.json']));records={v['path']:v for v in manifest['records']}
names=['TRAIN固定视频五折实际结果.md','raw_TRAIN_complete_D_receipt.json','prediction_D_preservation_receipt.json','prediction_Release_receipt.json','closed_Release_receipt.json','prepared_reference.json','D_C_state_sync_receipt.json','analysis_batch_sent_receipt.json']
files=[r/n for n in names]+list((r/'bundle').glob('*.py'))+list((r/'bundle').glob('*.json'))+list((r/'saved_local_sources').glob('*.py'))
for n in ['natural_exit.json','audit_report_natural_exit.json','remote_qualification.json','audit_report_qualification.json','analysis/TRAIN_finite_prediction_report.json','analysis/saved_fit_CPU_audit.json','prediction/TRAIN_decoder_ledger.json','prediction/fit_identity_records.json','prediction/fresh_preflight.json','prediction/prediction_result.json']:
 files.append(r/'closed_verified_original'/n)
for f in files:
 rel='results/raw_TRAIN_fixed_video_fivefold_20261009/'+f.relative_to(r).as_posix();original=f.read_bytes();data,changes=sanitize(original);oid=cmd(['hash-object','-w','--stdin'],data).decode().strip()
 cmd(['update-index','--add','--cacheinfo','100644',oid,rel]);assert cmd(['show',':'+rel])==data
 records[rel]=dict(path=rel,origin=str(f),kind='result',bytes=len(data),original_sha256=hashlib.sha256(original).hexdigest(),published_sha256=hashlib.sha256(data).hexdigest(),sanitizations=changes)
manifest['records']=[records[k] for k in sorted(records)];data=json.dumps(manifest,ensure_ascii=False,indent=2).encode();oid=cmd(['hash-object','-w','--stdin'],data).decode().strip();cmd(['update-index','--add','--cacheinfo','100644',oid,'source_manifest.json']);assert cmd(['show',':source_manifest.json'])==data
tree=cmd(['write-tree']).decode().strip();commit=cmd(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit-tree',tree,'-p',old,'-m','Preserve fixed TRAIN video-fivefold ridge predictions and audited negative increments']).decode().strip()
cmd(['update-ref','refs/heads/main',commit,old]);cmd(['push','origin','refs/heads/main:refs/heads/main']);assert cmd(['ls-remote','origin','refs/heads/main']).decode().split()[0]==commit
receipt=dict(status='D_BARE_RAW_TRAIN_REPORT_EXACT_BLOB_GITHUB_VERIFIED',actualclock_before_publication_UTC=a.stamp,commit=commit,parent_commit=old,C_checkout_not_modified=True,D_bare=str(bare),files=len(files)+1,raw_TRAIN_probe_complete=True,AB_original_scores_unchanged=True,full_neural_video_fivefold_complete=False)
for target in [r/'GitHub_publication_receipt.json',dc/'GitHub_publication_receipt.json']:target.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
print(json.dumps(receipt))
