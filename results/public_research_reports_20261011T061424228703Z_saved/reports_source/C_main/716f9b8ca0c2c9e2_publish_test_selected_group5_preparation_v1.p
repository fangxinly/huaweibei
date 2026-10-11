"""Publish exact new CV preparation artifacts; synchronize material task steering."""
import datetime as dt, hashlib, json, os, pathlib, shutil, subprocess, sys
P=pathlib.Path; BASE=P(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'work'))
from raw_TRAIN_capture_v1 import raw,sha,seal
from raw_TRAIN_save_and_publish_v1 import publish as publish_small
from prepare_github_source_upload import sanitize
DC=P('D:/CodexBackups/selective_flow_20261003_1105/candidate_posttrain_lowC_20261009T005229Z')
OUT=BASE/'outputs'
GIT='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'

def now():return dt.datetime.now(dt.timezone.utc).isoformat()
def sync(state):
    b=raw(state)
    for p in [DC/'D_current_research_state.json',OUT/'自主优化实际接续.json']:
        tmp=p.with_name(p.name+'.group5.tmp');tmp.write_bytes(b);os.replace(tmp,p)
    assert (DC/'D_current_research_state.json').read_bytes()==(OUT/'自主优化实际接续.json').read_bytes()

def main():
    assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
    root=P(json.loads((BASE/'work/test_selected_group5_pointer.json').read_bytes())['root'])
    plan=json.loads((root/'group5_design.json').read_bytes())
    assert plan['methods']==['anchored_message20','old_fixed_A','factorized_aux','regression_aux','careflow']
    assert not plan['execution_enabled'] and plan['storage']['human_authorization']=='把内容都上传github'
    ledger=json.loads((root/'candidate_ledger.json').read_bytes())
    public=root/'pub';public.mkdir()
    manifest_rows=[]
    for p in sorted(root.rglob('*')):
        if not p.is_file() or public in p.parents or root/'public_source_and_summary' in p.parents or p.suffix=='.zip':continue
        parts=list(p.relative_to(root).parts)
        if parts[0].startswith('contract_qualification_actual_'):
            parts[0]='qualification'
        rel=P(*parts).as_posix();b=p.read_bytes();changed={}
        if p.suffix in ['.py','.json','.md','.txt','.log']:
            b,changed=sanitize(b)
        dst=public/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b)
        manifest_rows.append(dict(path=rel,origin_SHA=sha(p),published_SHA=hashlib.sha256(b).hexdigest(),sanitizations=changed))
    # Original preparation has no SSH secrets. If any redaction is needed, keep
    # private originals on D and publish only this explicitly distinct capture.
    (public/'public_original_identity_manifest.json').write_bytes(raw(dict(actual_UTC=now(),files=manifest_rows)))
    proof=seal(public,'complete_actual_group5_public_preparation.zip')
    (root/'public_capture_D_receipt.json').write_bytes(raw(proof))
    publish_small(P(proof['archive']),proof['archive_SHA'],'group5-test-selected-preparation-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
    release=json.loads((root/'Release_receipt.json').read_bytes());assert release['remote_digest_verified']
    bare=DC/'publication.git'
    def git(args,data=None):return subprocess.check_output([GIT,'--git-dir='+str(bare)]+args,input=data,cwd=root)
    parent=git(['rev-parse','HEAD']).decode().strip()
    latest=json.loads((DC/'GitHub_publication_receipt.json').read_bytes());assert parent==latest['commit']
    assert git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==parent
    git(['read-tree','HEAD'])
    inventory=json.loads(git(['show','HEAD:source_manifest.json']))
    records={v['path']:v for v in inventory['records']}
    prefix='results/historical_TEST_selected_group5_20261010'
    published=[]
    for p in sorted(public.rglob('*')):
        if not p.is_file() or p.suffix=='.zip':continue
        rel=prefix+'/'+p.relative_to(public).as_posix();b=p.read_bytes()
        oid=git(['hash-object','-w','--stdin'],b).decode().strip()
        git(['update-index','--add','--cacheinfo','100644',oid,rel]);assert git(['show',':'+rel])==b
        records[rel]=dict(path=rel,origin=str(p),kind='TEST_selected_pooled_group5_preparation_not_training',
                          bytes=len(b),original_sha256=sha(p),published_sha256=sha(p),sanitizations={})
        published.append(rel)
    for name in ['Release_receipt.json','public_capture_D_receipt.json']:
        p=root/name;b,changes=sanitize(p.read_bytes());rel=prefix+'/'+name
        oid=git(['hash-object','-w','--stdin'],b).decode().strip()
        git(['update-index','--add','--cacheinfo','100644',oid,rel]);assert git(['show',':'+rel])==b
        records[rel]=dict(path=rel,origin=str(p),kind='new_group5_actual_publication_receipt',bytes=len(b),
                          original_sha256=sha(p),published_sha256=hashlib.sha256(b).hexdigest(),sanitizations=changes)
        published.append(rel)
    inventory['records']=[records[k] for k in sorted(records)]
    b=raw(inventory);oid=git(['hash-object','-w','--stdin'],b).decode().strip()
    git(['update-index','--add','--cacheinfo','100644',oid,'source_manifest.json'])
    tree=git(['write-tree']).decode().strip()
    commit=git(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit-tree',tree,'-p',parent,
                '-m','Preserve human TEST-selected 93-video fivefold roster and GitHub storage design']).decode().strip()
    git(['update-ref','refs/heads/main',commit,parent])
    git(['push','origin','refs/heads/main:refs/heads/main'])
    assert git(['ls-remote','origin','refs/heads/main']).decode().split()[0]==commit
    for rel in published:
        assert hashlib.sha256(git(['show',commit+':'+rel])).hexdigest()==records[rel]['published_sha256']
    receipt=dict(status='D_BARE_NEW_GROUP5_PREPARATION_EXACT_BLOB_GITHUB_VERIFIED',actual_UTC=now(),
                 commit=commit,parent_commit=parent,C_checkout_not_modified=True,
                 all_published_blobs_verified=True,files=len(published)+1,no_training_dispatched=True)
    for p in [DC/'GitHub_publication_receipt.json',root/'GitHub_publication_receipt.json']:p.write_bytes(raw(receipt))
    state=json.loads((DC/'D_current_research_state.json').read_bytes())
    state['latest_human_TEST_selected_group5']=dict(root=str(root),status=plan['status'],design_SHA=sha(root/'group5_design.json'),
        methods=plan['methods'],pooled_rows=2195,pooled_videos=93,final_method_fold_outputs=25,actual_stage_queue=35,
        human_scope=['test表现最好的几次','全数据约 93 视频五折','把内容都上传github'],
        inherited_no_TEST_candidate_selection_constraint_superseded_only_for_new_exploratory_comparison=True,
        all_original_results_and_once_preserved=True,execution_enabled=False,training_started=False,
        GitHub_permanent_large_capture_storage_authorized=True,requires_all_fold_D_full_state_reservation=False,
        staging_CPU_source_runtime_lease_once_gates_unchanged=True,
        D_original_receipt=json.loads((root/'D_preservation_receipt.json').read_bytes()),
        public_release=release,github_publication=receipt)
    state['github_source']=receipt;state['updated_at_utc']=now()
    state['next_gate']='Prioritize the human-authorized TEST-selected pooled2195/93video fivefold comparison: 4 ours plus CaReFlow, fresh fold parents/teachers, source/native/synthetic/serial GitHub chunk restore qualification, measured batch5 lease time and2h saving reserve, separately frozen new stage once. Candidate/ID design saved; no training yet. Original TRAIN-only probes and official evaluations are completed historical evidence, never rerun.'
    sync(state)
    shutil.copyfile(root/'五折名单与执行设计.md',OUT/'历史TEST优势方案与CaReFlow五折执行设计.md')
    with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:
        f.write('\n'+now()+' 人类指定历史TEST最好方案/全2195条93视频五折/全部GitHub保存。候选消息20、旧固定A、新factorizedA/regressionB及CaReFlow，25最终外折结果/35训练阶段；候选与无标签视频划分核验、D准备ZIP及GitHub源码与Release保存实际过。消息旧TEST含FIT/INNER重叠，探索性选择披露。没有新训练或SSH，完整执行源码/教师/原生预检/串行Release还原/租期与once门待，不因D同时存不下全部状态而停止独立准备。commit'+commit+'。\n')
    closure=root/'publication_closure';closure.mkdir()
    for name in ['Release_receipt.json','GitHub_publication_receipt.json','public_capture_D_receipt.json']:
        shutil.copyfile(root/name,closure/name)
    shutil.copyfile(P(__file__),closure/P(__file__).name)
    (closure/'state_sync_receipt.json').write_bytes(raw(dict(actual_UTC=now(),D_C_same_SHA=sha(DC/'D_current_research_state.json'),git_commit=commit)))
    closed=seal(closure,'complete_actual_group5_publication_closure.zip')
    (root/'publication_closure_D_receipt.json').write_bytes(raw(closed))
    print(json.dumps(dict(root=str(root),GitHub=receipt,Release=release,closure=closed)))

if __name__=='__main__':main()
