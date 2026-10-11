"""Preserve newly completed source review and unexecuted protocol advice."""
import ast, datetime, hashlib, json, os, pathlib, shutil, subprocess, sys, tomllib
P=pathlib.Path; sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw,sha,seal
EV=P('D:/CodexBackups/selective_flow_20261003_1105/candidate_final100_actual_20261009T012656Z')
DC=EV.parent/'candidate_posttrain_lowC_20261009T005229Z'
C=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
BASE=P('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs')
def now(): return datetime.datetime.now(datetime.timezone.utc)
def gates():
    assert shutil.disk_usage('C:/').free>200*1024**2
    assert shutil.disk_usage('D:/').free>40*1024**2
def sync(s):
    for p in [DC/'D_current_research_state.json',C/'自主优化实际接续.json']:
        tmp=p.with_name(p.name+'.recipe_review.tmp');tmp.write_bytes(raw(s));os.replace(tmp,p)
    assert (DC/'D_current_research_state.json').read_bytes()==(C/'自主优化实际接续.json').read_bytes()
def note(t):
    with (C/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+now().isoformat()+' '+t+'\n')
FILES=['归一化配方来源_独立审阅_20261010.txt','归一化配方来源_独立审阅_20261010.json','归一化配方来源_落盘回执_20261010.json','有界证据增量_收益归因与嵌套交叉拟合协议_20261010.md','有界证据增量_协议状态_20261010.json']
def save():
    gates();stamp=now();r=EV/('raw_TRAIN_recipe_complete_review_actual_'+stamp.strftime('%Y%m%dT%H%M%SZ'));r.mkdir()
    assert sha(BASE/FILES[0])=='ef16c132096109827fb2bf17e753f24b0a4db6e5eaa2d400ecf19bd98372611f'
    assert sha(BASE/FILES[1])=='bd52ab82da7215c141d686a64f460de829d4dc545dc66099d90206178d38c95c'
    for n in FILES:shutil.copyfile(BASE/n,r/n)
    shutil.copyfile(P(__file__).parent/'recipe_review_thread_snapshot_actual.json',r/'advisor_read_thread_actual_tool_response.json')
    shutil.copyfile(__file__,r/P(__file__).name)
    review=json.loads((r/FILES[1]).read_bytes());proposal=json.loads((r/FILES[4]).read_bytes())
    state=json.loads((DC/'D_current_research_state.json').read_bytes());source=P(state['latest_raw_TRAIN_recipe_scope']['root'])
    assert sha(source/'MISA_create_dataset.py')==review['MISA_source_SHA256']
    src=(source/'MISA_create_dataset.py').read_text();assert 'self.train = load_pickle' in src and 'visual.mean(0, keepdims=True)' in src
    pr=json.loads((source/'MAG_PR10/PR10_response.json').read_bytes());code=pr['body'].split('```python',1)[1].split('```',1)[0]
    tree=ast.parse(code);calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['mean','std']]
    assert len(calls)==2 and all(not n.keywords and len(n.args)==1 for n in calls)
    assert code.index('visual[np.isinf(visual)] = visual_max')<code.index('visual[np.isneginf(visual)] = visual_min')
    assert proposal['status']=='ANALYSIS_ONLY_PROPOSAL_NOT_EXECUTED_NOT_PREREGISTERED'
    assert not proposal['lambda_grid_frozen'] and not proposal['new_dataset_decode_fit_solve_score_support_GPU_SSH_or_contact']
    records={n:dict(SHA=sha(r/n),bytes=(r/n).stat().st_size) for n in FILES}
    decision=dict(actual_UTC=stamp.isoformat(),full_report_JSON_persistence_receipt_and_new_proposal_read=True,original_files=records,
      input_source_ZIP_SHA=review['new_scope_ZIP_SHA256'],source_SHA_and_AST_independently_checked=True,
      accepted_source=['MAG example whole-input reduction is not channel-wise temporal centering; actual call granularity unknown','MAG isinf assignment catches both signs before isneginf; example limitation only, not current MOSI cause','MISA cache can bypass generation; inner tuple difference requires a conversion chain rather than proving non-ancestry','Identity, generating recipe and upstream exposure are distinct closure states','Stop unanchored neighboring-repository expansion; reopen on direct current-SHA generator/converter evidence'],
      accepted_proposal_as_analysis_only=['Projection guarantee requires confirmed target support; clipping gains must be separated from modality gains','For exact ridge minimizer J(gamma)<=J(0)=Q implies norm<=sqrt(Q/lambda), hence abs(q-b)<=B*sqrt(5Q/lambda)','Complete nested video isolation must cover base models, calibration, selection and shadow construction; global OOF table slices are insufficient','Shadow training uses AV but inference uses text; beating a finite shadow does not identify conditional MI or PID','Zero candidate does not guarantee outer-fold benefit; overlap of folds is not independent replication'],
      deferred=['Separate fully fixed estimator question with legal input provenance, explicit candidate budget, exact source/synthetic qualification and fresh resource/once gates','No new code implementation, real arrays, fits, solves, scoring or remote diagnostic in this review'],
      unresolved=['Current SHA-bound generator/conversion lineage','Normalization axis, units, sentinels, precision and exposure of current inputs','Predictive gain, uncertainty and mechanism of proposed bounded estimator'],
      main_decision='Close this bounded public-source inquiry with unknown provenance. Preserve proposal as advice only, neither preregistered nor frozen. Keep all original rows, folds, lambda, predictions and negative results; no TEST-based rescue.',
      current_dataset_decode_fit_solve_score_SSH=0,all_completed_once_untouched=True,whole_research_complete=False)
    (r/'independent_decision.json').write_bytes(raw(decision))
    report='''# 来源审核收口与后续方案资格

新来源报告、JSON、落盘回执和有界证据方案已全文阅读，原报告SHA ef16c132、JSON bd52ab82逐字节一致。原JSON中只读未落盘字段是历史记录，另一个原落盘回执记录后来的真实保存，未回填旧字段。

独立核对原保存MISA源码SHA、缓存加载分支和MAG PR归约AST。MAG省略axis仅说明整个实际输入归约，调用粒度未知；isinf先替换两种无穷再isneginf的代码次序属于公开示例局限，不构成当前MOSI故障原因。MISA内层四元素格式与当前三元素不同，可能的转换链尚无证据，不据此断言绝不可能祖先来源。当前文件身份、生成配方、上游曝光分别记录；停止没有身份锚点的横向仓库扩张，出现直接SHA绑定才恢复追溯。

新方案的五维增量界由精确最小化目标与Cauchy-Schwarz得到，依赖lambda>0；投影不增加逐点损失需要目标支持已确认。裁切收益和模态增量分开、完整嵌套视频隔离覆盖校准/选模/影子组件、全局OOF切片不够，这些设计边界接受。影子训练使用音视频，推断仅文本；胜过有限影子不证明条件信息或PID。它仍是未预注册、未冻结、未执行的分析方案，未选择候选/网格/预算，也不证明收益或流的必要性。

本次只有来源静态审核、数学审阅和保存。没有本任务数组/标签解码、拟合、求解、评分或SSH；原全部视频、fold、lambda、预测和负结果保留，所有已消费once不重做，整体研究及完整神经视频五折未完成。
'''
    (r/'来源审核收口与方案资格.md').write_text(report,encoding='utf8')
    proof=seal(r,'complete_actual_recipe_independent_review_original.zip');proof['actual_UTC']=now().isoformat();(r/'D_preservation_receipt.json').write_bytes(raw(proof))
    sent=state['latest_raw_TRAIN_recipe_analysis_sent'];sent.update(full_reply_received=True,full_report_read=True,latest_observed_status='completed_full_report_and_followup_proposal_read',cursor='f8668b33-ccbc-454f-96ee-d12086ff737c:3',completed_source_turn_ID='01a12370-f4a9-7c33-b28e-f9f45502162a',later_human_optimization_turn_ID='01a1237c-9ea4-7133-a1e1-dc0fd561908d',report_SHA=records[FILES[0]]['SHA'],review_JSON_SHA=records[FILES[1]]['SHA'],no_same_batch_poll_or_resend=True,independent_decision_root=str(r))
    state['latest_raw_TRAIN_recipe_complete_review']=dict(root=str(r),decision=decision,D_receipt=proof);state['updated_at_utc']=now().isoformat()
    state['next_gate']='Public recipe inquiry closed with unknown current generator. Stop unanchored repository expansion. Later estimator work needs a separately fixed TRAIN-only scientific question, qualified complete nested supervised provenance, source/synthetic/resource/once gates; proposal is not frozen or executed.'
    sync(state);adv=json.loads((C/'研究建议交流接续.json').read_bytes());adv['latest_raw_TRAIN_recipe_analysis_sent']=sent;adv['latest_raw_TRAIN_recipe_complete_review']=state['latest_raw_TRAIN_recipe_complete_review'];(C/'研究建议交流接续.json').write_bytes(raw(adv))
    note('新来源建议ef16c132/JSONbd52ab82及后续有界证据方案已全文读取并独立决定，D完整ZIP'+proof['archive_SHA']+'全SHA/CRC/unique/exactset过。来源仍未知，停止无绑定邻近扩张；新方案仅分析未冻结未执行，不重做once。新批full_reply_received真，不再轮询重发。')
    print(json.dumps(dict(root=str(r),proof=proof)))
def publish(r):
    from prepare_github_source_upload import sanitize
    gates();pub=json.loads((r/'Release_receipt.json').read_bytes());proof=json.loads((r/'D_preservation_receipt.json').read_bytes())
    assert pub['remote_digest_verified'] and pub['source_SHA']==proof['archive_SHA'] and sha(P(pub['source']))==proof['archive_SHA']
    git='C:/Users/21234/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe';bare=DC/'publication.git'
    def cmd(args,data=None):return subprocess.check_output([git,'--git-dir='+str(bare)]+args,input=data,cwd=r)
    old=cmd(['rev-parse','HEAD']).decode().strip();latest=json.loads((DC/'GitHub_publication_receipt.json').read_bytes());assert old==latest['commit']
    assert old==cmd(['ls-remote','origin','refs/heads/main']).decode().split()[0]
    out=r/'public_source_and_summary';out.mkdir();names=FILES+['independent_decision.json','来源审核收口与方案资格.md','D_preservation_receipt.json','Release_receipt.json',P(__file__).name]
    for n in names:shutil.copyfile(r/n,out/n)
    cmd(['read-tree','HEAD']);manifest=json.loads(cmd(['show','HEAD:source_manifest.json']));records={v['path']:v for v in manifest['records']}
    for f in sorted(out.iterdir()):
        rel='results/TRAIN_recipe_independent_review_20261010/'+f.name;original=f.read_bytes();data,changes=sanitize(original)
        oid=cmd(['hash-object','-w','--stdin'],data).decode().strip();cmd(['update-index','--add','--cacheinfo','100644',oid,rel]);assert cmd(['show',':'+rel])==data
        records[rel]=dict(path=rel,origin=str(f),kind='saved_recipe_independent_review_unexecuted_proposal',bytes=len(data),original_sha256=hashlib.sha256(original).hexdigest(),published_sha256=hashlib.sha256(data).hexdigest(),sanitizations=changes)
    manifest['records']=[records[k] for k in sorted(records)];data=raw(manifest);oid=cmd(['hash-object','-w','--stdin'],data).decode().strip();cmd(['update-index','--add','--cacheinfo','100644',oid,'source_manifest.json']);assert cmd(['show',':source_manifest.json'])==data
    tree=cmd(['write-tree']).decode().strip();commit=cmd(['-c','user.name=Codex','-c','user.email=codex@users.noreply.github.com','commit-tree',tree,'-p',old,'-m','Preserve completed provenance review and unexecuted bounded estimator proposal']).decode().strip()
    cmd(['update-ref','refs/heads/main',commit,old]);cmd(['push','origin','refs/heads/main:refs/heads/main']);assert cmd(['ls-remote','origin','refs/heads/main']).decode().split()[0]==commit
    receipt=dict(status='D_BARE_RECIPE_COMPLETE_REVIEW_EXACT_BLOB_GITHUB_VERIFIED',actual_UTC=now().isoformat(),commit=commit,parent_commit=old,C_checkout_not_modified=True,all_published_blobs_verified=True,files=len(names)+1,no_new_numeric_execution=True)
    for p in [DC/'GitHub_publication_receipt.json',r/'GitHub_publication_receipt.json']:p.write_bytes(raw(receipt))
    state=json.loads((DC/'D_current_research_state.json').read_bytes());state['github_source']=receipt;state['latest_raw_TRAIN_recipe_complete_review'].update(Release_publication=pub,GitHub_publication=receipt);state['updated_at_utc']=now().isoformat();sync(state)
    closure=r/'publication_closure';closure.mkdir()
    for n in ['Release_receipt.json','GitHub_publication_receipt.json',P(__file__).name]:shutil.copyfile(r/n,closure/n)
    (closure/'state_sync_receipt.json').write_bytes(raw(dict(actual_UTC=now().isoformat(),D_C_same_SHA=sha(DC/'D_current_research_state.json'),actual_git_commit=commit)))
    closed=seal(closure,'complete_actual_recipe_review_publication_original.zip');(r/'publication_closure_D_receipt.json').write_bytes(raw(closed))
    note('新来源完整审阅/未执行方案D bare exactblob GitHub'+commit+'和Release digest真实过，公示闭合ZIP'+closed['archive_SHA']+'全SHA/CRC/unique/exactset过，D/C同字节，C旧HEAD不动。')
    print(json.dumps(dict(receipt=receipt,closure=closed)))
def close(r):
    gates();path=P('C:/Users/21234/.codex/automations/automation/automation.toml');cfg=tomllib.loads(path.read_text(encoding='utf8'))
    assert cfg['status']=='ACTIVE' and cfg['rrule']=='FREQ=MINUTELY;INTERVAL=10' and cfg['prompt'].startswith('最新Oct10实际UTC') and '来源完整审阅收口' in cfg['prompt'][:600]
    proof=dict(actual_UTC=now().isoformat(),status=cfg['status'],rrule=cfg['rrule'],prompt_SHA=hashlib.sha256(cfg['prompt'].encode()).hexdigest(),toml_SHA=sha(path))
    (r/'automation_actual_verification.json').write_bytes(raw(proof))
    (r/'execution_exit_observation.json').write_bytes(raw(dict(actual_UTC=now().isoformat(),save_tool_natural_exit=0,Release_publisher_tool_natural_exit=0,git_publisher_tool_natural_exit=0,no_remote_numeric_execution=True)))
    closure=r/'completion_closure';closure.mkdir()
    for n in ['automation_actual_verification.json','execution_exit_observation.json','publication_closure_D_receipt.json','GitHub_publication_receipt.json',P(__file__).name]:shutil.copyfile(r/n,closure/n)
    closed=seal(closure,'complete_actual_recipe_review_completion_original.zip');(r/'completion_D_receipt.json').write_bytes(raw(closed))
    state=json.loads((DC/'D_current_research_state.json').read_bytes());state['latest_raw_TRAIN_recipe_complete_review'].update(automation_actual_verification=proof,completion_D_receipt=closed);state['updated_at_utc']=now().isoformat();sync(state)
    (r/'final_state_sync_receipt.json').write_bytes(raw(dict(actual_UTC=now().isoformat(),D_C_same_SHA=sha(DC/'D_current_research_state.json'))))
    print(json.dumps(dict(completion=closed,automation=proof,D_C_same_SHA=sha(DC/'D_current_research_state.json'))))
if __name__=='__main__':
    stage=sys.argv[1]
    if stage=='save':save()
    elif stage=='publish':publish(P(sys.argv[2]))
    elif stage=='close':close(P(sys.argv[2]))
    else:raise ValueError(stage)
