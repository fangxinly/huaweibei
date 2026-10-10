import datetime,json,os,pathlib,shutil,sys
P=pathlib.Path
sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw,sha,seal
r=P(sys.argv[1]);dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z'
c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
old=r.parent/'raw_TRAIN_input_source_scope_actual_20261009T200729Z'
paths=[old/'pinned_upstream/datasets/download_datasets.sh',r/'BERT_multimodal_transformer/source/datasets/download_datasets.sh',r/'ITHP/source/datasets/download_datasets.sh']
values=[p.read_bytes() for p in paths]
assert values[0]==values[1]==values[2]
scopes={n:json.loads((r/n/'selected_scope.json').read_bytes()) for n in ['BERT_multimodal_transformer','ITHP']}
evidence=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    exact_download_script_byte_equality=True,download_script_SHA=sha(paths[0]),
    current_official_pickle_SHA='5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b',
    source_commits=scopes,
    supported='The cited public repositories and pinned CaReFlow download scripts share identical bytes and resource locations.',
    not_proven=['Downloaded artifact byte identity: no dataset was fetched or compared','Construction recipe or per-segment centering for the current SHA-bound pickle','Channel semantics/units/sentinel/imputation/precision or upstream fit/pretraining exposure'],
    notebook_scope='Only code-cell source inspected; contains loading, interactive viewing, shape/alignment assertions, no construction or normalization recipe. Fetched notebook code never executed and outputs not inspected.',
    audit_scope='Selected pinned public files plus recursive filename inventory, not exhaustive content/history inspection.',
    new_arrays_or_labels_inspected=False,new_fit_score_diagnostic=False,new_SSH_connections=False,
    original_results_unchanged=True,
    decision='Shared download location supports a source association, not artifact identity or a preprocessing proof. The near-zero mean/centering hypothesis remains unconfirmed. Do not import any recipe/threshold from another dataset branch or retune the completed probe.')
(r/'source_association_decision.json').write_bytes(raw(evidence))
b=scopes['BERT_multimodal_transformer']['commit'];i=scopes['ITHP']['commit']
report='''# 引用仓库的 raw 输入来源关联

本次仅读取公开源码，固定了各自实际提交，并保存 HTTP 回执与文件 SHA；没有下载数据、运行外部源码、连接租用节点或重新诊断/训练/评分。

已保存的 CaReFlow 下载脚本，与两个致谢仓库的下载脚本逐字节相同。这建立了共同资源位置的来源关联，不能证明那些位置当前或历史返回的数据与本任务官方 pickle SHA 相同。

检查 MAG-BERT notebook 的代码单元，仅看到加载、交互查看和词对齐/形状断言，没有构造或归一化配方。未运行代码、未查看 notebook 输出。检查范围为所选源文件及文件名清单，不是全仓库全文或历史穷尽审计。

逐片段中心化仍未得到源码证明，通道语义/单位、有限缺失哨兵、精度、填充及上游拟合/预训练曝光也未闭合。原全部视频、折、lambda、预测和负结果保持；不能借此改尺度下限或按 TEST 救分。

固定主源：
'''+f'\n- [MAG-BERT 下载脚本](https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/{b}/datasets/download_datasets.sh)\n- [MAG-BERT notebook](https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/{b}/examine.ipynb)\n- [ITHP 下载脚本](https://github.com/joshuaxiao98/ITHP/blob/{i}/datasets/download_datasets.sh)\n'
(r/'引用仓库输入来源关联.md').write_text(report,encoding='utf-8')
out=r/'decision_closure';out.mkdir()
for n in ['source_association_decision.json','引用仓库输入来源关联.md','fetch_natural_receipt.json','D_preservation_receipt.json','fetch_ledger.json']:
    shutil.copyfile(r/n,out/n)
shutil.copyfile(__file__,out/P(__file__).name)
proof=seal(out,'complete_actual_source_association_decision_original.zip')
(r/'decision_D_preservation_receipt.json').write_bytes(raw(proof))
state=json.loads((dc/'D_current_research_state.json').read_bytes())
state['latest_raw_TRAIN_cited_public_source']=dict(root=str(r),finding=evidence,source_D_receipt=json.loads((r/'D_preservation_receipt.json').read_bytes()),decision_D_receipt=proof)
state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
state['next_gate']='Cited public repositories share the exact download script, but no SHA-bound construction/centering recipe is established. Continue provenance inspection only; do not rerun probe/support/scoring or borrow another branch recipe.'
for p in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
    tmp=p.with_name(p.name+'.citedsource.tmp');tmp.write_bytes(raw(state));os.replace(tmp,p)
assert (dc/'D_current_research_state.json').read_bytes()==(c/'自主优化实际接续.json').read_bytes()
with (c/'研究接续状态.md').open('a',encoding='utf-8') as f:
    f.write('\n'+state['updated_at_utc']+' 新公开来源核验：CaReFlow pinned/MAG-BERT/ITHP下载脚本同字节，只有共同资源关联，无数据下载或原pickle SHA等同证明。所选源码/notebook未给中心化构造配方，通道单位与曝光仍未知。新源完整ZIPaa31b202已D保存，独立决定ZIP'+proof['archive_SHA']+' 全SHA/CRC/unique/exactset过；无新数值/fit/score/诊断/SSH，原结果保持，D/C同步。\n')
print(json.dumps(dict(root=str(r),finding=evidence,decision_proof=proof),ensure_ascii=False))
