import ast,datetime,json,os,pathlib,shutil,sys
P=pathlib.Path;sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw,sha,seal
r=P(sys.argv[1]);dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
pr=json.loads((r/'MAG_PR10/PR10_response.json').read_bytes());comments=json.loads((r/'MAG_issue11_comments_response.json').read_bytes());scope=json.loads((r/'source_scope.json').read_bytes())
assert pr['user']['login']=='RE-N-Y' and pr['merged'] and '## MOSEI pre-processing' in pr['body']
assert any(x['user']['login']=='RE-N-Y' and '#10' in x['body'] and 'MOSEI' in x['body'] for x in comments)
snippet=pr['body'].split('```python',1)[1].split('```',1)[0];tree=ast.parse(snippet)
reductions=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['mean','std']]
assert len(reductions)==2 and all(len(n.args)==1 and not n.keywords for n in reductions)
source=(r/'MISA_create_dataset.py').read_text();assert 'visual.mean(0, keepdims=True)' in source and 'np.std(visual, axis=0, keepdims=True)' in source
assert 'acoustic.mean(0, keepdims=True)' in source and 'EPS = 1e-6' in source
out=r/'complete_source_and_decision';out.mkdir()
for n in ['MAG_issue11_response.json','MAG_issue11_comments_response.json','MISA_commit_response.json','MISA_create_dataset.py','MISA_README.md','source_scope.json','fetch_ledger.json','fetch_raw_TRAIN_recipe_sources_v1.py','D_preservation_receipt.json']:
    shutil.copyfile(r/n,out/n)
shutil.copytree(r/'MAG_PR10',out/'MAG_PR10')
shutil.copyfile(P(__file__).parent/'fetch_MAG_recipe_PR10_v1.py',out/'fetch_MAG_recipe_PR10_v1.py');shutil.copyfile(__file__,out/P(__file__).name)
decision=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    MAG_issue11_maintainer_points_to_PR10=True,MAG_PR10_dataset_scope='MOSEI, not proof for current MOSI',
    MAG_PR10_example_reduction_axis='No axis argument: whole-array reduction as written; does not imply each channel has zero temporal mean',
    MAG_PR10_head=pr['head']['sha'],MAG_PR10_merge=pr['merge_commit_sha'],
    MISA_commit=scope['MISA_commit'],MISA_source_SHA=sha(r/'MISA_create_dataset.py'),
    MISA_MOSI_recipe='SDK word alignment/averaging; remove sp; per-instance per-channel centering/scaling with EPS1e-6; separate split pickles and four-element modality tuple',
    MISA_current_pickle_binding=False,source_AST_axis_check_passed=True,external_code_executed=False,
    current_dataset_array_or_label_decodes=0,new_fit_solve_score_diagnostic=False,new_SSH_connection=False,
    public_PR_patch_metadata_inspected=True,notebook_dataset_outputs_executed=False,
    uncertainty=['Actual current MOSI pickle generator, exact source version and artifact SHA binding','Channel units/sentinels/precision/imputation and upstream exposure scope','Whether current near-zero channel means came from per-channel centering; hypothesis only'],
    independent_decision='Two primary sources specify different normalization axes and datasets. Do not transfer PR10 MOSEI operations or MISA EPS/row filtering to this MOSI pickle. Shared URLs, formats or similar moments are not construction proof. Preserve all original videos/folds/lambda/predictions and results.',
    original_results_unchanged=True,whole_neural_fivefold_complete=False)
(r/'recipe_scope_decision.json').write_bytes(raw(decision));(out/'recipe_scope_decision.json').write_bytes(raw(decision))
report='''# 归一化配方的主源证据与适用范围

MAG-BERT 协作者在 issue11 指向 PR10，限定为当时发布的 MOSEI。合并 PR 的说明包含清洗/归一化示例；mean/std 没有 axis 参数，按写法归约整个数组，不能推出每个通道的时间均值为零。AST 静态核对通过，未执行该源码。

MISA 的 MOSI 生成源码明确按实例、按通道沿时间轴中心化和缩放，EPS为1e-6；它先用 SDK 词对齐平均并删除 sp，保存分开的 split pickle 和含 actual_words 的四元素模态 tuple。它是对照配方，不是本次官方三元素 tuple/pickle 的生成身份证据，不能移植它的 epsilon 或过滤步骤。

因此，中心化是已有方法的真实操作，但本次输入是否如此生成仍未知。两来源的不同归一化轴尤需区分；共同下载链接、格式和类似数值现象都不替代源版本和文件 SHA 的绑定。通道语义/单位、哨兵、精度、填充与曝光范围未闭合。

只抓取公开源码/API回执，PR补丁包含公开 notebook 元数据；没有执行 notebook 或任何下载的源码，没有读取本任务数据数组/标签，没有新训练/诊断/求解/评分或 SSH。全部原视频、折、lambda、预测和结果保持。

主源：
- [MAG issue11 协作者说明](https://github.com/WasifurRahman/BERT_multimodal_transformer/issues/11#issuecomment-737673762)
- [MAG PR10](https://github.com/WasifurRahman/BERT_multimodal_transformer/pull/10)
'''+f"- [固定 MISA MOSI 生成代码](https://github.com/declare-lab/MISA/blob/{scope['MISA_commit']}/src/create_dataset.py#L197)\n"
(r/'归一化配方适用范围.md').write_text(report,encoding='utf-8');(out/'归一化配方适用范围.md').write_text(report,encoding='utf-8')
proof=seal(out,'complete_actual_recipe_scope_original.zip');(r/'recipe_scope_D_receipt.json').write_bytes(raw(proof))
state=json.loads((dc/'D_current_research_state.json').read_bytes());state['latest_raw_TRAIN_recipe_scope']=dict(root=str(r),decision=decision,D_receipt=proof)
state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();state['next_gate']='Need current MOSI SHA-bound generator/version evidence. MAG PR10 is MOSEI whole-array normalization; MISA is separate per-channel pipeline. Do not infer current centering or transfer EPS/filtering, and do not rerun completed numeric protocols.'
for p in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
    tmp=p.with_name(p.name+'.recipescope.tmp');tmp.write_bytes(raw(state));os.replace(tmp,p)
assert (dc/'D_current_research_state.json').read_bytes()==(c/'自主优化实际接续.json').read_bytes()
with (c/'研究接续状态.md').open('a',encoding='utf-8') as f:f.write('\n'+state['updated_at_utc']+' 新主源配方范围核验：MAG issue11协作者指向PR10的MOSEI示例mean/std无axis，全数组归约；MISA MOSI按实例通道时间轴中心化EPS1e-6，独立生成/保存流程不同，均未绑定本次官方pickle SHA。原来源未知保留，不移植EPS/过滤/结构救分。完整新ZIP'+proof['archive_SHA']+'全SHA/CRC/unique/exactset过；只静态源/API/AST，无本任务数组/标签/新fit-score-diagnostic/SSH，D/C同步。\n')
print(json.dumps(dict(root=str(r),proof=proof)))
