"""Preserve local pinned-source inspection; no numeric data decoding or fitting."""
import datetime,hashlib,json,pathlib,shutil,sys,os
sys.path.insert(0,str(pathlib.Path(__file__).parent))
from raw_TRAIN_capture_v1 import seal,raw,sha
P=pathlib.Path
base=P('C:/Users/21234/Documents/Codex/2026-10-01/1-zhang-s-yang-y-chen')
ev=P('D:/CodexBackups/selective_flow_20261003_1105/candidate_final100_actual_20261009T012656Z')
dc=ev.parent/'candidate_posttrain_lowC_20261009T005229Z'
c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
auditbase=base/'outputs/own_flow_v6_followup/careflow_upstream_audit_20261004T2120Z'
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
now=datetime.datetime.now(datetime.timezone.utc)
r=ev/('raw_TRAIN_input_source_scope_actual_'+now.strftime('%Y%m%dT%H%M%SZ'));r.mkdir()
audit=json.loads((auditbase/'upstream_source_audit.json').read_bytes())
manifest=json.loads((base/'outputs/careflow_reproduction/official_manifest.json').read_bytes())
assert audit['commit']==manifest['commit']=='5c9f9c7a0bb3f1202ebb3258052da2b99710c565'
selected=['README.md','global_configs.py','train_reflow_new.py','datasets/download_datasets.sh']
rows={x['path']:x for x in audit['rows']};bindings=[]
for n in selected:
 p=auditbase/'upstream'/n;assert sha(p)==rows[n]['upstream_sha256']
 dest=r/'pinned_upstream'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
 bindings.append(dict(path=n,source_path=str(p),actual_SHA=sha(p),historically_recorded_upstream_SHA=rows[n]['upstream_sha256'],SHA_match=True))
for name,p in [('upstream_source_audit.json',auditbase/'upstream_source_audit.json'),('official_manifest.json',base/'outputs/careflow_reproduction/official_manifest.json'),('excluded_MMSA_preprocess_mosi.py',base/'work/preprocess_mosi.py')]:shutil.copyfile(p,r/name)
assert 'd3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9' in (r/'excluded_MMSA_preprocess_mosi.py').read_text()
findings=dict(actual_UTC=now.isoformat(),scope='Local exact pinned-source inspection, not a fresh remote-source fetch or exhaustive repository audit',commit=audit['commit'],source_bindings=bindings,official_raw_data_SHA='5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b',supported=['README describes word-aligned words/visual/acoustic tuples with labels and segment IDs','global_configs declares MOSI visual47/acoustic74 dimensions','Download script fetches prebuilt pickle files and does not contain a feature extraction or normalization recipe','Author token conversion repeats modality rows for subword tokens, truncates and pads downstream of pickle construction'],not_established_by_inspected_sources=['Per-channel names and physical units','Raw missing-value sentinel rules','Exact extraction/alignment/normalization recipe for the SHA-bound official pickle','Upstream fitting, pretraining or evaluation-exposure scope'],excluded_source=dict(path=str(base/'work/preprocess_mosi.py'),script_SHA=sha(r/'excluded_MMSA_preprocess_mosi.py'),expected_input_SHA='d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9',reason='Different MMSA aligned_50.pkl branch, not provenance for current official raw TRAIN inputs'),interpretation=['Dimensions alone do not identify COVAREP/FACET/OpenFace channels','Downstream author token padding or minmax normalization cannot be attributed to the raw ridge input pipeline','Source gaps do not establish leakage, missingness artifacts or the cause of extreme held-video error'],new_arrays_decoded=False,new_fit_score_or_probe=False,remote_connections=False,primary_results_unchanged=True,next_gate='Freeze a separate TRAIN-only saved-fit support diagnostic before any numeric execution: inspect finite/nonfinite rates and held-fold standardized support without changing folds, features, lambda, videos or original predictions. First qualify exact inputs, synthetic checks, resources and once role; source-unit uncertainty remains explicit.')
(r/'source_scope_findings.json').write_bytes(raw(findings))
report='''# raw TRAIN 输入来源核验

核验仅使用本地已保存的固定提交源字节，SHA 与历史上游回执逐项一致；没有重新下载数据、解码数组或训练计分。

现有证据支持词对齐的 words/visual/acoustic tuple、视觉47维和音频74维。下载脚本取得预制 pickle，没有 raw 特征提取或标准化配方。作者的 subword 复制、截断、填充属于 pickle 之后的处理，不能反向证明 raw 数据如何生成。

检查的源文件未建立每通道名称/单位、raw 缺失哨兵、提取/对齐/归一化细节或上游拟合与预训练曝光范围。维数不足以认定具体特征工具。另一份本地 MMSA aligned_50.pkl 预处理脚本要求不同输入 SHA，不能作为本次官方 raw 输入的来源依据。

这段核验没有确认极端误差的原因。原固定五折全部视频和负增量保持。下一数值工作必须另冻 TRAIN-only 的已保存拟合支持诊断，检验非有限率与 held-fold 标准化范围；不改原折、特征、lambda、视频或预测，不重复原训练/计分。
'''
(r/'输入来源核验.md').write_text(report,encoding='utf8');shutil.copyfile(__file__,r/P(__file__).name)
receipt=seal(r,'complete_local_source_scope_original.zip');receipt.update(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(r),local_source_inspection_only=True)
(r/'D_preservation_receipt.json').write_bytes(raw(receipt))
state=json.loads((dc/'D_current_research_state.json').read_bytes());state['latest_raw_TRAIN_input_source_scope']=dict(root=str(r),findings=findings,D_receipt=receipt);state['next_gate']=findings['next_gate'];state['updated_at_utc']=receipt['actual_UTC']
for dest in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
 tmp=dest.with_name(dest.name+'.sourcescope.tmp');tmp.write_bytes(raw(state));os.replace(tmp,dest)
assert sha(dc/'D_current_research_state.json')==sha(c/'自主优化实际接续.json')
with (c/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n\n'+receipt['actual_UTC']+' raw输入本地固定源SHA核验已保存：词对齐/47与74维有来源；下载预制pickle，没有原通道单位/缺失哨兵/提取归一化配方或曝光范围。另MMSA aligned50脚本不同输入SHA已排除，不能误归当前输入。无新数值解码/拟合/评分/连机，原结果不变。完整D '+str(r)+' SHA '+receipt['archive_SHA']+' 全SHA/CRC/unique/exactset过；D/C同步。\n')
print(json.dumps(receipt))
