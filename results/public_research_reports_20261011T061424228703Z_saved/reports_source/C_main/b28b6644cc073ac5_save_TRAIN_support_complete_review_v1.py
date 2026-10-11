import datetime,json,os,pathlib,shutil,sys
P=pathlib.Path;sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw,sha,seal
base=P('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review')
ev=P('D:/CodexBackups/selective_flow_20261003_1105/candidate_final100_actual_20261009T012656Z');dc=ev.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
now=datetime.datetime.now(datetime.timezone.utc);r=ev/('TRAIN_support_complete_review_actual_'+now.strftime('%Y%m%dT%H%M%SZ'));r.mkdir()
files={'TRAIN支持范围诊断_独立数学与科学边界_20261010.md':'13b04da22b96be3ee5d282186fc385044dc5a57202d9127111eb40aea3bf1ee1','TRAIN支持诊断独立审阅_20261010.json':'77eb4ce95aa92b84534f5c3c8360e27e892d1118afa8670b4dc14ffb1b93004b'}
for n,h in files.items():assert sha(base/'outputs'/n)==h;shutil.copyfile(base/'outputs'/n,r/n)
shutil.copyfile(base/'work/audit_saved_TRAIN_support_batch_20261010.py',r/'audit_saved_TRAIN_support_batch_20261010.py')
review=json.loads((r/'TRAIN支持诊断独立审阅_20261010.json').read_bytes());state=json.loads((dc/'D_current_research_state.json').read_bytes());original=P(state['latest_TRAIN_support_diagnostic_complete']['root'])/'verified_original/diagnostic'
assert review['archive_SHA256']==state['latest_TRAIN_support_diagnostic_complete']['D_receipt']['archive_SHA']
assert review['raw_or_NPZ_array_decodes_by_this_review']==0 and not review['new_fit_forward_support_diagnostic_or_score_by_this_review']
channels=json.loads((original/'per_channel_saved_fit_support.json').read_bytes());counts=json.loads((original/'TRAIN_raw_nonfinite_and_zero_counts.json').read_bytes())
for v in review['selected_saved_channels_all_folds']:
 a=next(x for x in channels if x['fold']==v['fold'] and x['path']==v['path'])['channels'][v['index']]
 assert all(a[k]==value for k,value in v.items() if k not in ['fold','path'])
for v in review['selected_saved_zero_count_records']:assert v==next(x for x in counts if x['row_id']==v['row_id'])
for v in review['saved_missingness_features_inactive']:
 a=next(x for x in channels if x['fold']==v['fold'] and x['path']==v['path'])['channels'][v['index']];assert not a['active'] and a['coefficient']==0
decision=dict(actual_UTC=now.isoformat(),full_report_and_JSON_read=True,report_SHA=files['TRAIN支持范围诊断_独立数学与科学边界_20261010.md'],review_JSON_SHA=files['TRAIN支持诊断独立审阅_20261010.json'],input_original_SHA=review['archive_SHA256'],selected_saved_records_independently_matched=True,adopted=['Fixed representation/fit states demonstrably amplify held values; same error-concentration video identity remains descriptive','Positive unit rescaling leaves Z invariant if active status unchanged; an absolute scale number alone is not evidence of bad channels','Scale sensitivity across original fit subsets is described, not a learning curve or proof of general distribution shift','All missingness-ratio columns inactive/zero coefficient; actual nonfinite count0 excludes this pipeline NaN/Inf replacement as the direct route for these specific terms','Labels were not indexed or numerically decoded; original pickle bytes and old supervised coefficients are not label-free or unsupervised','The two maximal-term rows have2 word-aligned timepoints; this is exploratory motivation, not deletion or causal evidence'],unresolved=['Raw channel semantics/units/precision/upstream normalization and exposure scope','Whether per-segment centering creates theoretically near-zero means; hypothetical only','Net error attribution, independent replication, task information and new method benefit'],decision='Keep all original rows/videos/folds/lambda/predictions and negative results. Prioritize provenance evidence for any later separately fixed question. No new real diagnostic, learning, threshold/lambda modification or TEST-based rescue authorized by this review.',no_new_array_decode_fit_score=True,overall_research_complete=False)
(r/'independent_decision.json').write_bytes(raw(decision));shutil.copyfile(__file__,r/P(__file__).name)
receipt=seal(r,'complete_actual_support_independent_review_original.zip');(r/'D_preservation_receipt.json').write_bytes(raw(receipt))
sent=state['latest_TRAIN_support_analysis_sent'];sent.update(full_reply_received=True,latest_observed_status='completed',full_report_original=str(r/'TRAIN支持范围诊断_独立数学与科学边界_20261010.md'),report_SHA=decision['report_SHA'],review_JSON_SHA=decision['review_JSON_SHA'],no_same_batch_poll_or_resend=True)
state['latest_TRAIN_support_complete_review']=dict(root=str(r),decision=decision,D_receipt=receipt);state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();state['next_gate']='Source/provenance evidence for whether near-zero summary means reflect a defined upstream operation; treat as exploratory unresolved question. Do not rerun completed support/probe/scoring or choose fixes using TEST.'
for p in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
 tmp=p.with_name(p.name+'.supportreview.tmp');tmp.write_bytes(raw(state));os.replace(tmp,p)
assert (dc/'D_current_research_state.json').read_bytes()==(c/'自主优化实际接续.json').read_bytes()
adv=json.loads((c/'研究建议交流接续.json').read_bytes());adv['latest_TRAIN_support_analysis_sent']=sent;(c/'研究建议交流接续.json').write_bytes(raw(adv))
with (c/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n\n'+state['updated_at_utc']+' 新支持批完整建议报告13b04da2/JSON77eb4ce9已全文读取、匹配保存通道/行/缺失比例记录并独立决定，完整D ZIP'+receipt['archive_SHA']+' 全SHA/CRC/unique/exactset过。接受单位换算Z不变、fit极端值进入与否影响经验尺度、nonfinite0/缺失比例不活跃、两最大项行2时点；逐片段中心化/近常量残差仅未证实来源假说，先补来源。不执行新数组/fit/score/阈值，不按TEST救分，全部原结果保持。建议批full_reply_received真，不再轮询重发。\n')
print(json.dumps({'root':str(r),'receipt':receipt,'report_SHA':decision['report_SHA'],'D_C_same_SHA':sha(dc/'D_current_research_state.json')}))
