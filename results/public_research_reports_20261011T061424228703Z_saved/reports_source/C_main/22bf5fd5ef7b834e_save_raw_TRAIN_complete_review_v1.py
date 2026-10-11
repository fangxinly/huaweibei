import datetime,hashlib,json,pathlib,shutil,sys,zipfile,os
P=pathlib.Path;ev=P('D:/CodexBackups/selective_flow_20261003_1105/candidate_final100_actual_20261009T012656Z');dc=ev.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs');src=P('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs')
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
state=read(dc/'D_current_research_state.json');batch=state['latest_raw_TRAIN_analysis_sent'];assert batch['batch_archive_SHA']=='6c3799526decdedbe2ff328573e34d916593d2bcaf167597ddf87edef21c802f'
r=ev/('raw_TRAIN_complete_review_actual_'+sys.argv[1]);r.mkdir()
sources={'固定raw_TRAIN视频五折_有限结论与估计器弱点_20261010.md':'c69abec1e88e8ead1ebfa2b6a78ae5538986ce1857b64ac855539f4cd5084541','固定raw_TRAIN五折_原报告分布审视_20261010.json':'c3818bd6bd5f4caba1ce4a9b11f8cbc69e9069d2e939e75ce10e4459e2e1d4a5'}
for n,h in sources.items():assert sha(src/n)==h;shutil.copyfile(src/n,r/n)
review=read(r/'固定raw_TRAIN五折_原报告分布审视_20261010.json');main=state['latest_raw_TRAIN_saved_error_concentration'];assert review['source_report_SHA256']==main['source_report_SHA']
assert review['primary_pooled_MSE']==state['latest_raw_TRAIN_probe_complete']['summary']['MSE']
for name,key in [('T_V','T_V_top_video'),('T_A_V','T_A_V_top_video'),('T_A','T_A_top_video')]:
 a=review['path_error_concentration'][name];b=main[key];assert a['video_id']==b['video_id'] and a['rows']==b['rows'] and a['share_of_path_total_squared_error']==b['fraction_of_path_total_SSE']
decision=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),report_SHA=sources['固定raw_TRAIN视频五折_有限结论与估计器弱点_20261010.md'],attachment_SHA=sources['固定raw_TRAIN五折_原报告分布审视_20261010.json'],full_report_and_attachment_read=True,independent_main_description_matches=True,accept=['All four original pooled increments remain negative for this prespecified finite estimator','Concentrated held-video error does not establish universal modality harm','Correctly solved ridge can extrapolate badly','Weak hash-bag text baseline and temporal summaries do not estimate strong-backbone residual complementarity','Video-fold overlap is not independent repetition','Upstream preprocessing/exposure scope remains unknown'],independently_checked_algebra='In exact arithmetic, each active standardized fit column has squared norm n. Thus trace(X^T X)=n*q; eigenvalues of X^T X+nI are at least n and at most n*(q+1), giving condition-number bound q+1<=502. This is an algebraic bound, not a measured condition number or proof of correct upstream inputs.',hypotheses_only=['Very small but nonzero fit scale can amplify held-video inputs','Missingness coding or raw channel units can change held-video support','Possible video distribution shift; no causal source identified'],reject_as_unjustified=['Delete the extreme video or change weights to rescue the original result','Treat 28/52 video improvements as statistical significance','Rerun original prediction/scoring/state audit or tune using seen TEST','Infer modality information absence, MI/PID, Bayes ceiling or A/B message mechanism'],next_scope='First inspect source-level raw channel units, missing-value coding and alignment/preprocessing fit provenance. Any new numeric diagnostic needs a separate explicit hypothesis, fixed source/row/role/once protocol and synthetic/fresh resource qualification before execution.',new_arrays_decoded=False,new_fit_or_scoring=False,new_remote_connection=False,new_diagnostic_dispatched=False,AB_primary_results_unchanged=True)
(r/'independent_decision.json').write_bytes(raw(decision));shutil.copyfile(__file__,r/P(__file__).name)
archive=r/'complete_independent_raw_TRAIN_review_original.zip';members=[]
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(r.iterdir()):
  if p==archive:continue
  b=p.read_bytes();z.writestr(p.name,b);members.append(dict(name=p.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
 z.writestr('member_manifest.json',raw(members))
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist())) and set(z.namelist())=={m['name'] for m in members}|{'member_manifest.json'}
 for m in members:assert len(z.read(m['name']))==m['bytes'] and hashlib.sha256(z.read(m['name'])).hexdigest()==m['sha256']
receipt=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(r),archive=str(archive),archive_SHA=sha(archive),all_member_SHA_CRC_unique_exact_set_passed=True,analysis_only=True)
(r/'D_preservation_receipt.json').write_bytes(raw(receipt));batch.update(full_reply_received=True,latest_observed_status='completed_full_report_read',completed_report_SHA=decision['report_SHA'],completed_attachment_SHA=decision['attachment_SHA'],no_same_batch_poll_or_resend=True)
state['latest_raw_TRAIN_analysis_completed']=dict(root=str(r),decision=decision,D_receipt=receipt,full_reply_received=True,independent_decision_saved=True);state['next_gate']=decision['next_scope'];state['updated_at_utc']=receipt['actual_UTC']
for dest in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
 tmp=dest.with_name(dest.name+'.rawreview.tmp');tmp.write_bytes(raw(state));os.replace(tmp,dest)
assert sha(dc/'D_current_research_state.json')==sha(c/'自主优化实际接续.json')
adv=read(c/'研究建议交流接续.json');adv['latest_raw_TRAIN_fixed_probe_analysis_sent']=batch;adv['latest_raw_TRAIN_fixed_probe_analysis_completed']=state['latest_raw_TRAIN_analysis_completed'];(c/'研究建议交流接续.json').write_bytes(raw(adv))
with (c/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n\n'+receipt['actual_UTC']+' 新raw建议批01a12239完整报告c69abec1…与附件c3818bd6…已全文阅读/实SHA核/独立决定保存，不再轮询或重发；误差集中数字与主任务独立原报告分解一致。接受只有限估计器負增量、外推可能而非已识别原因；标准化ridge精确算术条件数界q+1<=502仅代数不是新测，不能归因求解失稳。旧原结果/全部视频保留，不新解码/拟合/评分或连机。下一步先查raw单位/缺失编码/上游对齐处理源，任何新数值诊断另冻独立资格。完整D '+str(r)+' SHA '+receipt['archive_SHA']+' 全SHA/CRC/unique/exactset和D/C同步过。\n')
print(json.dumps(dict(receipt=receipt,completed_report_SHA=decision['report_SHA'],full_report_read=True,independent_decision_saved=True,no_new_probe=True)))
