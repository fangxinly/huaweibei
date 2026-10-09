import datetime,hashlib,json,pathlib,shutil,sys,os,zipfile
P=pathlib.Path
sys.path.insert(0,str(P(__file__).parent));from raw_TRAIN_save_and_publish_v1 import checkzip
def raw(v):return (json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def write(p,v):p.write_bytes(raw(v))
r=P(sys.argv[2]);ev=r.parent;dc=ev.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
if sys.argv[1]=='--verify':
 receipt=read(r/'remote_original/closed_capture_receipt.json');path=r/'remote_original/complete_actual_raw_TRAIN_closed.zip';assert sha(path)==receipt['archive_SHA']
 members=checkzip(path,'capture_member_manifest.json');dest=r/'closed_verified_original';assert not dest.exists()
 with zipfile.ZipFile(path) as z:z.extractall(dest)
 assert read(dest/'natural_exit.json')['natural_exit']==0 and read(dest/'audit_report_natural_exit.json')['natural_exit']==0
 audit=read(dest/'analysis/saved_fit_CPU_audit.json');report=read(dest/'analysis/TRAIN_finite_prediction_report.json');assert audit['status']=='REAL_TRAIN_SAVED_FIT_CPU_AUDIT_PASSED' and len(audit['checks'])==20 and not audit['VAL_TEST_numeric_decode']
 inner=dest/'analysis/complete_actual_TRAIN_audit_report.zip';proof=read(dest/'analysis/capture_receipt.json');assert sha(inner)==proof['sha256'];checkzip(inner,'member_manifest.json')
 assert report['rows']==1281 and report['videos']==52 and len(report['per_video'])==52 and report['saved_fit_CPU_audit_passed']
 assert read(dest/'prediction_Release_receipt.json')['remote_digest_verified']
 summary=dict(actual_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),root=str(r),remote_root='/data/coding/N2_raw_TRAIN_probe_actual_20261009T194523Z',SSH_session=10986,old_unknown_session_31293_never_reuse=True,prediction_natural_exit=read(dest/'natural_exit.json'),analysis_natural_exit=read(dest/'audit_report_natural_exit.json'),archive=str(path),archive_SHA=sha(path),archive_bytes=path.stat().st_size,all_member_SHA_CRC_unique_exact_set_passed=True,content_members=len(members),prediction_SHA=report['prediction_SHA'],CPU_checks=20,max_normal_equation_relative_residual=max(x['relative_normal_equation_residual'] for x in audit['checks']),max_prediction_abs_error=max(x['prediction_max_abs_error'] for x in audit['checks']),MSE=report['pooled_row_weighted_MSE'],pairs=report['paired_comparisons'],real_probe_executed=True,official_AB_reexecuted=False,not_full_neural_video_fivefold=True,upstream_preprocessing_and_pretraining_scope_unknown=True,old_approval_rejection_retained_this_protocol_now_executed=True,bottom_level_approval_policy_repair_not_claimed=True)
 write(r/'raw_TRAIN_complete_D_receipt.json',summary)
 lines=['# 原始TRAIN固定视频五折预测增量探针实际结果','',f"真实预测与审核正常退出；1281行、52视频、五折，四路固定ridge，lambda=1。20份fit折统计/系数正规方程/原预测均通过CPU审核。预测先D及Release完整保存，再计算误差。",'', '| 输入 | 行加权MSE（越低越好） |','|---|---:|']
 for k,v in summary['MSE'].items():lines.append(f'| {k} | {v:.9f} |')
 lines+=['','四项预设MSE降低量（基线减增广）均为负：']
 for item in summary['pairs']:lines.append(f"- {item['baseline']} → {item['augmented']}：{item['pooled_row_MSE_reduction']:.9f}")
 lines+=['','这套固定有限估计器中，加入音频或视觉后的外视频预测误差更大；没有正向增量证据。不能据此认定模态无信息、MI/PID、机制、因果或稳定显著收益。官方raw上游处理和预训练曝光范围未知。','', '本探针仅TRAIN诊断，不是完整神经网络视频五折。A/B正式作者评分、推理和训练均未重复；TEST不用于选择结构、参数或救分。','',f"完整原件SHA：{summary['archive_SHA']}",f"原预测SHA：{summary['prediction_SHA']}",'']
 (r/'TRAIN固定视频五折实际结果.md').write_text('\n'.join(lines),encoding='utf8');print(json.dumps(summary))
elif sys.argv[1]=='--sync':
 summary=read(r/'raw_TRAIN_complete_D_receipt.json');pub=read(r/'closed_Release_receipt.json');assert pub['remote_digest_verified'] and pub['source_SHA']==summary['archive_SHA']
 state=read(dc/'D_current_research_state.json');state['latest_raw_TRAIN_probe_complete']=dict(summary=summary,D_receipt=str(r/'raw_TRAIN_complete_D_receipt.json'),Release_receipt=pub,report=str(r/'TRAIN固定视频五折实际结果.md'))
 state['latest_raw_TRAIN_probe_protocol_prepared'].update(real_probe_executed=True,remote_execution_qualified=True)
 state['latest_remote_policy_rejection']['historical_after_human_retry']=True
 state['current_research_execution_blocker']=None;state['next_gate']='Preserve and independently analyze the completed fixed TRAIN probe. No positive finite-estimator increment; do not tune using TEST, repeat AB scoring, or treat this as completed neural video-fivefold research.'
 state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for dest in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
  tmp=dest.with_name(dest.name+'.rawactual.tmp');tmp.write_bytes(raw(state));os.replace(tmp,dest)
 assert (dc/'D_current_research_state.json').read_bytes()==(c/'自主优化实际接续.json').read_bytes()
 write(r/'D_C_state_sync_receipt.json',dict(actual_UTC=state['updated_at_utc'],same_SHA=sha(dc/'D_current_research_state.json')))
 with (c/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n\n'+state['updated_at_utc']+' 人类重试后原N2旧31293 Unknown已禁复用，新同SSH协议10986真实认证/UUID/runtime/资产/原源码核过，真实raw TRAIN固定视频五折预测1465自然0UTC19:46:34，20fit状态CPU审核和一次报告自然0UTC19:49:49。四路MSE T2.157974826/T+A3.332043809/T+V10.407500873/T+A+V14.945894100，四预设增量均负，只有限估计器负结果，不冒MI/PID/机制/模态无信息。完整原件D '+str(r)+' SHA '+summary['archive_SHA']+' 与Release远端digest过；D/C同步。旧审批失败保留，此次协议可执行但不冒底层策略修复；A/B正式评分/推理/训练未重做，整体神经五折未完成。\n')
 print(json.dumps(dict(state_synced=True,archive_SHA=summary['archive_SHA'])))
else:raise ValueError('Unknown mode')
