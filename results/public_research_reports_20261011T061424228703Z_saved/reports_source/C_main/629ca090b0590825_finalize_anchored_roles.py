import hashlib,json,shutil
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
ws=Path(__file__).resolve().parent.parent
cp=ws/'outputs/原方案改进VAL_TEST五项实际接续.json';c=json.loads(cp.read_text(encoding='utf8'));d=Path(c['D_root'])
closure=d/'final_session_closure_20261008T052045Z.json';close=json.loads(closure.read_text(encoding='utf8'));assert close['all_explicit_exit_or_bye_natural0']
c['session_closure_pending']=False;c['all_four_connections_closed_natural0']=True;c['final_actualclock_UTC']=close['actualclock_UTC'];c['session_closure_reference']={'D_path':str(closure),'SHA':sha(closure)};c['D_remaining_bytes']=shutil.disk_usage(d).free
joint={'report_ZIP_SHA':sha(c['report_ZIP']),'closure_SHA':sha(closure),'captures':{s:sha(d/s/'complete_actual_capture.zip') for s in ('A_infer','B_audit','B_score')},'source_ZIP_SHA':sha(d/'frozen_VAL_TEST_eval_source.zip'),'whole_goal_complete':False}
c['final_joint_SHA']=hashlib.sha256(json.dumps(joint,sort_keys=True).encode()).hexdigest()
out=d/'final_joint_actual_20261008T052045Z.json';assert not out.exists();out.write_text(json.dumps(dict(actualclock_UTC=close['actualclock_UTC'],joint=joint,joint_SHA=c['final_joint_SHA']),indent=2),encoding='utf8')
cp.write_text(json.dumps(c,ensure_ascii=False,indent=2),encoding='utf8')
oldp=ws/'outputs/原AnchoredFlow消息增量优化实际接续.json';o=json.loads(oldp.read_text(encoding='utf8'));o['new_real_TEST_or_VAL_scores']=True;o['official_role_current_upgrade_result']=str(cp);o['official_role_training_overlap_disclosed']=True;o['official_role_report_SHA']=c['report_ZIP_SHA'];oldp.write_text(json.dumps(o,ensure_ascii=False,indent=2),encoding='utf8')
sp=ws/'outputs/研究接续状态.md';text=sp.read_text(encoding='utf-8-sig');prefix='最新实际 2026-10-08 05:20:45 UTC：当前原best36+固定20轮消息增量模型已按人类要求完成VAL229/TEST685全五项。先读《原方案改进VAL_TEST五项实际接续.json》及报告。VAL66.8122%/96.7593%/96.7654%/.354493326/.941719066；TEST67.5912%/95.2672%/95.2586%/.366326781/.925169325。同流关消息MAE分别.374509591/.389757819。官方角色含合并FIT训练重叠VAL135/TEST436，INNER34/88、OUTER60/161，非独立官方benchmark/不能冒五项超过CaReFlow。A原完整编码器914条child2399自然0，零标签全参数缓冲RNG不变/dummy0vs7/replay/直接前向0；D完整capture通过。B17912原源状态CPU完整流消息重放自然0/最大9.54e-7，不是CPUencoder；B18085一次VAL/TEST标签评分自然0，五项独立算术过。三真实原capture全部D源/argv/PID/完整SHA ZIPCRC唯一通过；冻结source与组合checkpoint仍完整引用保存。未重训/校准/选checkpoint/额外删大件。SSH96928/63581与SFTP33625/49113均明确exit/bye自然0，禁复用。最终joint '+c['final_joint_SHA']+'，D剩余'+str(c['D_remaining_bytes'])+'B；第三租期保守Oct8UTC14:00平台未核，C未fresh。整体优化/正式超过/完整五折/租期保存全目标未完成，原10min不改。以下历史。\n\n';sp.write_text(prefix+text,encoding='utf8')
print(json.dumps({'complete_current_VAL_TEST_request':True,'report':c['report'],'final_joint_SHA':c['final_joint_SHA'],'D_remaining_bytes':c['D_remaining_bytes'],'all_four_connections_closed_natural0':True},ensure_ascii=False))
