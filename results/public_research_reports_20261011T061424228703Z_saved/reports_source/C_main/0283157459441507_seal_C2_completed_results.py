import json,hashlib,zipfile,shutil,sys
from pathlib import Path
clock=sys.argv[1];o=Path('outputs');r=json.loads(Path('work/C2_current_dispatch.json').read_text(encoding='utf8'));d=Path(r['D'])
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
old=o/'所有固定模型VAL_TEST五项实际对齐结果.json';assert sha(old)=='e111bfc051f539f53ab11380ec7d697d7e9ba9163f4bfa816f7bd2b6144e1b7b'
for n in ['所有固定模型VAL_TEST五项实际对齐结果.json','所有固定模型VAL_TEST五项实际对齐结果.md','所有模型VAL_TEST实际接续.json','研究接续状态.md']:shutil.copy2(o/n,d/('prior_'+n))
g=read(d/'A_original_capsule/original/actual_stage_receipt.json');c=read(d/'B_original_capsule/original/actual_stage_receipt.json')
assert g['scores']==c['scores'] and c['CPU_model_forward'] is False
cp=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_c2_completed_20261005/c/full_checkpoint.pt');assert sha(cp)==g['checkpoint_sha256']
report=read(old);report.update(status='ALL_REQUESTED_FIXED_MODELS_VAL_TEST_FIVE_COMPLETE_D_OTHER_NODE_CPU',actualclock_record_UTC=clock,C2_explicit_budget_change_question_pending=False)
report['models']['old_C2']=dict(VAL=g['scores']['val'],TEST=g['scores']['test'],scope='Original saved fullTRAINfit/DEVselected best37; no training/selection changes; original DEV replay exact0',checkpoint_sha256=g['checkpoint_sha256'],DEV_replay_max_error=g['original_DEV_replay_max_error'])
report['evidence']['C2']=dict(GPU_pid=g['pid'],other_node_CPU_pid=c['pid'],CPU_model_forward=False,A_capture_SHA=sha(d/'A_complete_capture.zip'),B_capture_SHA=sha(d/'B_complete_capture.zip'),full_checkpoint_D=str(cp),full_checkpoint_SHA=sha(cp),source_plan_SHA=r['plan_sha'],complete_source_and_fullargv_natural0=True,archive_CRC_unique_member_SHA_passed=True)
report['C2_latest_execution_complete']=True;report['C2_original_failure_preserved']=True;report['C2_failure_before_model_forward']=True
report['C2_authorization']=dict(verbatim='你只要测出来就行',actualclock_freeze=r['actualclock'],reserve_seconds=3000,immediate_D_and_other_CPU_completed=True,not_claiming_full_hour_reserve=True)
write(old,report)
md=['固定模型 VAL / TEST 五项实际结果','',f'记录：{clock}。全部指定固定模型已完成；VAL=官方DEV229，TEST=685。首三项为百分比。','', '| 模型 | 集合 | Acc7 % | Acc2 % | F1 % | MAE | Corr |','|---|---|---:|---:|---:|---:|---:|']
for n,label in [('careflow','CaReFlow best93'),('minimal_fixed_F','固定F best89'),('old_A','旧A best40'),('old_B','旧B best11'),('old_C2','旧C2 best37'),('anchored_selected20','新20轮（含训练/选模样本）')]:
 for role in ['VAL','TEST']:
  v=report['models'][n][role];md.append('| '+label+' | '+role+' | '+' | '.join([f'{100*v[k]:.4f}' for k in ['Acc7','Acc2','F1']]+[f'{v[k]:.8f}' for k in ['MAE','Corr']])+' |')
md+=['','C2本次A child5525自然0、B child33274自然0。原完整DEV预测重放误差0，换dummy标签和全部参数/RNG不变；完整source/预测/标签数组/进程argv/自然退出/capture已D保存，ZIP SHA/CRC/唯一/原件manifest逐一通过。D已有746207712字节完整原模型重新SHA核回。B复核保存数组的五项算术，未做CPU模型前向。','', 'C2 VAL MAE0.59932545→TEST0.64339851；旧A/B同样约0.598→0.643/0.646。原0.59是VAL开发集表现，未保留到TEST。C2 TEST Acc2/F1比固定CaReFlow高，Acc7/MAE/Corr差，没有五项全面超过。旧模型与新CaReFlow训练seed/容量/教师成本不匹配，不冒因果比较。','', '新20轮VAL/TEST是合并数据五折第0折模型的描述分数：VAL含FIT135/INNER34/OUTER60，TEST含FIT436/INNER88/OUTER161，不能称独立TEST改善、完整五折或论文SOTA。完整五折仍未完成，不使用TEST调整结构。201开发子集MAE0.591577864与此VAL229/TEST685不同。','', '原C2 child5494执行前预算门失败保留。人类回复“你只要测出来就行”后，本次单独冻结只C2评测和即时保存，采用3000秒保存预留；不声称当前能完整预留1小时，不修改原失败或旧源。实际补测12:32完成、B12:33完成。13:00三机真实动态保存仍待，保守13:30非平台确认。']
(o/'所有固定模型VAL_TEST五项实际对齐结果.md').write_text('\n'.join(md),encoding='utf8')
state=read(o/'所有模型VAL_TEST实际接续.json');state.update(status=report['status'],actualclock_record_UTC=clock,report_SHA=sha(old),C2_TEST_complete=True,C2_budget_exception_user_question_pending=False,C2_candidate_GPU_disabled=True,C2_completed_execution_plan=r['plan_sha'],C2_GPU_pid=g['pid'],C2_other_node_CPU_pid=c['pid'],C2_budget_human_reply='你只要测出来就行',C2_actual_reserve_seconds=3000,new_four_closed_explicit_exit_bye_zero_ids=[11048,46135,81189,52219],current_live_sessions=[]);write(o/'所有模型VAL_TEST实际接续.json',state)
st=o/'研究接续状态.md';st.write_text(f'最新实测{clock}：指定模型全部VAL/TEST五项完成，先读《所有固定模型VAL_TEST五项实际对齐结果.md/json》《所有模型VAL_TEST实际接续.json》。C2 A5525/B原保存数组CPU33274自然0，VAL MAE.599325451/TEST.643398511，TEST Acc7.448175182/Acc2.882442748/F1.882414104/Corr.847895881；D整原模型SHA+双真实capture/源/argv/ZIPCRC唯一全过。原预算失败保留，人类“你只要测出来就行”后单独冻结C2只测与即时保存3000s预留，不冒完整1h。旧0.59开发表现未保留TEST，无五项全面超过；新20混合角色分数非独立TEST，完整五折未完成。四会话11048/46135/81189/52219全exit/bye0关闭禁复用。13:00真实动态保存仍待，D fresh2449735680B/CFresh631812096B，比之前减少原因未核，不删旧原件。原10分钟保持，整体目标未完成。以下均历史。\n\n'+st.read_text(encoding='utf8'),encoding='utf8')
seal=d/'complete_result_seal';seal.mkdir()
for p in [old,o/'所有固定模型VAL_TEST五项实际对齐结果.md',o/'所有模型VAL_TEST实际接续.json',o/'研究接续状态.md',Path(__file__),Path('work/audit_C2_capture_local.py')]:shutil.copy2(p,seal/p.name)
write(seal/'manifest.json',dict(actualclock_UTC=clock,sha256={p.name:sha(p) for p in seal.iterdir() if p.is_file()},A_capture_SHA=sha(d/'A_complete_capture.zip'),B_capture_SHA=sha(d/'B_complete_capture.zip')))
with zipfile.ZipFile(d/'complete_results.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in seal.iterdir():z.write(p,p.name)
with zipfile.ZipFile(d/'complete_results.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
print(json.dumps(dict(status=report['status'],report_SHA=sha(old),seal_SHA=sha(d/'complete_results.zip'),C2_TEST=g['scores']['test'])))
