"""Update only mutable short continuation and communication ledger after real audits."""
from pathlib import Path
import argparse,datetime,hashlib,json
p=argparse.ArgumentParser();p.add_argument('--late',type=Path,required=True);p.add_argument('--final',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];o=root/'outputs'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
late=read(a.late/'joint_capture_audit.json');final=read(a.final/'joint_capture_audit.json')
assert late['status']==final['status']=='THREE_ACTUAL_LATE_LEASE_CAPTURES_SHA_CRC_MEMBERS_AND_PRIOR_EVIDENCE_VERIFIED'
assert read(a.final/'session_closure.json')['all_actual_exit0']
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第五次独立审视_CAL覆盖完成与隔离参考预检.md')
ledger=read(o/'研究建议交流接续.json');ledger['updated_at_utc']=now
ledger['human_reciprocal_authorization']='你和额外的聊天之间是相互交流沟通，你自己也要清醒明智，有些思路要试了才知道，但是也不要一根筋，也要适当查找合适的文献'
ledger['workflow']['send_trigger']='重大实际实验D/CPU保存后，或实质文献/数学/决策批明确非新实验；健康查询或同一证据不重复'
ledger['workflow']['reciprocal_tools_authorized']=True
ledger['review_thread'].update(status='fifth_completed_sixth_failed_usage_limit_not_completed',last_wait_cursor='5c299d87-8d70-412a-94b9-bdc45953b3ac:20',last_completed_turn='01a11020-a3b5-71d0-a685-dbb1e679cdfc',current_turn='01a1102b-c971-7010-89e4-6674534b2d4e')
if not any(v.get('path')==str(review).replace('\\','/') for v in ledger['received_reviews']):
 ledger['received_reviews'].append({'path':str(review).replace('\\','/'),'sha256':sha(review),'completed_utc':'2026-10-06T07:42:55Z','read_and_considered':True,'accepted':'最小fixed参考删除预测不用utility及10参考支路；pair预先删除；一pass两Euler、FIT尾批23、初始/预检分离、全程预算与严格重放。仅草案未部署。'})
memo=o/'文献证据与双向研究决策_20261006T0742Z.md'
batch='literature_reciprocal_decision_20261006T0744Z'
if not any(v.get('batch')==batch for v in ledger['evidence_batches']):
 ledger['evidence_batches'].append({'batch':batch,'status':'sent_once_sixth_turn_failed_usage_limit','report':str(memo),'report_sha256':sha(memo),'sent_utc':'2026-10-06T07:44:27Z','prompt_preserved':str(o/'文献交流与自动化原回执_20261006T0744Z.json'),'repeat_send_allowed':False,'scientific_result':False})
ledger['scientific_state']='八臂负结果和CAL覆盖完成保留；第五已审阅；第六用量中断无完整建议；最小流参考v2仅草案。当前只租期末真实保存，无新GPU实验/预检/100。'
ledger['lease_preservation']={'latest_directory':str(a.final),'joint_audit_sha256':sha(a.final/'joint_capture_audit.json'),'missing_windows':['UTC08:08','UTC10:08'],'actual_final_capture_utc':[v['inventory_utc'] for v in final['rows'].values()],'estimated_expiry_utc':'2026-10-06T12:08:17Z','platform_verified':False,'new_training_allowed_with_two_hour_save_reserve':False}
ledger['restart_preference']={'human_requested_conditionally':True,'executed':False,'current_tools_working':True,'record':str(o/'权限异常恢复偏好与官方依据.md')}
(o/'研究建议交流接续.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
captures='；'.join(n.upper()+v['inventory_utc'] for n,v in final['rows'].items())
state=f'''更新UTC {now}。只读此短接续与按需原证据；研究及租期保存未整体完成。
1) 本轮三fresh授权P4 UUID/空compute/原完整argv与完成记录通过，无新训练。UTC11:33三capture19自然exit0，D {a.late} 全SHA/CRC/member/旧科学与大引用不变通过。
2) UTC11:38窗口真实三capture：{captures}。D {a.final} joint_capture_audit.json通过；所有六会话明确exit/bye实际exit0，ID见session_closure.json，禁止复用。大weight只fresh稳定inode SHA引用，不是本轮重下载；之前完整D/异节点CPU原证据仍保留。
3) UTC08:08和10:08强化保存没有执行，历史缺口不回填。期限只估UTC12:08:17/BJ20:08:17，平台未核；至少2h保存余量已不满足，禁止新预检/100以填卡。下一轮临近期满实际保存/状态核验优先，不能伪称租期结束或整体完成。D优先每次fresh查盘禁删。
4) 固定EVAL863/34八臂负结果保留：F MSE.014974953、native.014086934、CAL_OT.015010218、CAL_Opm.015021068；不扩张本版/新100。CAL418零标签原生半径覆盖96→418仅机制，不重做同EVAL救分。九旧流100/full/20、三91818教师100/full、OOF/Oracle/同折采集D+CPU不重跑重传。原C仅10失败/旧C最终/旧14:30缺口不回填。
5) 第五独立审视已完成/全文审阅；采纳最小fixed流v2草案删除预测不用utility与10参考支路、预先删除pair；保留MSE+.02FM+.01cycle+.05unimodal+.01variance、一pass两Euler。新源/GPU预检/100未实现。公共DeBERTa+随机任务、fold0 FIT695-only/INNER153零标签重放/OUTER433禁标签、尾23/初始化分离/全保留梯度/完整weight真实D+异节点CPU/预算仍必须实核。
6) 已检索原始CaReFlow/残差提升/learning-to-defer/Gradient-Blending/CRC/LTT/非单调稳定性论文并完成文献memo；文献非实验结果。公平流参考与便宜plain剩余误差可学习性分开研究，先合法输入/标签角色/成本与停止标准冻结；Q=delta²-2(y-pF)delta与直接残差基线是候选，未实际执行。经验cap/epsilon非真风险界；视频相关不能把418行当418独立样本。
7) 同一建议聊天01a10fcb-6663-70a2-9a76-60e5634d0c03，人类已明确双向交流授权，分析者不GPU/改主源/训练/子代理。第六turn真实failed usage limit，只有commentary，无完整第六建议；不得冒返回。交流JSON按批次去重，不为健康查询重发。第五与文献/v2决策原件本轮永久D研究记录保存。
8) 权限异常时人类建议正常重启已记录，当前SSH/SFTP发送正常，没有重启/底层修复。官方故障排查仅支持持续卡住时等活跃聊天结束后正常重启，未证明重启解除历史审批拒绝。若再拒绝，保留错误不换命令/连接/工具绕过；不能强制杀Codex掩盖恢复失败。
9) 只新A REDACTED_SERVER_HOST.invalid:53449 GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7；B REDACTED_SERVER_HOST.invalid:53416 GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067；C REDACTED_SERVER_HOST.invalid:53458 GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa。旧址/N/R永禁连。真实password提示后仅本聊天人类原凭据，禁密码文件/命令/自动提示/猜测，退出核实际结果。
10) capture19 SHA6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad，公共root与Jacobian1327 deployment不改；新子根覆盖，实际CAPTURE_COMPLETE/exit0后receipt再ZIP审核。仅官方TRAIN/dev，禁TEST挑结构/SOTA/稳定5seed/语义真值/整体crossfit；原10分钟频率，未变/仅本地准备静默。禁subagents/设置/浏览器UI/续租/关机/停健康训练/改完成冻结源。
'''
(o/'研究接续状态.md').write_text(state,encoding='utf-8')
report=f'''租期末强化保存与下一阶段研究决策

实际更新UTC {now}。本轮没有新GPU实验、校准或成绩；租期仅估UTC12:08:17，平台未核。

实际三节点捕获和D独立审核：{a.late}、{a.final}。准确时间看joint_capture_audit.json。receipt后完整ZIP下载、SHA/CRC/唯一成员核验与原CPU/科学资料联结均通过。此前完整weights永久D与异节点CPU保留；本轮大SHA引用不是新weight下载或新的CPU模型前向。所有会话已明确退出。

UTC08:08与10:08捕获未执行，缺口如实保留。UTC11:38窗口实际保存另有当前时刻证据，不能回填早期时点。当前剩余租期不足正式协议要求的2h保存余量，因此本租期不启动新完整流预检/100或学生100。

第五审视已读并落实到最小目标v2草案，未实施科学源。第六独立文献审视因usage limit失败；未来恢复时从既有memo和明确问题续接，不当作已返回意见。主聊天自己的文献分析有效但不冒独立验证。

后续两条问题分开：一是公共预训练/新任务/FIT-only完整流参考的执行可行性；二是同折plain参考上的低成本剩余误差是否可预测。后者需先冻结合法输入、角色、低容量结构和停止标准，再真正执行；不能在已读EVAL上继续挑分。比较消息控制和直接输出残差基线，如果消息没有额外价值允许简化。尚无这些新方法的分数。

原F/native/CAL_OT/CAL_Opm固定EVAL MSE仍为.014974953/.014086934/.015010218/.015021068。教师OOF MSE.6293898611/MAE.5965080822只是教师质量，不能混表。全TRAINfit/DEVselected旧参考仍不能冒全流程crossfit或全新确认集。
'''
(o/'租期末强化保存与下一阶段研究决策.md').write_text(report,encoding='utf-8')
print(json.dumps({'updated_utc':now,'captures':captures,'new_training_started':False}))
