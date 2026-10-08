"""Record successful original-source continuation and preserve local original receipts."""
import datetime,hashlib,json,shutil,zipfile,os,tomllib
from pathlib import Path

WORK=Path(__file__).parent
CWD=WORK.parent
OUT=CWD/'outputs'
D=Path('D:/CodexBackups/selective_flow_20261003_1105')
BASE=D/'staged_reference_continue100_actual_20261006T165611Z'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,a:Path(p).write_text(json.dumps(a,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
space={'actual_utc':now,'C_free_bytes':shutil.disk_usage('C:/').free,'D_free_bytes':shutil.disk_usage('D:/').free}
assert space['D_free_bytes']>4*1024**3
audit=read(BASE/'local_live_capture_joint_audit.json')
tools=read(WORK/'continue100_original_tool_outputs_20261006T1702Z.json')
for key in ('A_SSH_actual_exit','A_SFTP_actual_bye','B_SSH_exit_result','B_SFTP_exit_result'):
    item=tools[key]
    if item.get('status')=='fulfilled':item=item['value']
    assert item['exit_code']==0
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第十二次独立审视_旧DEV弱信号与公平参考锚点.md')
acceptance={'path':str(review),'sha256':sha(review),'read_and_considered':True,'actual_review_record_utc':now,
 'accepted':'旧DEV正平均收益只支持弱待核假设，不能排除常数偏差/选择/波动；未来完整fixed pF同锚、r=y-pF、新h加法，匹配U/W与同h读出；权重正则归一的恒等边界保留。',
 'reason':'有限h与真实条件均值不同，旧pooled与新视频等权不同；消除符号/参考坐标歧义，不改变当前冻结100源。',
 'deferred':'旧已选DEV视频bootstrap仅可描述组成敏感性，不能给选择后泛化覆盖；新64mask与无约束Q扩头未执行，不挡原100。',
 'new_experiment_or_score':False}
continuation={
 'status':'ACTUAL_ORIGINAL_LAST10_CONTINUE100_RUNNING_WITH_D_LIVE_CAPTURE_AUDITED',
 'record_actual_utc':now,'remote_run':audit['run_root'],'wrapper_pid':1485,'child_pid':1486,
 'actual_child_launch_utc':'2026-10-06T16:56:31.474825+00:00','actual_resumed_training_start_utc':audit['actual_resumed_training_start_utc'],
 'resumed_epoch':10,'resumed_optimizer_steps':220,'resumed_model_state_sha256':'45aeab53724cb03e4631ffba63fddd587f36065afe58ab9921a0325351a5064a',
 'actual_last_readonly_query_utc':'2026-10-06T17:02:26.095837+00:00','last_observed_complete_epoch':23,'last_observed_formal_updates':506,
 'last_query_no_traceback':True,'last_query_driver_host_pid':23348,'driver_and_container_PID_not_identical_claimed':True,
 'formal100_complete':False,'source_plan_and_order_unchanged':True,
 'preservation_gate_sha256':'6aaa3f96d53584b67a19090bb1b5aa07814433478f06b3f1b9c7c628d26a2bc6',
 'D_live_capture_audit':str(BASE/'local_live_capture_joint_audit.json'),'D_live_capture_audit_sha256':sha(BASE/'local_live_capture_joint_audit.json'),
 'captured_complete_epoch':13,'captured_updates':286,'live_capsule_is_whole_model_Adam_RNG_checkpoint':False,
 'whole_checkpoint_only_at_phase_end':True,'OUTER_or_head_evaluation_started':False,
 'latest_original_tools':str(WORK/'continue100_original_tool_outputs_20261006T1702Z.json'),
 'new_generalization_or_donor_benefit_result':False,
 'control_sessions':{'A_SSH75480':'actual_exit0','A_SFTP28718':'actual_bye_exit0','B_SSH62058':'actual_exit0','B_SFTP85817':'actual_bye_exit0','all_older_ids':'do_not_reuse; no invented historical exit'},
 'GPU_access':'A tiny CUDA original child exit0; actual ongoing full-model training, B current read-only connection verified; C not freshly rechecked; no tool-code repair/restart claim',
 'next':'Read new run out/progress and original natural_exit/receipt; once complete, whole pt D plus other-node CPU and new actual capture before new head work.'}
write(OUT/'完整流续训恢复实际核验.json',continuation)
report=f'''完整流原断点续训：实际恢复与D保存核验

记录UTC {now}。人类明确“你试一试啊”后，新A认证/UUID及原Torch CUDA点积11自然exit0。原16:32拒绝与旧session Unknown process记录保留；当前能力由新实测确认，无重启或底层修复结论。

B原62成员保存包已真正下载D，全SHA/CRC/唯一成员及完整原CPU回执、两整pt关联通过。前10总保存门UTC16:54:54.702059实际PASS，joint SHA6aaa3f96d53584b67a19090bb1b5aa07814433478f06b3f1b9c7c628d26a2bc6，原权重不重传。

原same-source continue100 wrapper1485/child1486 UTC16:56:31.474825启动，out/actual_training_start UTC16:57:04.151106实际恢复epoch10/step220/完整model-Adam-scheduler-RNG，模型SHA45aeab53724cb03e4631ffba63fddd587f36065afe58ab9921a0325351a5064a。不从best或预检两步恢复，固定source/seed/loss/全100订单/日程不改。

新根 {audit['run_root']}。UTC17:02:26.095837实际查询epoch23/506次正式更新、正确P4 UUID/compute、无Traceback、没有natural_exit；进程driver hostPID23348与container1486分别记录。该状态只截至实际查询，未称100完成或新泛化改善。

capture26 child1634 UTC16:58:26.240153实际CAPTURE_COMPLETE/natural0，receipt后41原成员ZIP真实D，全SHA/CRC/成员/source/完整argv/原220恢复/原10history及每个已捕获FIT batch订单过。capsule截至完整epoch13/286次，后续日志另含epoch14到step297；不是原子全状态或中间整pt保存。源只阶段结束产生新完整model/Adam/RNG。D {BASE}。

本地审核首版错误对字符串model_state_sha数组使用isfinite，发生TypeError；修正为仅数值数组有限性审计后自然exit0。原GPU训练和数组未修改，非GPU运行失败。

A SSH75480/SFTP28718与B62058/85817明确exit/bye actual0，detached训练持续；所有旧ID禁复用。完成100仍需实际自然exit0、完整best/resume真正D与异节点CPU、newroot/source/argv动态capture审核。OUTER/head/TEST标签未启，新头与收益对照未实施。

第12完整独立建议已全文审阅，SHA{sha(review)}。采纳统一完整fixed pF锚、残差加法符号及配对比较边界；旧DEV弱关联仍非确认。新真实GPU续训没有因该建议改变冻结训练。实际建议不是新成绩。
'''
(OUT/'完整流续训恢复实际核验.md').write_text(report,encoding='utf-8')
status=read(OUT/'完整流单参考分段训练实际接续.json')
status.update({'status':continuation['status'],'updated_actual_utc':now,'continuation_started':True,'formal100_complete':False,
 'original_B_capture_ZIP_and_receipt_downloaded_D':True,'final_all_capture_preservation_gate_passed':True,
 'run_root':audit['run_root'],'continuation':continuation,'epochs_completed':23,'formal_updates':506})
write(OUT/'完整流单参考分段训练实际接续.json',status)
md=OUT/'完整流单参考分段训练实际接续.md'
old=md.read_text(encoding='utf-8')
old=old.replace('续11–100待fresh UUID/空compute/fullargv/资产源/remote和D12GiB/原保守时间投影+2h保存核实；从完整last10恢复，不从best或预检两步恢复，不重复epoch10或重置日程。当前未启动续训，整体研究未完成。',
 '续11–100已在fresh实际门控和全部保存门通过后，从完整last10真实恢复；截至UTC17:02:26完成23轮506update。完整接续根与证据见完整流续训恢复实际核验.md/json；100未完成，整体研究未完成。')
md.write_text(old+'\n\n最新实际接续（'+now+'）：原B capture完整D核验与前10保存总门已过；原source continue100 wrapper1485/child1486运行，截至UTC17:02:26为23轮/506update，100未完成。新root '+audit['run_root']+'；D '+str(BASE)+'。\n',encoding='utf-8')
ledger=read(OUT/'研究建议交流接续.json')
ledger['updated_at_utc']=now
ledger['review_thread']['status']='TWELFTH_FULL_REPORT_READ_AND_INDEPENDENTLY_CONSIDERED'
ledger['latest_review']={'status':'TWELFTH_FULL_REPORT_READ_AND_INDEPENDENTLY_CONSIDERED','batch':'claude_suggestion_archived_dev_residual_signal_actual_20261006T164446Z','review':acceptance}
if not any(x.get('sha256')==acceptance['sha256'] for x in ledger['received_reviews']):ledger['received_reviews'].append(acceptance)
ledger['second_lease_new_assets']['staged_reference_training']=status
ledger['second_lease_new_assets']['current_ssh_sessions']={}
ledger['second_lease_new_assets']['current_sftp_sessions']={}
ledger['latest_actual_GPU_access_and_continuation']=continuation
ledger['scientific_state']='原10完整D/B/CPU保存总门通过后同源11-100真实恢复并运行；截至17:02:26 epoch23/506。当前无100完成/OUTER新泛化/供体收益结论。旧DEV残差仅弱未确认描述。'
for b in ledger.get('sent_batches',[]):
 if b.get('batch')=='claude_suggestion_archived_dev_residual_signal_actual_20261006T164446Z':b['status']='COMPLETE_REPORT_FULLY_READ_AND_CONSIDERED';b['review_report_sha256']=acceptance['sha256']
auto=read(WORK/'continue100_existing_automation_actual_update.json')
assert not auto['result'].get('isError',False)
config_path=Path(os.environ.get('CODEX_HOME',str(Path.home()/'.codex')))/'automations'/'automation'/'automation.toml'
config=tomllib.loads(config_path.read_text(encoding='utf-8'))
assert config['prompt']==auto['args']['prompt'] and config['rrule']==auto['args']['rrule'] and config['target_thread_id']==auto['args']['targetThreadId'] and config['status']=='ACTIVE'
ledger['automation'].update({'result':'EXISTING_AUTOMATION_OFFICIAL_UPDATE_SUCCESS_ACTIVE_VERIFIED_ON_DISK','updated_at_utc':now,'prompt_sha256':hashlib.sha256(auto['args']['prompt'].encode()).hexdigest(),'original_tool_receipt':str(WORK/'continue100_existing_automation_actual_update.json')})
write(OUT/'研究建议交流接续.json',ledger)
short=OUT/'研究接续状态.md'
lines=short.read_text(encoding='utf-8').splitlines()
lines[0]=f'更新UTC {now}。继续主动研究/真实优化与第二租期保存，整体未完成。'
for i,s in enumerate(lines):
 if s.startswith('4)'):lines[i]='4) 前10自然exit0/220更新/整D-B原TorchCPU/两capture/next-update与strict重放总保存门UTC16:54:54全部通过，joint6aaa3f96...。原source continue100已启动wrapper1485/child1486 UTC16:56:31，out/actual_training_start16:57:04真实last10/step220/完整Adam-scheduler-RNG恢复。新根'+audit['run_root']+'，截至UTC17:02:26实际epoch23/506update/无Traceback，100未完成；源/全订单/日程不改，不重启/重复10。先读完整流续训恢复实际核验.md/json。'
 if s.startswith('5)'):lines[i]=s.replace('仅准备，precheck2步不冒100','已按同源启动11-100，precheck2步不冒正式更新')
 if s.startswith('9)'):lines[i]='9) 人类明确重试后新A实际认证/UUID/CUDA点积11自然0及新完整训练运行，B实际认证/空compute/原包D下载核过；C未fresh查不冒三机当前。A75480/SFTP28718和B62058/SFTP85817明确exit/bye actual0，训练detached持续；所有旧ID禁复用，旧79535 Unknown process不冒exit0。原16:32审批拒绝保留，不当当前能力，非底层修复/重启。'
 if s.startswith('10)'):lines[i]='10) 建议聊天第12完整已全文读，接受弱未确认旧DEV关联/统一新完整F锚/新h加法/视频等权匹配U-W与同h读出/归一ridge边界；不改当前冻结100源。旧DEV组重采样最多描述敏感性，不冒选择后泛化CI。第11原总门规则仍适用；无待返回第12。健康或同证据不重复发。'
 if s.startswith('13)'):lines[i]=s.replace('10/100及远端拒绝/Bcapture待D状态不变。','该旧DEV结果不作改变冻结100的依据；最新远端/保存/续训以第4/9条为准。')
 if s.startswith('14)'):lines[i]='14) 第12批原发送一次保留，现完整报告已审阅；未发送新重复批。capture26 UTC16:58:26自然0后41原成员ZIP实D，全SHA/CRC/unique/新源/argv/恢复/原10history/捕获全部FIT订单审核过，capsule完整13轮286step，非中间整model-Adam-RNG。D staged_reference_continue100_actual_20261006T165611Z；结束仍须新整权重D+异节点CPU+动态capture，不以小包冒完成。'
short.write_text('\n'.join(lines)+'\n',encoding='utf-8')
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
records=BASE/('actual_access_resume_records_'+stamp)
records.mkdir(exist_ok=False)
files=[WORK/'GPU连接与CUDA原运算重试实核_20261006T165149Z.json',WORK/'continue100_original_tool_outputs_20261006T1702Z.json',WORK/'capture_staged_reference_live_v26.py',WORK/'audit_continue100_live_capture_local_v1.py',Path(__file__),WORK/'continue100_existing_automation_actual_update.json',BASE/'local_live_capture_joint_audit.json',review,
 OUT/'完整流续训恢复实际核验.md',OUT/'完整流续训恢复实际核验.json',OUT/'完整流单参考分段训练实际接续.md',OUT/'完整流单参考分段训练实际接续.json',OUT/'研究接续状态.md',OUT/'研究建议交流接续.json']
members={}
for p in files:
    sub='outputs' if p.parent==OUT else ('review' if p==review else 'evidence')
    name=sub+'/'+p.name
    dest=records/name;dest.parent.mkdir(exist_ok=True)
    shutil.copyfile(p,dest)
    assert sha(dest)==sha(p)
    members[name]={'original_path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
write(records/'original_member_manifest.json',{'actual_utc':now,'members':members,'fresh_local_space':space,'scope':'Live continuation engineering evidence; no final100 or new model checkpoint claim'})
zip_path=records/'records.zip'
with zipfile.ZipFile(zip_path,'x',zipfile.ZIP_DEFLATED) as z:
 for name in members:z.write(records/name,name)
 z.write(records/'original_member_manifest.json','original_member_manifest.json')
with zipfile.ZipFile(zip_path) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)+1
 for n,e in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==e['sha256']
receipt={'status':'ACTUAL_GPU_ACCESS_ORIGINAL220_RESUME_LIVE_CAPTURE_AND_REVIEW_RECORDS_D_SAVED',
 'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'records_directory':str(records),'members':len(members)+1,'zip_sha256':sha(zip_path),'whole_weights_new_download':False,'formal100_complete':False,'new_GPU_scientific_score_claimed':False,'new_other_node_CPU_audit':False}
write(records/'preservation_receipt.json',receipt)
print(json.dumps(receipt))
