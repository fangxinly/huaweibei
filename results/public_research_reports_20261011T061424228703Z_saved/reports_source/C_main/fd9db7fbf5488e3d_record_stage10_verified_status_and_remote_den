"""Verified completed prefix and original CPU evidence; new remote denial preserved."""
import datetime,hashlib,json,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
BASE=Path('D:/CodexBackups/selective_flow_20261003_1105/staged_reference_first10_actual_20261006T151954Z')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
assert shutil.disk_usage('D:/').free>12*1024**3
dest=BASE/('verified_prefix_and_remote_query_denial_'+stamp);dest.mkdir(exist_ok=False)
for n in ('研究接续状态.md','研究建议交流接续.json','完整流单参考分段训练实际接续.md','完整流单参考分段训练实际接续.json'):
 p=dest/'preceding_state'/n;p.parent.mkdir(exist_ok=True);shutil.copy2(OUT/n,p)
local=read(BASE/'local_D_joint_audit_v2.json');cpu=read(BASE/'B_full_state_CPU_original_receipt.json');ex=read(BASE/'B_full_state_CPU_original_exit.json')
raw=BASE/'a/original_small_files/run';r=read(raw/'out/actual_training_receipt.json');gpu_exit=read(raw/'natural_exit.json')
assert gpu_exit['exit_code']==0 and gpu_exit['natural_wait_verified'] and gpu_exit['child_pid']==r['pid']
assert r['epochs']==10 and r['optimizer_steps']==220 and not r['formal100_complete']
assert ex['exit_code']==0 and ex['natural_wait_verified'] and ex['child_pid']==cpu['pid'] and ex['child_full_argv'][1:]==cpu['argv']
assert ex['original_receipt_sha256']==sha(BASE/'B_full_state_CPU_original_receipt.json')
assert cpu['original_training_receipt_sha256']==sha(raw/'out/actual_training_receipt.json')==local['original_training_receipt_sha256']
assert cpu['original_natural_exit_sha256']==sha(raw/'natural_exit.json')
assert cpu['source_sha256']==sha(BASE/'a/original_small_files/source/audit_staged_reference_CPU_v1.py')
for name in ('complete_resume_full.pt','selected_best_full.pt'):
 assert sha(BASE/'a'/name)==cpu['states'][name]['file_sha256']==local['full_files'][name]['sha256']
 assert (BASE/'a'/name).stat().st_size==local['full_files'][name]['bytes'] and cpu['states'][name]['tensors']==374
assert cpu['steps']==220 and cpu['complete_Adam_parameter_states']==364 and not cpu['GPU_used'] and not cpu['CPU_model_forward']
assert local['exact_next_update_proof']['continuous']==local['exact_next_update_proof']['restored'] and local['fresh_disk_replay_error']==0
assert not (BASE/'b/capture_receipt.json').exists() and not (BASE/'b/snapshot.zip').exists()
denial={'status':'NEW_ACTUAL_REMOTE_READONLY_QUERY_REJECTED_BEFORE_TRANSMISSION',
 'local_record_actual_utc':now.isoformat(),'pre_call_clock_utc':'2026-10-06 16:32:13 UTC',
 'tool':'write_stdin','session_id':79535,'intended_action':'Actual current UUID/compute/free-space, original stage10 receipt, continuation-root inventory read-only query.',
 'original_error':'write_stdin rejected: approval required by policy, but AskForApproval is set to Never',
 'remote_command_transmitted':False,'retry_or_alternate_remote_route_used':False,
 'current_GPU_UUID_compute_space_and_connection_exit': 'UNKNOWN_AFTER_REJECTED_QUERY',
 'prior_actual_A_query_utc':'2026-10-06T15:37:21.704492+00:00',
 'no_server_password_or_human_authorization_cause_claimed':True,
 'no_tool_repair_or_restart_claimed':True,'all_sessions_exit_not_newly_verified':True}
(dest/'original_remote_query_denial.json').write_text(json.dumps(denial,indent=2)+'\n',encoding='utf-8')
proof={'status':'ACTUAL_SHARED10_GPU_D_COMPLETE_FILES_ORIGINAL_B_CPU_PASSED_B_CAPTURE_DOWNLOAD_PENDING',
 'updated_actual_utc':now.isoformat(),'run_root':local['original_run'],'epochs':10,'formal_updates':220,
 'stage10_natural_exit_utc':gpu_exit['actual_exit_utc'],'original_training_receipt_sha256':sha(raw/'out/actual_training_receipt.json'),
 'original_A_capture_receipt_sha256':sha(BASE/'a/capture_receipt.json'),
 'original_B_CPU_receipt_sha256':sha(BASE/'B_full_state_CPU_original_receipt.json'),
 'original_B_CPU_exit_sha256':sha(BASE/'B_full_state_CPU_original_exit.json'),
 'original_B_CPU_exit_utc':ex['actual_exit_utc'],'complete_Adam_parameter_states':364,
 'whole_D_and_B_files':local['full_files'],'INNER_best_epoch':r['best_epoch'],
 'INNER_video_MSE_selection_only':r['INNER_video_equal_MSE_selection_only'],'final_budget':r['final_budget'],
 'next_update_check':r['next_update_continuation_check'],'donor_mechanism':r['donor_mechanism'],
 'fresh_strict_disk_replay_error':0.0,'stage10_CPU_and_D_complete':True,
 'original_B_capture_observed_COMPLETE_and_natural0_utc':'2026-10-06T15:40:36.719462+00:00',
 'original_B_capture_ZIP_and_receipt_downloaded_D':False,'final_all_capture_preservation_gate_passed':False,
 'continuation_started':False,'formal100_complete':False,'OUTER_or_head_labels_used':False,
 'current_remote_readonly_query_denied':denial,'CPU_model_forward':False,
 'scientific_claim':'One reference prefix, no matched control or donor superiority, no final generalization result.'}
(dest/'verified_prefix_status.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
old=read(OUT/'完整流单参考分段训练实际接续.json');old.update(proof)
old['prefix_status_original_D']=str(dest/'verified_prefix_status.json');old['epochs_completed_claimed']=True
(OUT/'完整流单参考分段训练实际接续.json').write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
doc=f'''完整流单参考：前10轮实际完成，续训当前未启动

实际更新UTC {now.isoformat()}。A child1173 UTC15:19:19.953311自然exit0，10/100轮、220次正式更新，每轮尾23有真实optimizer/scheduler。完整全链185402807参数、364参数张量，公共DeBERTa+随机任务初始化，FIT695/30-only监督与统计，INNER153/4用于earliest视频等权MSE选模。

当前best第10轮，INNER视频等权MSE2.185963203771532；第1轮同指标2.3439716052058257。它是训练期选模分数，不是OUTER/DEV最终效果，也不能与旧全TRAINfit参考.014或teacher OOF.629直接比较。没有新的头训练、匹配消融/共享10对照结果，100尚未完成。

工程验证已实际通过：阶段374.684秒；累计allocated3.655GiB/reserved4.104GiB，低于原6GiB，无中途peak重置。完整selectedbest741731206bytes与完整last10/Adam/scheduler/RNG/history/best续训状态2966997912bytes已真实D+B保存。D全SHA/ZIPCRC/唯一/新源/argv/完整100订单及220正式批通过；B独立TorchCPU child14108 UTC15:39:08.417034自然exit0，两整pt各374状态/185403174元素、364个Adam参数状态step220/scheduler220/RNG及153行数组与GPU原回执通过；不是CPU模型前向。

同一下一批的连续和fresh磁盘恢复各一孤立FIT更新，完整model/Adam/scheduler/起终RNG和目标精确摘要相同，原220状态还原，正式221未提交。自己的完整pt同实例和fresh公共实例INNER重放误差0，dummy标签0/7误差0，FITstats不变。

预声明同四FIT行供体置零的feedback.0680423/context变化.0142784/第二Euler状态.000123739/终端scalar变化9.10461e-6/恢复0，3前向2.465秒。原始NPZ已本地独立重建各变化，真实路径作用不证明改善预测。原两步终端0及全部负结果保留。

A capture25 UTC15:20:09实际COMPLETE/natural0后receipt，53成员ZIP与两大稳定inode引用均过；两大文件本轮另真实下载，区分引用与完整原件。B audit-root capture UTC15:40:36实际COMPLETE/natural0已工具观察，但其ZIP和receipt尚未下载D；最终全capture保存门控没有过，不能冒完整结束。

北京时间2026-10-07约00:32，实时只读GPU查询被write_stdin自动审批拒绝，原错误approval required by policy but AskForApproval Never，未传远端。没有重试或换命令/连接/工具绕过。当前GPU状态与会话退出无法新核实，之前A状态仅截至UTC15:37:21；此前自然exit0和B原CPU成功不受此新拒绝改变。11–100续训从未启动，不冒健康训练正在执行。未声称服务器/密码/人类未授权或工具修复，不强杀Codex。

下一仍按原plan/last10完整状态/2200日程继续11–100，但先恢复合法工具发送、下载B原capture并完成联合核验、fresh UUID/argv/空compute/资产源/remote和D12GiB/保守投影+2h保存余量。当前不换路推进。研究、正式100、开发评价与租期保存均未整体完成。

原件 {BASE.as_posix()}。本次verified prefix/新拒绝独立目录 {dest.as_posix()}。旧完成冻结源、权重、数组和原回执不改。
'''
(OUT/'完整流单参考分段训练实际接续.md').write_text(doc,encoding='utf-8')
ledger=read(OUT/'研究建议交流接续.json');ledger['second_lease_new_assets']['staged_reference_training']=old
ledger['scientific_state']='单fixed前10自然完成/220正式更新/D+B完整state与B原TorchCPU/精确nextupdate/fresh重放0过；Bcapture原ZIP尚未D，11-100未启动。UTC16:32新只读工具审批拒绝未传远端，不绕过，当前远端未知。无100/OUTER/供体收益。'
ledger['latest_remote_query_denial']=denial;ledger['updated_at_utc']=ledger['updated_utc']=now.isoformat()
(OUT/'研究建议交流接续.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
lines=(OUT/'研究接续状态.md').read_text(encoding='utf-8').splitlines();lines[0]=f'更新UTC {now.isoformat()}。继续主动研究/真实优化与第二租期保存，整体未完成。'
for i,s in enumerate(lines):
 if s.startswith('4) '):lines[i]='4) 单fixed前10 child1173 UTC15:19:19自然exit0，10/100及220updates/tail23正式通过；总374.7s/allocated3.655/reserved4.104GiB<6。INNER best10选择MSE2.185963非泛化。完整last10+Adam/scheduler/RNG/best和selectedbest真实D+B/B原TorchCPU child14108 UTC15:39:08自然0过；nextupdate完整model/Adam/scheduler/起终RNG精确一致，原220恢复/strict153 replay0。供体同四FIT terminal9.10e-6是作用非收益。A capture25全D过；B capture实COMPLETE/0但原ZIP与receipt未D，最终保存门控未过。11-100从未启动，无新OUTER/head/100成绩，先读实际接续.md/json。'
 if s.startswith('9) '):lines[i]='9) UTC16:32 SSH79535只读GPU实时查询write_stdin被approval required by policy but AskForApproval Never拒绝，未传远端；没有正式恢复证据禁重复试探/换命令工具连接绕过。先读latest_remote_query_denial及本轮D原错误。当前UUID/compute/空间/六session退出未知，A前10与B原CPU自然0只属于原child，不冒当前会话exit0。之前活动SSH79535/62639/17537和SFTP6854/47542/93212尚未实际exit/bye核验，禁假报关闭；健康任务不得停。密码仅actualprompt后人类新凭据禁文件/命令/自动/猜测/审视；不归因服务器/用户、不称工具修复。'
lines.append('12) 最新已验证官方DEV229仍旧A/B/C2 MAE.59844172/.59862399/.59932536、MSE.67633808/.67539716/.67760897，C2未胜A/B。旧内部EVAL863 F.014974953/native.014086934约降5.93%、CAL_OT.015010218失败，非新泛化。新完整流只前10的INNER2.185963选择指标，暂无新外折性能改善结论。')
(OUT/'研究接续状态.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
files=[Path(__file__),BASE/'local_D_joint_audit.json',BASE/'local_D_joint_audit_v2.json',BASE/'B_full_state_CPU_original_receipt.json',BASE/'B_full_state_CPU_original_exit.json',BASE/'local_audit_version_correction.json',BASE/'local_audit_exact_source_restoration.json']
files += [OUT/n for n in ('研究接续状态.md','研究建议交流接续.json','完整流单参考分段训练实际接续.md','完整流单参考分段训练实际接续.json')]
files += [ROOT/'work'/n for n in ('audit_staged10_D_local_v1.py','audit_staged10_D_local_v2.py','run_staged_CPU_original_wrapper_v1.py','capture_staged_CPU_audit_v1.py','audit_staged_reference100_CPU_v2.py','finalize_staged10_preservation_v1.py')]
members={}
for p in files:
 n=('outputs/' if p.parent==OUT else 'evidence/')+p.name;q=dest/n;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);members[n]={'sha256':sha(q),'bytes':q.stat().st_size}
for p in dest.rglob('*'):
 if p.is_file() and p.name not in ('member_manifest.json','records.zip','preservation_receipt.json'):members[p.relative_to(dest).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
(dest/'member_manifest.json').write_text(json.dumps(members,indent=2)+'\n')
with zipfile.ZipFile(dest/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in list(members)+['member_manifest.json']:z.write(dest/n,n)
with zipfile.ZipFile(dest/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,m in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==m['sha256']
receipt={'actual_utc':now.isoformat(),'directory':str(dest),'records_sha256':sha(dest/'records.zip'),'members':len(members),'SHA_CRC_unique_passed':True,'original_B_CPU_passed':True,'B_capture_saved_D':False,'continuation_started':False,'remote_query_rejected_before_transmission':True}
(dest/'preservation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
