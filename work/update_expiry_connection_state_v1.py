import datetime as dt, hashlib, json, shutil, zipfile
from pathlib import Path

b=Path.cwd();D=Path('D:/CodexBackups/selective_flow_20261003_1105/lease_expiry_readonly_20261006T120417Z');now=dt.datetime.now(dt.timezone.utc).isoformat();sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert shutil.disk_usage('D:/').free>1073741824
preceding=D/'preceding_state';preceding.mkdir(exist_ok=False)
for name in ['研究接续状态.md','研究建议交流接续.json','租期末强化保存与下一阶段研究决策.md']:shutil.copy2(b/'outputs'/name,preceding/name)
audit=json.loads((D/'local_preservation_and_connection_audit.json').read_text(encoding='utf-8'));assert audit['status']=='EXISTING_THREE_D_COMPLETE_SNAPSHOTS_FRESH_LOCAL_SHA_CRC_UNIQUE_MEMBERS_PASSED'
report=f'''租期估计时点连接变化与永久D核验

实际更新UTC {now}。本轮没有新科学实验或远端capture；估计UTC12:08:17已经接近/经过，但平台尚未核验，不能据此说三实例已释放、重建或租期保存整体完成。

A：fresh原授权地址SSH实际exit1，OpenSSH在认证前报告ED25519主机key与known_hosts第20行不一致。远端候选指纹SHA256:+vqL/Af8vr0nRkVUGfj6KNUvQzZ8lpHGszpp2gGglIo，未经平台核实。没有发送密码/远端命令，没有删除信任记录、替换key、关闭校验、换工具/地址绕过。原因可能是节点身份变化或其它问题，当前未知，不能归因租期结束。

B：fresh原授权地址提示真实password后，只发送本聊天人类原凭据一次。随后真实Permission denied并再次提示password，未发送远端命令。停止尝试并取消认证，实际工具exit1；不把取消称exit0，不重复/猜测凭据，不归因用户没授权或服务器密码错误已证实。session9330已经关闭，禁止复用。

C：fresh原地址实际password后登录；UTC12:06:36.166245+00:00读取GPU UUID GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa匹配，0MiB/0%/空compute；完整Python argv仅本次查询pid15204的['python','-']。capture19 SHA与冻结一致，/data空闲38,557,020,160bytes。查询teacher/run下history/selection/completion返回空字典，不能当新完成核验或判定历史结果不存在；既有完成证据仍永久D。只读查询后明确exit $?，真实SSHexit0，session29197已关闭禁复用。没有SFTP或新capture。

UTC12:08:02.557702+00:00重新核验三份既有永久D最终snapshot.zip：fresh整包SHA等于原receipt，ZIPCRC及唯一成员通过。仍来自真实UTC11:39捕获，绝不将本地核验时间冒新远端capture/完整weight下载。完整模型原件/原异节点CPU记录仍保留。科学权重、冻结源和旧缺口均不改；UTC08:08/10:08未执行不回填。

上述原工具开连接/认证输出与B取消实际结果已保存。C及B后续读取是对原可见工具结果的准确转录，注明来源，未冒远端下载回执；原只读命令另保存。A/B当前UUID、compute、argv、空间未知。已一次向用户提出从平台核对A SSH指纹与B凭据变更的信息请求；没有可信新信息时，不自动反复连接/改信任设置或通知同一阻塞。没有重启Codex，当前异常不同于历史工具审批拒绝。

原10分钟任务保持ACTIVE；继续合法本地研究和永久保存，不因估计期满或小常数正数启动GPU/100，也不续租/关机/停止健康训练或发新任务。第七独立审视完整已审阅，本轮健康/连接事件不向建议聊天重复发送。
'''
(b/'outputs/租期估计时点连接变化与永久D核验.md').write_text(report,encoding='utf-8')
statepath=b/'outputs/研究接续状态.md';state=statepath.read_text(encoding='utf-8')
state=state.replace('本轮没有新远端查询，不冒当前UUID/compute最新。','本轮后续仅C在UTC12:06:36真实UUID/空compute/capture19源/空间通过并exit0；A主机key改变认证前exit1、B原凭据一次拒绝取消exit1，A/B当前状态未知。详见租期估计时点连接变化与永久D核验.md。')
state+=f'\n临期连接补充UTC {now}：A未经平台核实的新ED25519指纹SHA256:+vqL/Af8vr0nRkVUGfj6KNUvQzZ8lpHGszpp2gGglIo与known_hosts不符，严禁修改信任或绕过；B凭据一次被拒绝，已取消exit1；C实际只读exit0，所查run完成路径为空不冒新完成核验。B9330/C29197均已关闭，旧ID禁用。平台指纹/B凭据核对已一次问用户，无新可信信息不重复尝试或通知。UTC12:08:02三既有D ZIP freshSHA/CRC/唯一成员全过，仍11:39真实capture，不冒新remote capture；估计期限经过也未证平台释放。原证据{D.as_posix()}。本轮无新科学结果/训练/续租/关机/重启。\n'
statepath.write_text(state,encoding='utf-8')
lp=b/'outputs/研究建议交流接续.json';ledger=json.loads(lp.read_text(encoding='utf-8'));ledger['updated_at_utc']=now
ledger['lease_connection_events']=dict(directory=str(D),A='HOST_KEY_CHANGED_BEFORE_AUTH_EXIT1_NO_REMOTE_COMMAND',A_candidate_fingerprint='SHA256:+vqL/Af8vr0nRkVUGfj6KNUvQzZ8lpHGszpp2gGglIo',A_platform_verified=False,B='ORIGINAL_CREDENTIAL_DENIED_ONCE_CANCELLED_EXIT1_NO_REMOTE_COMMAND',B_session=9330,C='READONLY_EXPECTED_UUID_EMPTY_COMPUTE_SOURCE_SPACE_EXIT0_COMPLETION_PATHS_EMPTY',C_session=29197,C_actual_utc='2026-10-06T12:06:36.166245+00:00',all_new_sessions_closed=True,new_remote_capture=False,platform_release_or_expiry_verified=False,user_platform_information_request_sent_once=True,automatic_retries_without_new_verified_evidence=False)
ledger['lease_preservation']['latest_local_reaudit']=dict(directory=str(D),audit_sha256=sha(D/'local_preservation_and_connection_audit.json'),utc=audit['utc'],fresh_remote_capture=False)
args=json.loads((D/'official_automation_update.json').read_text(encoding='utf-8'))['args'];ledger['automation'].update(updated_at_utc=now,prompt_sha256=hashlib.sha256(args['prompt'].encode('utf-8')).hexdigest(),original_tool_receipt=str(D/'official_automation_update.json'),result='EXISTING_AUTOMATION_OFFICIAL_UPDATE_SUCCESS_ACTIVE')
lp.write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
with (b/'outputs/租期末强化保存与下一阶段研究决策.md').open('a',encoding='utf-8') as f:f.write(f'\n临期实际UTC {now}：C12:06:36只读核验/退出成功；A SSH key变化、B原凭据一次拒绝，停止相关连接待平台核对。三既有D快照12:08:02 freshSHA/CRC过，未新增远端capture或确认平台释放。第七完整已读，无待审视。原证据{D.as_posix()}。\n')
record=D/'records';record.mkdir(exist_ok=False)
sources=list(D.glob('*.json'))+[D/'readonly_query.txt',statepath,lp,b/'outputs/租期末强化保存与下一阶段研究决策.md',b/'outputs/租期估计时点连接变化与永久D核验.md',b/'work/update_expiry_connection_state_v1.py']
manifest={}
for p in sources:
 q=record/p.name;assert not q.exists();shutil.copy2(p,q);manifest[q.name]=dict(sha256=sha(q),bytes=q.stat().st_size)
with zipfile.ZipFile(D/'original_connection_and_preservation_records.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for name in manifest:z.write(record/name,name)
with zipfile.ZipFile(D/'original_connection_and_preservation_records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(manifest)
 for name,entry in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256']
(D/'permanent_record_receipt.json').write_text(json.dumps(dict(status='ACTUAL_CONNECTION_ERRORS_READONLY_C_AND_PRIOR_D_REAUDIT_SHORT_STATE_PERMANENT_SHA_ZIP_CRC_PASSED',utc=now,members=manifest,package_sha256=sha(D/'original_connection_and_preservation_records.zip'),whole_research_or_lease_completed=False,new_remote_capture=False),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':'PERMANENT_CONNECTION_EVIDENCE_AND_SHORT_STATE_SAVED','members':len(manifest),'C_free':shutil.disk_usage('C:/').free,'D_free':shutil.disk_usage('D:/').free,'platform_expiry_confirmed':False}))
