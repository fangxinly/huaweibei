from pathlib import Path
import datetime,hashlib,json,shutil,zipfile
cwd=Path(__file__).parent.parent;out=cwd/'outputs';work=cwd/'work'
d=Path('D:/CodexBackups/selective_flow_20261003_1105/native_radius_cal_mechanism_actual_20261006T072031Z');m=d.parent/'matched_message_pools_actual_v2_20261006T065508Z'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();now=datetime.datetime.now(datetime.timezone.utc).isoformat()
records=d/'research_records';records.mkdir(exist_ok=True);prior=records/'preceding_state';prior.mkdir(exist_ok=True)
for name in ['研究接续状态.md','研究建议交流接续.json']:
    if not (prior/name).exists():shutil.copy2(out/name,prior/name)
assert json.loads((d/'joint_preservation_audit.json').read_text())['status']=='CAL_NATIVE_RADIUS_D_ORIGINAL_GPU_OTHER_NODE_CPU_AND_ACTUAL_C_B_CAPTURE_JOINTLY_VERIFIED'
assert json.loads((m/'joint_preservation_audit.json').read_text())['status']=='MATCHED_MESSAGE_POOLS_D_ORIGINALS_OTHER_NODE_CPU_AND_ACTUAL_B_CAPTURE_JOINTLY_PASSED'
cl=json.loads((d/'session_closure.json').read_text());assert all(s['result']['value']['exit_code']==0 for s in cl['sessions'])
review=Path('C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/outputs/第四次独立审视_负结果与原生位移半径.md');shutil.copy2(review,records/review.name)
coord=json.loads((out/'研究建议交流接续.json').read_text(encoding='utf-8'));coord['updated_at_utc']=now
coord['review_thread'].update(status='fifth_review_sent_pending',last_wait_cursor='5c299d87-8d70-412a-94b9-bdc45953b3ac:15',last_completed_turn='01a11012-98d1-7f31-ae0c-f9ff8e6ad222',current_turn=None)
if not any(x.get('sha256')==sha(review) for x in coord['received_reviews']):coord['received_reviews'].append({'path':review.as_posix(),'sha256':sha(review),'completed_utc':'2026-10-06T07:24:55Z','read_and_considered':True,'accepted':'仅一次CAL零标签原生位移覆盖；现已完成418。停止同EVAL收益扩张和新100；下一隔离流参考协议。'})
for batch,root,filename in [('matched_message_pools_v2_20261006T065508Z',m,'fourth_review_send.json'),('native_radius_CAL_20261006T072031Z',d,'fifth_review_send.json')]:
    if not any(x.get('batch')==batch for x in coord['evidence_batches']):coord['evidence_batches'].append({'batch':batch,'status':'sent','joint_evidence':(root/'joint_preservation_audit.json').as_posix(),'joint_sha256':sha(root/'joint_preservation_audit.json'),'prompt_preserved':(root/'research_records'/filename).as_posix(),'repeat_send_allowed':False})
coord['scientific_state']='固定EVAL八臂负结果与CAL零标签覆盖均实际完成/D/B CPU/actual capture联合通过；未新100；完整隔离流参考仅草案。第五建议待回复。'
coord['permanent_current_evidence_directory']=d.as_posix()
for item in coord['deferred_suggestions']:
    if '参考' in item['suggestion']:item.update(status='protocol_draft_next_priority_not_GPU_prechecked',reason='现有plain teacher非flow；需要公共预训练随机全链和FIT-only来源及实测预算。')
    if '100' in item['suggestion']:item.update(reason='匹配池未胜F/native，有限池Oracle无教师并集优势；CAL覆盖仅机制，公平流参考未过。')
(out/'研究建议交流接续.json').write_text(json.dumps(coord,ensure_ascii=False,indent=2),encoding='utf-8')
state=f'''更新UTC {now}。以此短接续为准，旧完整报告按需查，不重注入长heartbeat。当前没有运行的新训练，后续研究和租期保存未完。

1) 新CAL视频等权拟合已完成/D/B CPU/actual capture；lambda三折0/.0127280875/.0624909847，输出cap.030717943/.030769765/.035916110，仅经验参数。详情固定CAL视频等权残差信号实际结果.md。
2) 新真实匹配消息八臂正式child14820 UTC07:01:20自然exit0。全部预测SHA后读固定EVAL863/34视频：F MSE.014974953、旧native.014086934、CAL_OT.015010218、CAL_Opm.015021068；CAL_OT44修改24有害。同cap有限标签Oracle_OT.014314224弱于Opm.014165174，不是部署策略或上界。停止扩大这版末端方向/接受器，不新100。D matched_message_pools_actual_v2_20261006T065508Z完整51MB原数组/42原包/B原CPU三审核与actual C07:05/B07:10 capture联合过。详情匹配消息候选对照实际结果与优化决策.md。
3) 原生位移共同半径仅CAL418/18零标签GPU机制child14988 UTC07:21:24自然exit0，12.016s/52,818,432bytes；无拟合、EVAL前向/指标、路径重解或接受后新控制。F/native重放2.38e-7，参数不变。OT cap有效唯一候选均值1.8325→8.8852、非F有效且输出变化>1e-6行96→418，旧native幅度1仅396行cap内。418半径均非零，未验证一般零半径GPU回退或新选择器；覆盖不等于收益。D native_radius_cal_mechanism_actual_20261006T072031Z完整16,399,068bytes NPZ SHA f6685f7797f0deca4ceaf12858b2ba26831c9ca6179313f8497a1cf61d6ec8b6，13原包/B原NumPy回执/actual C07:23:16+B07:24:49 capture全过。非F细分析只本地原数组，非新异节点脚本执行。详情原生位移半径CAL零标签机制实际结果.md。
4) 下一实质优先视频隔离完整流参考可行性协议草案.md：固定fold0 FIT695/INNER153/OUTER433，公共DeBERTa+随机任务/完整fixed flow/reader/donor/readout，FIT-only统计，新源与GPU预检尚未实现。现有teacher plain masked mean-pool fusion不是flow参考，禁旧A/C2/老师任务权重或全TRAIN缓存初始化冒隔离。先原输入整流/两FIT有限全参数梯度/长度1/INNER零标签磁盘重放/实测峰值时间/完整权重永久空间及至少2h保存余量，再独立正式协议/100订单；不填卡跳依赖。不再对已读EVAL扫半径/惩罚报确认收益。
5) 九旧流100/full/20诊断、三91818老师100/full、1281/52 OOF、TRAIN Oracle与同折FIT_INNER采集D/CPU保留不重跑重传。原C仅10失败，旧C最终/旧14:30缺口不回填。老师OOF MSE.6293898611/MAE.5965080822仅老师质量；C2 DEV MAE.599325/MSE.677609仍未改善A/B。旧流仅520506供体头、老师184749003参数全微调，勿混称端到端。当前参考全TRAINfit/DEVselected、全TRAIN曾探索，非全新确认或全流程crossfit；经验cap/epsilon非真风险安全界，Oracle非泛化上界。
6) 本轮C/B SSH67559/89114与SFTP18058/32904全部明确exit/bye actualexit0，禁止旧IDwrite_stdin。工具当前可用不称底层修复。需要时fresh真实password提示后仅人类原凭据，先UUID/完整argv/compute/源/完成/空间；新审批拒绝不换路绕过。仅新A53449 UUID8e10cc68-ef5e-9064-64ba-fec82c5089c7/B53416 a259ab5e-884a-ec9f-4208-1bc34a6e5067/C53458 2a0c83a4-0a1a-494d-6e80-0c756fb075aa，域名见冻结plan；旧址/N/R永禁连。A无本轮fresh材料，不冒新三节点capture。
7) 独立建议同聊天01a10fcb-6663-70a2-9a76-60e5634d0c03第四建议完成/全文审阅并采纳一次覆盖与参考优先；第五新覆盖原证据D/CPU后已发送一次，待回复不冒已返。按研究建议交流接续.json去重，只实质新结果保存后发送，建议仅分析不连GPU/改主源/训练/子代理，主聊天负责实验。
8) capture19 SHA6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad，公共root/data/coding/soft_vector_research_20261005T1220Z、deployment/data/coding/jacobian_aligned_v2_deployment_20261005T1327Z；新根必须覆盖，真实CAPTURE_COMPLETE/exit0后receipt→ZIP全SHA/CRC/member/原件大引用审核。旧full freshSHA引用非再次下载。D优先每次fresh C/D盘禁删/改完成冻结源。
9) 租期仅估Oct6UTC12:08:17/BJ20:08:17，平台未核。UTC08:08/10:08/11:38强化真实动态保存尚待实际执行，不伪称/回填，完成即保存。禁subagents/设置/浏览器/续租/关机/停健康训练/TEST挑结构/五seed稳定/SOTA/语义真值。原10分钟频率，健康未变或仅本地准备静默，只实质结果/新失败/完成/必要用户行动通知。
'''
(out/'研究接续状态.md').write_text(state,encoding='utf-8')
names=['研究接续状态.md','研究建议交流接续.json','匹配消息候选短实验执行接续.json','原生位移半径机制执行接续.json','匹配消息候选对照实际结果与优化决策.md','原生位移半径CAL零标签机制实际结果.md','视频隔离完整流参考可行性协议草案.md']
for name in names:shutil.copy2(out/name,records/name)
for name in ['record_matched_pool_results_v1.py','diagnose_native_radius_cal_v1.py','run_native_radius_cal_v1.py','audit_native_radius_cal_v1.py','prepare_native_radius_cal_v1.py','prepare_native_radius_cal_v2.py','verify_native_radius_cpu_v1.py','audit_native_radius_capture_v1.py','package_native_radius_originals_v1.py','analyze_native_radius_cal_coverage_v1.py','record_native_radius_results_v1.py','finalize_current_research_state_v1.py']:shutil.copy2(work/name,records/name)
for file in ['nonF_CAL_coverage_analysis.json','joint_preservation_audit.json','session_closure.json']:shutil.copy2(d/file,records/file)
manifest={p.relative_to(records).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in records.rglob('*') if p.is_file() and p.suffix not in ['.zip'] and p.name!='records_manifest.json'}
(records/'records_manifest.json').write_text(json.dumps({'status':'LOCAL_CURRENT_RESEARCH_RECORDS_SHA_VERIFIED','updated_utc':now,'remote_capture_or_CPU_rerun_claimed':False,'files':manifest,'fresh_free_D':shutil.disk_usage(d).free,'fresh_free_C':shutil.disk_usage(cwd).free},ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(records/'research_records.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in manifest:z.write(records/name,name)
    z.write(records/'records_manifest.json','records_manifest.json')
with zipfile.ZipFile(records/'research_records.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,it in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==it['sha256']
print(json.dumps({'status':'COMPACT_STATE_AND_DECISIONS_SAVED_D','local_record_files':len(manifest),'record_zip_sha256':sha(records/'research_records.zip'),'free_D':shutil.disk_usage(d).free,'free_C':shutil.disk_usage(cwd).free}))
