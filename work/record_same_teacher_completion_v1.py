from pathlib import Path
import datetime, hashlib, json, shutil, zipfile
w=Path(__file__).parent;repo=w.parent;outputs=repo/'outputs'
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
context=read(w/'same_teacher_actual_context_20261006T0536Z.json');d=Path(context['destination'])
joint=read(d/'completed_joint_audit.json');assert joint['status']=='THREE_SAME_FOLD_TEACHER_GPU_COLLECTIONS_D_ORIGINALS_OTHER_NODE_CPU_AND_ATOMIC_CAPTURE_JOINED_VERIFIED'
sessions=read(outputs/'同折教师采集完成六会话退出核验_20261006T055329Z.json')
assert all(s['result']['value']['exit_code']==0 for s in sessions['sessions'])
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
for name,source in [('同折教师实际采集与保存联合核验.json',d/'completed_joint_audit.json'),
                    ('同折教师实际采集完成快照联合核验.json',d/'completed_capture_audit.json')]:
    shutil.copyfile(source,outputs/name);assert sha(source)==sha(outputs/name)
table=['| 折/节点 | FIT/INNER行 | 正式子进程 | 自然完成UTC | 正式耗时 | 峰值显存 | INNER重放/换标签误差 | 异节点CPU |',
       '|---|---:|---:|---|---:|---:|---:|---|']
for node,r in joint['nodes'].items():
    table.append(f"| {r['fold']}/{node.upper()} | {r['arrays']['fit']['rows']}/{r['arrays']['inner']['rows']} | {r['actual_child_pid']} | {r['actual_finished_utc']} | {r['seconds']:.2f}s | {r['peak_allocated_bytes']/1024**3:.4f}GiB | {r['inner_replay_max_error']}/{r['label_replacement_max_error']} | {r['independent_cpu_node'].upper()} |")
report=f'''当前三折同T_k FIT/INNER实际采集已完成，UTC{now.isoformat()}。原始GPU回执/数组、D永久保存、A→B/B→C/C→A独立NumPy CPU原回执和完成原子快照已联合核验；不是新的100轮控制训练或幅度校准收益。

本轮先确认旧C会话11935已Unknown process，再正常fresh SSH/SFTP成功发送命令。三实际GPU UUID全部吻合，源/完整argv/空compute/三教师100完成/空间重新核验。历史审批拒绝仍保留，但不能再沿用为当前阻塞；未修改工具底层、设置、浏览器或使用替代客户端。SSH62016/95684/40329及SFTP39820/66853/62545均明确exit/bye实际exit0。

'''+ '\n'.join(table)+f'''

严格原完整选中权重加载，301状态张量。FIT-only归一化缓冲exact；完整INNER原输入与原选中预测重放0，毒化标签替换0；同折FIT/INNER样本的真实标签字段未用于采集，数据集dummy label=0，OUTER角色请求被拒绝。原构造创建临时optimizer/scheduler随后丢弃，零优化器步骤，参数SHA前后相同，无参数梯度。没有新OUTER推理或DEV/TEST请求。全程峰值包含构造/加载；20min/4GiB/1GiB预算通过，不把原构造前计量漏洞的v1冒已执行。

六新NPZ各仅row_ids/fold/mu，FIT695/715/714与INNER153/142/143，原行序和折号均独立审核。总2562个折内标量行包含跨折重复样本，不是新2562样本，也不是新OOF。S_k只能使用自身T_k的FIT/INNER目标，不能混别折教师目标造成间接外折泄漏。

冻结采集源和计划未修改：work/correction_calibration_collection_v2_20261006T0414Z，plan SHA57b740fed4ee9ac2d9736956f2e6cbc3b9dcd30b102610abe49f563a493738f7，collector SHA7e25fd0f09da4b61418931108efc49a506a6926f47da7e7f34e5574f36844e47。远端根{context['remote_root']}，目录0414只是预先冻结ID，实际执行时刻见原回执；CPU根/data/coding/group_teacher_v1_20261005T1650Z/cpu_preservation/same_teacher_collection_20261006T053835Z。

永久D {d.as_posix()}：三原小文件ZIP各36成员，CRC/唯一成员/全部SHA通过；两个阶段均实际自然exit0。CPU为异训练节点文件/原回执/NumPy数组审核，原CPU回执、子进程exit和数组审核已实际下载，不是CPU完整模型前向。旧完整权重没有再次上传/下载。

预检后actual capture19 UTC05:41:32/32/42附近，完成后UTC05:51:35附近（精确时刻见receipt）。每份实际CAPTURE_COMPLETE/exit0后先receipt再ZIP下载；完整SHA/CRC/成员唯一、完整新子根科学源/argv/原回执/数组、CPU新原回执及全部之前冻结成员和大SHA均独立审核。旧大权重freshSHA引用不冒再次下载。新检查/保存助手单独留源码，旧capture19及完成教师科学源未修改。

下一优先：依照三级消息控制建议评估与短实验修订设计.md冻结匹配的短GPU机制实验。生成方向、有限幅度与接受器分开；p_F锚和目标detach，lambda=0精确回退、终端真实前向的消息减半/输出变化上限、主辅梯度与长度1二阶必须实际验证。CAL18视频418行/EVAL34视频863行固定角色不改，仅CAL拟合；EVAL预测冻结后统计，已有全TRAIN探索不能冒全新确认集。当前仍没有真实CAL lambda/epsilon/上限拟合、没有新求解器或学生GPU预检/正式100。已有全TRAINfit/DEVselected参考A仍非全流程crossfit，完整参考编码器/流/供体/读出需按折另冻结才可讨论整体泛化。

最新控制实际分数不变：TRAIN原F MSE.0161276872，旧学习.0152679208（降5.33%），新OOF驱动末步.0767956564/害74.89%，μ选择.0789019134/害75.10%，属于负结果。三个已完成OOF教师MSE.6293898611/MAE.5965080822仅教师质量；两者不直接比较。本地标签已知受限输出组合Oracle6.61%仅事后见证，其约5.91%系数不可部署。九流100模型、三个教师100、TRAIN Oracle均保留，不重跑；原C仅10失败、旧C最终及旧14:30缺口不回填。

24h期限仍仅估Oct6UTC12:08:17/BJ20:08:17，平台未核。UTC08:08/10:08/11:38强化真实动态保存尚待实际执行，全部研究及租期保存未完成。
'''
(outputs/'同折教师FIT_INNER实际采集与保存核验.md').write_text(report,encoding='utf-8')
state=outputs/'研究接续状态.md';history=outputs/('研究接续状态_实际采集前历史_'+stamp+'.md')
shutil.copyfile(state,history)
short=f'''更新UTC{now.isoformat()}。本聊天GPU命令现可实际发送；旧11935 Unknown后fresh三SSH/SFTP正常，不沿用历史审批阻塞，不称底层工具修复。三UUID/完整argv/空compute/教师100完成/源/空间实际核验。SSH62016/95684/40329及SFTP39820/66853/62545全部明确exit/bye真实exit0，旧ID禁用，下轮fresh实际password提示后仅人类原凭据，禁密码文件/命令/自动提示/猜测。

本轮实质完成：同T_k FIT/INNER v2零标签GPU采集三折；A8264/B6743/C13299自然exit0 UTC05:47:22/05:44:29/05:47:24。FIT/INNER695/153、715/142、714/143；正式181.64/22.39/185.01秒，全程峰值约1.5834GiB，INNER重放0/换标签0/FIT统计exact/参数SHA不变/零optimizer步骤。临时optimizer对象构造后丢弃准确记录。六NPZ只row_ids/fold/mu，无新OUTER/DEV/TEST。

原GPU回执、6数组和三36成员小ZIP永久D {d.as_posix()}；异节点CPU A→B/B→C/C→A原回执/exit/NumPy数组审核已下载，非CPU模型前向。两次actual capture19完成后receipt→ZIP下载，全SHA/CRC/member和旧冻结源/完成证据/大SHA不变通过；完成最新capture实际05:51:35附近，精确看receipt。联合原证据见同折教师实际采集与保存联合核验.json、同折教师实际采集完成快照联合核验.json、同折教师FIT_INNER实际采集与保存核验.md。远端根{context['remote_root']}，源/plan仍本地冻结v2，完整旧权重未重传。

下一：新短幅度与接受机制实验先冻结并真实GPU预检，再执行；锚/目标detach、lambda0精确回退、真实消息终端前向/减半/输出变化上限、标签/主辅梯度/长度1二阶和预算/保存依赖全核。当前真实CAL拟合lambda/epsilon/上限、新消息求解器、学生GPU预检/正式100均未执行。S_k只同T_k自己的FIT/INNER标量；CAL18视频418行/EVAL34视频863行角色固定，只CAL拟合、EVAL预测SHA后统计；已全TRAIN探索/全TRAINfit+DEVselected参考不能冒全新确认或全流程crossfit。方向与接受分开，旧有效方向+教师受控补充只是假设，事后Oracle系数禁止部署。

科学边界不变：九流100/full/20诊断/D/CPU，三91818教师100/best85/45/68/full/严格重放0/D/CPU，1281/52视频OOF及TRAIN Oracle均完成保留不重跑重传。原C仅10失败，旧C最终和旧14:30缺口不回填。OOF教师MSE.6293898611/MAE.5965080822仅教师质量。固定C2 TRAIN原F.0161276872/旧学习.0152679208降5.33%/新OOF末步.0767956564害74.89%/μ选择.0789019134害75.10%，新生成负结果。标签已知有限Oracle65.11%与输出凸组合6.61%不冒泛化/严格上界。旧流只520506供体头，教师184749003参数全微调，勿混称端到端。

capture19 SHA6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad，公共root/data/coding/soft_vector_research_20261005T1220Z、deployment/data/coding/jacobian_aligned_v2_deployment_20261005T1327Z、实际clock唯一stamp。新根须覆盖；actual CAPTURE_COMPLETE/exit0后receipt→ZIP/全SHA/CRC/member，旧大权重引用非新下载。唯一新A53449/B53416/C53458及UUID见v2 plan和原回执，旧址/N/R永禁连。D优先逐次fresh查C/D，禁删旧文件/改完成冻结源/subagents/设置/浏览器/续租/关机/停健康训练/TEST挑结构/五seed稳定/SOTA/语义真值。

24h仅估Oct6UTC12:08:17/BJ20:08:17平台未核；UTC08:08/10:08/11:38仍须实际强化保存，不能伪称或回填。整体研究/后续实验/租期保存未完。按十分钟频率，健康未变或无行动静默，只实质结果/新失败/完成/必要行动通知。
'''
state.write_text(short,encoding='utf-8')
dest=d/'research_records_completed';dest.mkdir(exist_ok=False)
files=[p for p in w.glob('*same_teacher*.py') if p.is_file()]
files += [w/'same_teacher_actual_context_20261006T0536Z.json']
files += [outputs/n for n in ['研究接续状态.md','同折教师FIT_INNER实际采集与保存核验.md','同折教师实际采集与保存联合核验.json','同折教师实际采集完成快照联合核验.json','同折教师采集完成六会话退出核验_20261006T055329Z.json']]+[history]
space={drive:shutil.disk_usage(drive).free for drive in ['C:/','D:/']};assert space['D:/']>sum(p.stat().st_size for p in files)+1024**2
members={}
for source in files:
    relative=source.relative_to(repo);target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target);assert sha(source)==sha(target)
    members[relative.as_posix()]={'bytes':source.stat().st_size,'sha256':sha(source)}
zpath=dest/'research_records.zip'
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
    for name in members:z.write(dest/name,name)
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,m in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==m['sha256']
proof={'status':'COMPLETED_GPU_COLLECTION_RESEARCH_SHORT_STATE_SOURCES_AND_ORIGINAL_SESSION_EXITS_D_SHA_CRC_VERIFIED',
       'utc':now.isoformat(),'destination':str(dest),'fresh_free_bytes':space,'members':members,'zip_sha256':sha(zpath)}
out=outputs/'同折教师实际采集接续与科学记录永久D保存.json';out.write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copyfile(out,dest/out.name);assert sha(out)==sha(dest/out.name)
print(json.dumps({'status':proof['status'],'files':len(members),'destination':str(dest),'fresh_free_bytes':space,'zip_sha256':proof['zip_sha256']}))
