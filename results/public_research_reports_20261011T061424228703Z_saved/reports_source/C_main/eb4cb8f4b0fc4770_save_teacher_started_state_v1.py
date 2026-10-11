from pathlib import Path
import datetime,hashlib,json,shutil
w=Path(__file__).resolve().parent;o=w.parent/'outputs';d=Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_live_snapshots_20261006T022505Z')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
archive=o/'研究接续状态_三折教师启动前_20261006T0231Z.md';assert not archive.exists();shutil.copy2(o/'研究接续状态.md',archive)
text=f'''最新真实UTC {now}。

# 多模态流研究短接续

实质进展：按视频隔离标量教师三折正式100轮已真实启动，seed91818。A/B/C fold0/1/2，wrapper6772/5601/11556、child6773/5602/11557，UTC02:23:31实际启动；UTC02:30:42 fresh真实UUID/完整ps argv/compute/历史/源SHA/空间通过，20/19/19轮且增量健康，未完成100、未产最终OOF，不能提前断言收益。GPU内存约6.63GB；compute宿主命名空间PID12105/55751/12104与容器childPID不同，按UUID及实际argv核对。远端根 /data/coding/group_teacher_v1_20261005T1650Z，公共 /data/coding/soft_vector_research_20261005T1220Z。目录1650是ID，不是本次实际启动时间。

新机制：原v1三GPU四更新/七模块检查过，但进一步全参数检查发现作者遗留pooler两个参数无梯度，三个原检查真实自然exit1，原回执日志保留。新runtime_v2/plan_v3只在优化器前移除完全未用pooler，其他初始张量与v1逐一相同、原32输入预测0。真实修订预检291保留参数张量全有限梯度/四次更新、fit-only统计/标签替换0/outer访问guard/严格磁盘重载0/100订单/三折非统计初始化SHA通过；峰值5742105600bytes。三折fit/inner/outer695/153/433、715/142/424、714/143/424，视频互斥，只官方TRAIN52视频1281条。

新教师准确协议：公共DeBERTa+随机任务初始化，禁全TRAINfit A/C2初始化。移除8流与无用pooler，全量微调184749003保留参数，291参数张量每正式更新都核有限非None梯度；与旧冻结骨干流控制不同。完整100轮，只按INNER样本加权MSE最早严格最小选模，完成后整权重严格磁盘重载/inner原输入≤1e-6，再一次零标签OUTER预测μ标量，outer真实标签不进fit/统计/inner选模。DEV/TEST不读取；现有A参考仍全TRAINfit/DEVselected，跨折只转μ，不能称全流程crossfit或方向策略已更新。

冻结源：train_group_teacher_v1.py SHA7c94f90ff648909e0abe1a6ccdb7a88543650142c6633ebadb8a0062068b8791；runtime_v2 SHA6f9074453ed15c32537efb7803210801f6123cc1647c6f4fe4d7126e2616dc9e；plan_v3 SHAf0878166da09433c27119a08fdfd8682f9b9e29f2bfe0a758f29eee45cf131a4；formalplan_v4 SHA9d06176ababb06bc0c97daf4797b20a535d30f40cd8eb9e9e20fdabb81367bbe。统一非统计init SHA3813506519a54e1a7baa54ee37446b00e6b2f71f3a234621da1328db270a9567。100订单初始化/标签/梯度预检与真实GPU预算、三新完整weights≥8GiB保存空间先过才启动。实测保守预算约87–109分钟，不保证实际租期。

预检保存已完成：D group_teacher_precheck_20261006T0210Z a/b/c原receipt/witness/replay/旧失败/source/正式plan/预算。B和C同root/cpu_preservation/precheck_v2各41小文件原SHA及NumPy数组审核，两个原CPU回执已下载独立审核；A/C原证据异节点B、B原证据异节点C，B自检不能冒异节点。初始整模型仅GPU磁盘重载/fresh整体SHA引用，未下载D或作CPU整模型副本。报告 视频隔离教师机制核验与正式预算.md、v2三GPU机制回执独立核验.json、v2异节点CPU原回执核验.json、正式100冻结核验.json。

最新原子capture19已三节点核SHA6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad；--root公共根 --deployment /data/coding/jacobian_aligned_v2_deployment_20261005T1327Z --stamp真实clock唯一ID。覆盖教师全部新root和CPU、Oracle及全部旧证据；跳过.pending，对大文件打开稳定inode流式SHA。真UTC02:25:38三快照，真实CAPTURE_COMPLETE/childexit0后receipt/ZIP/exit文本已永久D group_teacher_live_snapshots_20261006T022505Z/{{a,b,c}}；全部ZIP/memberSHA/CRC/成员唯一、旧完成及预检证据不变、三训练协议/初始化/订单/完整实际argv/7/6/6轮live独立审核通过。目录022505只是ID。新完整selected模型只freshSHA引用，不冒D下载或完成异节点CPU保存。审计源audit_soft_snapshot_v19.py、audit_group_teacher_formal_snapshot_v1.py、audit_group_teacher_live_integrity_v1.py；证明视频隔离教师正式启动联合动态保存核验.json。

下一优先监管三教师100自然完成，不能停健康训练或改冻结源。完成后新整checkpoint每折全SHA/ZIPCRC/100最早INNER最小/严格磁盘原输入重放/outer原数组核验，fresh查D足够再三完整weights永久保存和异训练节点CPU完整副本、下载原CPU回执，不混旧证据。再独立合并1281原行序OOF及视频映射、核标量质量，根据OOF与Oracle证据设计后续控制而非填卡。当前D实查21270261760bytes/C3820146688bytes，禁删旧文件。

九个新旧阶段100模型91815/91816/91817A/B/C2的完整weights/20诊断/D/CPU已经完成保留，不重跑重传。原C v1仅10非有限失败，C2修订100不是旧optimizer恢复。既有C2 DEV MAE.599325/MSE.677609未优于对照。旧84/v2/v3/v4和旧v5A/B不重复，旧v5C最后81/临时31不能冒100；旧C最终/旧14:30保存缺口不回填；旧A异C原回执仅终端证据不冒已下载。

固定C2best37 TRAIN Oracle诊断已C PID10681 UTC01:57:09–01:58:09自然exit0且D train_oracle_completed_20261006T0158Z/c、B异节点9文件原CPU回执/原数组审核完成，不重复。TRAIN raw/learned/oracle/zero/mismatched MSE .01612769/.01526792/.00562681/.02165725/.13377095，学习降5.33%/标签Oracle65.11%；这是全TRAINfit/DEVselected权重的标签已知有限步诊断，不是严格上界/泛化/语义真值。报告TRAIN_Oracle可达性实验分析.md。

唯一新P4：A REDACTED_SERVER_HOST.invalid:53449 UUID GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7；B REDACTED_SERVER_HOST.invalid:53416 UUID GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067；C REDACTED_SERVER_HOST.invalid:53458 UUID GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa。旧址/N/R禁连。工具实际fresh连接/命令已正常，不沿用审批阻塞也不称底层代码修复。当前六会话SSH21296/97860/32862和SFTP11681/45461/62885明确exit/bye真实exit0，证据视频隔离教师正式启动轮会话关闭记录.json；旧ID禁止write_stdin，健康训练detach继续。之后fresh连接，实际password提示后只本聊天人类原凭据，禁密码文件/命令/自动提示/猜测，退出核exit0。

24h首次Oct5UTC12:08:17仅估Oct6UTC12:08:17/BJ20:08:17，平台期限未核；UTC08:08/10:08/11:38强化真实动态保存，完成即永久D+异节点CPU，不伪称期限捕获或回填。仅官方TRAIN1281/dev229，禁TEST选结构/五seed稳定/SOTA/共享互补干扰语义真值。禁subagents/设置/浏览器/续租/关机/停健康训练/改已冻结源；原十分钟频率，健康未变静默，只实质结果/失败/完成/必要用户行动通知。研究、教师与租期保存未完，不能称整体完成。
'''
(o/'研究接续状态.md').write_text(text,encoding='utf-8')
files=[o/'研究接续状态.md',o/'视频隔离教师正式启动轮会话关闭记录.json',archive,w/'save_teacher_started_state_v1.py']
records={}
for f in files:
 dest=d/f.name;assert not dest.exists();shutil.copy2(f,dest);assert sha(dest)==sha(f);records[f.name]={'bytes':f.stat().st_size,'sha256':sha(f)}
proof=o/'视频隔离教师正式启动短状态永久保存.json';assert not proof.exists();proof.write_text(json.dumps({'utc':now,'files':records,'destination':str(d)},ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(proof,d/proof.name)
print('TEACHER_STARTED_SHORT_STATE_AND_CLOSURE_PERMANENT_D_SHA_PASSED')
