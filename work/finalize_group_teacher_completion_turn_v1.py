from pathlib import Path
import argparse,json,datetime,hashlib,shutil
base=Path(__file__).resolve().parent.parent
p=argparse.ArgumentParser();p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--b-oof-snapshot',type=Path,required=True);a=p.parse_args()
get=lambda name:json.loads((base/'outputs'/name).read_text(encoding='utf-8'));sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
full=get('视频隔离教师完整模型CPU与完成快照联合保存核验.json');oof=get('视频隔离教师1281行OOF独立数组审核.json');extra=get('视频隔离教师合并OOF独立CPU新快照核验.json');closures=get('视频隔离教师完成轮会话关闭记录.json')
assert full['status']=='THREE_COMPLETED100_TEACHERS_D_FULL_WEIGHT_ORIGINAL_SEVEN_TWENTY_EXTRA_INDEPENDENT_CPU_RECEIPTS_AND_ATOMIC_SNAPSHOTS_JOINED_VERIFIED'
assert extra['status']=='COMBINED1281_OOF_INDEPENDENT_CPU_ORIGINAL_RECEIPTS_AND_NEW_REAL_B_ATOMIC_SNAPSHOT_JOINED_VERIFIED_OLD_COMPLETED_EVIDENCE_UNCHANGED'
assert all(x['result']['exit_code']==0 for x in closures['fresh_B_explicit_exit_bye'])
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ');parent=Path('D:/CodexBackups/selective_flow_20261003_1105');assert shutil.disk_usage(parent).free>2*1024**3;dest=parent/('group_teacher_completed_proofs_'+stamp);dest.mkdir()
state=base/'outputs/研究接续状态.md';shutil.copy2(state,dest/'previous_研究接续状态.md')
text=f'''最新真实UTC {now.isoformat()}。

# 多模态流研究短接续

实质完成：seed91818按视频隔离标量教师三折各100轮，UTC02:57:13/02:57:34/02:57:54自然exit0，选中85/45/68。原训练wrapper6772/5601/11556、child6773/5602/11557均已退休，未停健康训练。原根/data/coding/group_teacher_v1_20261005T1650Z，公共/data/coding/soft_vector_research_20261005T1220Z，原根1650只是ID。正式全量微调184749003参数/291张量每步有限梯度；新教师与旧冻结骨干流控制不同。

三新整模型各739126305bytes/301状态张量，原节点训练和原节点组装，严格磁盘重载INNER原输入误差0。D group_teacher_completed_20261006/a,b,c七required加20extra全SHA/ZIPCRC/100最早INNER最小/原数组/映射/100订单独立通过。A/B/C整SHA分别338df042d648d3bdcb221bc0cacbd3d5574a1dd1efac49b1823f091a75706695、5727b61173f6efc03402d754553129b44e2c1d4b817a43572b7a984055644c80、d5ee79575c49c241787b7e7dd3d044089b4858186d04505574c9bc66293e771e。独立CPU A→B/B→C/C→A，三原回执已下载审核，301有限CPU张量SHA一致/无CUDA初始化。不是另跑整模型CPU前向，不改原completion当时save=false字段。原C100history尚无completion时门控拒绝，后实际完成才保存。

三折fit/inner/outer695/153/433、715/142/424、714/143/424，TRAIN52视频1281行。公共DeBERTa+随机任务初始化、fit-only统计、只INNER100最早最小选模，OUTER零标签一次μ标量预测。1281原行μ OOF已D group_teacher_oof_20261006T0308Z，原数组/原TRAIN标签/视频映射/各fold及baseline另一个独立NumPy重建通过。MSE0.6293898611/MAE0.5965080822，fit-only均值MSE2.3107411669/MAE1.3247986001，相对MSE下降72.76%，52/52视频更好；只标量教师质量，不是新控制收益。原参考A仍全TRAINfit/DEVselected，不称全流程crossfit；OOF MAE不能和C2 DEV MAE直接比较。

合并OOF另有新B CPU保存与19文件SHA/CRC/原fold/原行/标签/metric重建、原回执/exit0下载，root /data/coding/group_teacher_v1_20261005T1650Z/cpu_preservation/oof_20261006T031527Z。该新CPU副本只含标量和小原数组，不重复完整权重。原μ NPZ SHA cff1a52f7e828c96cacf070ae4e051a5c34042ebb9435f44f5b3242cad91a7ea。联合证据视频隔离教师合并OOF独立CPU新快照核验.json。

当前capture19 SHA6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad原源不改。三实际完成快照UTC03:08:02.978–03:08:02.983，D {a.snapshot.as_posix()}，CAPTURE_COMPLETE/exit0后receipt/ZIP/exit原件全SHA/CRC/member唯一与旧源/旧完成/预检CPU不变、三真实100、D原full/CPU原回执和large引用联合过。OOF新根已由B实际{extra['actual_B_capture_utc']}原子快照额外覆盖，D {a.b_oof_snapshot.as_posix()}，19小文件/原CPU回执全核，之前三完成证据不变。目录ID不当实际时刻，fresh大SHA引用不冒再次下载。

最新工具状态：旧本轮B SSH10365/SFTP70527远端reset实际exit1，A/C SSH99748/67905也reset实际exit1；旧SFTP9698/19911发bye后tool exit0但输出含reset，不能混称所有原会话正常exit0。原因未知，不归因平台/租期/密码。一次fresh B SSH14252/SFTP97105实际password后正常、UTC03:17:30 UUID/完整原argv/100/源pins/空compute/空间核过，OOF CPU与新capture完成，随后两fresh会话明确exit/bye真实exit0。全部上述ID现已关闭，禁write_stdin；需要fresh连接。工具实际可用，不沿用旧审批阻塞、不称底层代码已修复。证据视频隔离教师完成轮会话关闭记录.json；无需猜密码或反复重试已关闭ID。

用户要求持续自主优化：本轮新增CPU固定候选诊断只读旧Oracle学习三步路径，不重跑GPU/不使用Oracle真标签路径作候选。原F+三旧学习步，同候选池：旧μ改1281/害39.58%/MSE净降5.41%；OOF μ改658回退623/害34.80%/净降4.00%；fit均值负对照净降1.76%。原数组/统计独立重建过，来源是旧残差生成路径，不能当新教师驱动向量求解能力或DEV结果。说明μ质量好不自动带来更大修改收益，要分开检验生成与接受。

本轮新推导：Q=2(p_F−μ)d+d²，Q−Q_hat=2(μ_hat−μ)d；只有真|μ_hat−μ|≤ε时，Q≤Q_hat+2ε|d|。输出自由标量代理最小d*=-sign(r_hat)(|r_hat|−ε)+，有限输出区间截断；接受须反残差方向且0<|d|<2(|r_hat|−ε)。10000合成样本/401网格数值见证通过，无数据/模型读取；不是非线性向量全局解或ε条件覆盖保证。真实观测误差含噪声，蒸馏误差不含教师偏差，禁混称真界。

新标签路径发现：把其他教师T_j的μ混作学生S_k训练目标时，T_j可能用过学生外折V_k标签，形成间接路径。后续按折匹配同教师T_k的fit/INNER标量，不转向量坐标；若OUTER标签用于校准，必须另分视频评估，不能同标签再称独立验证。既有A特征全TRAINfit边界依然存在。

下一优先新短OOFμ效用诊断：work/oof_teacher_utility_diagnostic_design_v1.json只是设计，尚无执行源冻结/部署/GPU预检/正式执行；教师OOF效用短实验执行状态.md。固定C2best37、原β65294.20088076077/尺度/三步.25/信赖域.25，rho=p0−mu_OOF无真y求解或候选选择，不更新参数、不读DEV/TEST，旧学习分支用原数组不重跑。先fresh C UUID/空compute/源资产/空间、源码角色预算冻结/原控制重放/换标签0/二阶HVP含长度1/无参数梯度更新/实际峰值时间与capture子根覆盖，才执行最大45min/峰值1GiB/新数据1GiB/至少2h保存余量。完成立即D/异节点CPU/原数组独立审核。随后根据新生成路径及OOF结果冻结折匹配学生、校准/评估角色和同容量/初始化/订单/共享期/预算三100控制，不能填卡跳依赖。

准确协议冻结：train_group_teacher_v1.py SHA7c94f90ff648909e0abe1a6ccdb7a88543650142c6633ebadb8a0062068b8791，runtime_v2 SHA6f9074453ed15c32537efb7803210801f6123cc1647c6f4fe4d7126e2616dc9e，plan_v3 SHAf0878166da09433c27119a08fdfd8682f9b9e29f2bfe0a758f29eee45cf131a4，formalplan_v4 SHA9d06176ababb06bc0c97daf4797b20a535d30f40cd8eb9e9e20fdabb81367bbe，统一非统计init3813506519a54e1a7baa54ee37446b00e6b2f71f3a234621da1328db270a9567。旧v1未用pooler无梯度exit1保留，v2仅去pooler/其他张量和原预测0边界保留。

九个91815/91816/91817 A/B/C2旧完成100/full/20诊断/D/CPU与原TRAIN Oracle完整保留，禁重跑重传。旧流只训练520506供体/匹配头，骨干/流/reader/decoder冻结，禁冒全模型。C2 DEV未优于对照；旧OracleTRAIN学习净降5.33%/真实标签65.11%只是标签已知有限步可达性，非严格上界/泛化/语义真值。原C v1仅10失败，旧v5C81/临时31、旧14:30捕获缺口不可回填，新C不能冒恢复旧C最终。旧84/v2/v3/v4/v5A/B不重复。

唯一新A REDACTED_SERVER_HOST.invalid:53449 UUID GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7；B REDACTED_SERVER_HOST.invalid:53416 UUID GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067；C REDACTED_SERVER_HOST.invalid:53458 UUID GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa。旧址/N/R永禁连。freshSSH/SFTP真实password后只本聊天人类三P4原凭据，禁密码文件/命令/自动提示/猜测，结束exit/bye核实际结果。

24h首次Oct5UTC12:08:17只估Oct6UTC12:08:17/BJ20:08:17，平台未核；UTC08:08/10:08/11:38仍需真实强化动态保存，完成即保存，不能回填期限标签。D优先fresh查盘不删旧文件。仅官方TRAIN1281/dev229，禁TEST选结构/五seed稳定/SOTA/语义真值/subagents/设置/浏览器/续租/关机/停健康训练/改冻结源。十分钟频率，健康未变或无行动静默，仅实质结果/失败/完成/必要行动通知。全部研究/后续实验/租期保存未完不称整体完成。
'''
state.write_text(text,encoding='utf-8')
names=['outputs/研究接续状态.md','outputs/视频隔离教师100轮与OOF效用分析.md','outputs/教师OOF效用短实验执行状态.md','outputs/视频隔离教师三完整模型异节点CPU原回执独立审核.json','outputs/视频隔离教师完整模型CPU与完成快照联合保存核验.json','outputs/视频隔离教师合并OOF独立CPU新快照核验.json','outputs/视频隔离教师1281行OOF独立数组审核.json','outputs/视频隔离教师对既有有限候选效用判断CPU诊断.json','outputs/视频隔离教师有限候选选择CPU独立核验.json','outputs/视频隔离教师完成轮会话关闭记录.json','outputs/视频隔离教师完成阶段旧源与旧保存不变核验.json','outputs/视频隔离教师三100完成快照独立核验.json','outputs/视频隔离教师到学生的标签路径与校准修订设计.md','outputs/原消息相对保守接受的输出尺度解析核验.md','outputs/原消息相对保守接受输出解析核验.json','work/oof_teacher_utility_diagnostic_design_v1.json','work/analyze_oof_teacher_fixed_candidate_selection_v1.py','work/audit_oof_teacher_fixed_candidate_selector_v1.py','work/analyze_group_teacher_oof_v1.py','work/audit_group_teacher_oof_arrays_v1.py','work/audit_group_teacher_oof_CPU_snapshot_v1.py','work/audit_group_teacher_completed_preservation_snapshot_v1.py','work/audit_group_teacher_cpu_receipts_local_v1.py','work/verify_conservative_output_prox_v1.py']
files={}
for name in names:
 source=base/name;out=dest/name;out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,out);assert sha(source)==sha(out);files[name]={'bytes':source.stat().st_size,'sha256':sha(source)}
oofdir=parent/'group_teacher_oof_20261006T0308Z'
for name in ['analyze_group_teacher_oof_v1.py','audit_group_teacher_oof_arrays_v1.py']:
 source=base/'work'/name;out=oofdir/name;assert not out.exists();shutil.copy2(source,out)
report={'status':'THREE_TEACHER100_D_FULL_CPU_ORIGINAL_RECEIPTS_SCALAR_OOF_CPU_NEW_RESEARCH_AND_SHORT_STATE_PERMANENTLY_VERIFIED_OVERALL_RESEARCH_CONTINUES','utc':now.isoformat(),'directory':str(dest),'files':files,'short_state_sha256':sha(state),'D_free_bytes':shutil.disk_usage(parent).free,'overall_research_complete':False}
(dest/'preservation_record.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');(base/'outputs/视频隔离教师100完成与新研究永久保存记录.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps({'directory':str(dest),'files':len(files),'D_free_bytes':report['D_free_bytes']}))
