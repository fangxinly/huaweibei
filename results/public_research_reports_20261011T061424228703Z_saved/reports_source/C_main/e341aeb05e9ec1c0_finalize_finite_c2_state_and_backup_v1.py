from pathlib import Path
import json,hashlib,datetime,shutil
w=Path(__file__).parent;root=w.parent;o=root/'outputs';now=datetime.datetime.now(datetime.timezone.utc)
proof=w/'new_p4_checks/finite_c2_session_closures_20261005T1636Z.json';c=json.loads(proof.read_text(encoding='utf-8'));assert len(c['closures'])==6 and all(x['exit_code']==0 for x in c['closures'].values())
state=o/'研究接续状态.md';previous=o/'研究接续状态_C2启动至完成中间_20261005T1636Z.md'
if not previous.exists():previous.write_bytes(state.read_bytes())
body=f'''# 多模态情感流研究短接续

最新真实UTC{now.isoformat()}。所有已启动正式训练/20诊断/组装/CPU验证均自然退出；当前无健康正式GPU训练在跑。研究、后续依赖与24h动态保存继续，不是整体完成。91815/91816六100完整weights/诊断/CPU与有限风险v1 A/B100已全部完成保留不重跑重传。原C v1 child2444仅10，第11轮非有限供体梯度自然exit1 UTC14:56:29；旧源/失败保留，不冒100/live或恢复原optimizer。

唯一授权新A REDACTED_SERVER_HOST.invalid:53449 GPU-8e10cc68-ef5e-9064-64ba-fec82c5089c7；B REDACTED_SERVER_HOST.invalid:53416 GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067；C REDACTED_SERVER_HOST.invalid:53458 GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa。旧地址/N/R永禁连。六当前SSH8920/7872/73602及SFTP13721/83343/75914已明确exit/bye全部实际exit0，关闭证明work/new_p4_checks/finite_c2_session_closures_20261005T1636Z.json；此前旧ID也已关闭，禁止write_stdin。后续freshSSH/SFTP只真实password后用本聊天人类新三P4原凭据，禁密码文件/命令/自动提示/猜测。

C2解析单有效token cosine=0其余原计算，旧参数/源不改。1281TRAIN前向0/41批梯度/失败witness/200freshAdam/init/after20/100orders/400fixed共享期张量0/原输入含620重放/换标签0/专门double有限差误差6.8323e-11/HVP/主辅隔离均正式前原回执独立通过。C2只重跑失败向量臂，seed91817前10fixed后90finite_vector，修订finite_formal_plan_v2 SHAaebbe5ec854f65ddc2029a2aba2a436fa088a8ef17fe5ac158c39f86f91122c4。正式root/data/coding/finite_task_risk_c2_deployment_20261005T1600Z；实际UTC16:02:12启动、16:19:38自然exit0，wrapper3571/child3572已退休，100最小DEV选中37。A/B原best40/11不重训，不能称原三臂全成功。

C2新完整模型746207712bytes/461张量统一新A组装，400旧非供体张量相等，严格整模型磁盘重载/官方DEV229原输入误差5.96e-7，整SHA14dc4ed9a3814f38c0441e6946b454d534a16be0b367e0680262adfc54a469d1。永久D:/CodexBackups/selective_flow_20261003_1105/finite_c2_completed_20261005/c，整SHA/ZIPCRC/选模/源码/原数组本地独立通过；七required加12extra保存新B，B独立TorchCPU全461有限/对应selectedaddon/未初始化CUDA通过，原CPU回执已下载审核。训练C/组装A/CPU目标B各异，根/data/coding/finite_c2_assembly_20261005T1615Z与finite_c2_preservation_20261005T1620Z。不能拿addon或large_file_manifest当新full下载。Windows本地审计v2默认gbk/v3引号修订失败保留，v4明确UTF8才通过。

A/B和C2三组20主条件+7raw校准冻结DEV诊断都自然exit0，default/换标签/上下文重放0，参数SHA不变。A/B首v1/v2仅float32风险恒等式误差越2e-6被拒收，失败保留；v3相同预测float64更严1e-12核验才通过，C2v4同精度/新runtime。20条件关断最终消息后不重新优化，换供体单方向表示循环移位；raw校准和最终传输收益估计对象不同。独立229数组/源/receipt已通过，报告有限任务风险两对照及C2三组20条件诊断与100完成独立核验.json。

DEV229 A/B/C2 MAE.59844172/.59862399/.59932536，MSE.67633808/.67539716/.67760897。C2未优于A/B；其相对自身raw平均风险降低.001686但raw本身更差，不冒全无效或稳定提升。代理下降B98.25%/C2100%，实际下降46.29%/50.22%，六方向最终收益符号一致C243.67%–48.47%；ρ预测符号A/B/C2约.515/.520/.511。C2信赖域max.09767<.25，边界0，不归因信赖域饱和。有限任务风险实验分析.md、有限任务风险C2修订实验分析.md、有限任务风险控制代理与实际风险及信赖域分析.json。

下一优先按 按视频分组TRAIN内教师隔离设计.md 实现真正外折未见标签的任务标量教师候选：TRAIN行/视频实际映射52组1281、标签无关三外折hold433/424/424及其他848/857/857已CPU只读原材料/独立审核，未拟合教师或有OOF预测。映射根/data/coding/train_group_mapping_20261005T1628Z及source1628根，原receipt/数组已capture14。先内层fit/select视频隔离、fit-only归一化、公共预训练+随机新任务初始化（不能全TRAINfit A/C2 checkpoint）、100上限/订单/原输入标签与梯度/峰值显存/实测时间/三完整weights保存预算全部冻结验证，再启动有依据跨折训练。不为填卡跳依赖或重训旧结果。μ_oof仅转任务输出，不跨折向量坐标；现有A特征仍全TRAINfit，不能冒称最终全流程crossfit。

骨干/流/reader/decoder参数冻结，仅520506供体与匹配头；原head1153fit/128holdout非全流程crossfit，教师全TRAINfit/DEVselected。β65294.20088076077、residualRMS.22068938092649817仅TRAIN冻结。平方有限风险2ρδ+δ²，准确δ不解决未知ρ。仅官方TRAIN1281/dev229，禁TEST选结构/提前五seed稳定/SOTA/共享补充干扰语义真值。

公共root/data/coding/soft_vector_research_20261005T1220Z；原科学有限风险v1 1450根、旧Jacobian1327根保留。三节点当前capture_soft_vector_v14.py SHA486879478e5cc07e83a9f259f630bc2c3ebb6aeab06952e36fc55f28ffc0c864已真实核对，覆盖全部既有新源/CPU/失败/二阶/20诊断/C2正式与组装/新CPU/视频映射。参数--root公共根 --deployment旧Jacobian1327根 --stamp真实clock唯一ID，等CAPTURE_COMPLETE且实际exit0后先receipt后ZIP。最新真实UTC16:30:53–16:31:19三snapshot永久D finite_c2_completed_snapshots_20261005T1627Z，目录1627只是ID。ZIP全SHA/CRC/成员唯一、旧v1完成/失败、新C2实际100、7+12CPU原回执/成员及全部20数组分别独立通过。大weights仅freshSHA引用，不重传旧weights；扩展后续新根必须新capture源版本。此前capture9缺extra/v10重复成员拒收仍保留，v11/12已过不冒旧包最新。

24h首实际Oct5UTC12:08:17仅估Oct6UTC12:08:17/BJ20:08:17，平台期限未核。Oct6UTC08:08/10:08/11:38附近须强化实际动态保存，完成即D及独立CPU。逐次freshC/D及节点空间，D优先禁删。禁subagents/设置/浏览器/续租/关机/停止健康训练/改旧冻结源。10分钟healthy未变或无行动静默，仅实质结果/失败/完成/必要行动通知，每轮真实工作更新短状态与证据。

旧84/v2/v3/v4全完成不重复。旧v5 A/B100、B独立A原回执已取，A独立C只有终端通过原回执未取；旧C最后真14:10只81，31临时full不冒100，新C空卷没恢复最终。旧13:40/14:10保存完成，旧14:30真实capture缺失永不回填。历史见研究接续状态_C2启动至完成中间_20261005T1636Z.md与前历史。全部研究/新教师候选/租期保存未完。
'''
state.write_text(body,encoding='utf-8')
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('finite_c2_records_'+now.strftime('%Y%m%dT%H%M%SZ'));dest.mkdir(exist_ok=False);paths=set()
for pat in ['*finite*','*single_token*','*capture_soft_v1[234]*','capture_soft_vector_v1[234].py','audit_soft_snapshot_v1[234].py','*train_group*']:
 for p in w.glob(pat):
  if p.is_file() and p.suffix in ['.py','.json','.npy']:paths.add(p)
for p in o.iterdir():
 if p.is_file() and (any(k in p.name for k in ['有限任务风险','单token解析分支','TRAIN视频','按视频分组']) or p.name=='研究接续状态.md'):paths.add(p)
for p in (w/'new_p4_checks').glob('finite_c2*'):paths.add(p)
paths.add(proof)
rows={};sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for p in sorted(paths):
 rel=p.relative_to(root);q=dest/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q);assert sha(p)==sha(q);rows[rel.as_posix()]=dict(bytes=p.stat().st_size,sha256=sha(p))
r=dict(status='FINITE_C2_ALL_NEW_RECORDS_REPORTS_CURRENT_SHORT_STATE_AND_SIX_REAL_CLOSURES_PERMANENT_D_SHA_VERIFIED',utc=now.isoformat(),directory=str(dest),files=rows,scope='C2 actual100/full/CPU/20diagnostics complete; originalC1failure retained. NewTRAINgroup mapping prerequisite only. Research and rental future captures continue.')
receipt=o/'有限任务风险C2研究资料及接续永久保存核验.json';receipt.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(receipt,dest/'backup_receipt.json');print(json.dumps({'directory':str(dest),'files':len(rows)}))
