from pathlib import Path
import datetime,json,hashlib,zipfile
r=Path(__file__).resolve().parents[1];o=r/'outputs';a=json.loads((o/'Jacobian20条件诊断与效用校准独立核验.json').read_text());sign=json.loads((o/'Jacobian残差符号与方向约束独立核验.json').read_text());now=datetime.datetime.now(datetime.timezone.utc)
rows=[]
for n,name in [('a','固定反馈'),('b','标量门'),('c','连续向量')]:
 x=a['results'][n];rows.append(f"| {name} | {x['best_epoch']} | {x['mae']:.8f} | {x['mse']:.8f} | {sign['rows'][n]['residual_sign_agreement']:.6f} |")
(o/'Jacobian方向约束三臂实验分析.md').write_text('''# Jacobian方向约束三臂实验分析

seed91816三组各100轮已完成，20冻结DEV条件/默认重放0/标签替换0/参数不变和独立数组审核通过。连续向量仍未显示超过匹配固定反馈的收益。只官方TRAIN1281/dev229探索，不访问TEST，不称多seed稳定、SOTA或语义共享/互补/干扰真值。

| 策略 | 最小DEV轮次 | 缓存DEV229 MAE | 缓存DEV229 MSE | 残差符号一致率 |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+'''

三臂共同前10fixed、后90分策略，统一520506训练参数/214506梯度头、完整初始化、100订单与共同10轮SHA。与91815是不同seed和不同估计器，不是配对seed证明，也不是重训旧实验。骨干/第一步流/后半段流及decoder冻结，六供体网络和三头训练，真实原输入全模型接入与标签隔离预检已通过。

J=∂p0/∂c在冻结模型中不请求标签采集，TRAIN a=2(p0-y)J恒等式最大误差1.86e-9，参考重放误差≤2.38e-7。归一化预测头输出投影到J/TRAIN_RMS的一维空间，各模态共享残差标量。最终预测梯度位于J空间的最大数值误差4.66e-10，故余弦只能近似±1；平均余弦约.0655–.0742反映残差符号只有.53275–.53712一致率，不能把方向约束本身当成准确任务判断。

估计残差DEV MSE约.68170/.67767/.68170，零残差基准.69225，存在少量平方误差信号但符号和通道实际效用仍弱。C六通道实际关通道收益均值为负；保留全反馈对全关平均收益约.01487。单通道与全关有限变化不能线性相加。C同weight改成fixed的MSE约.67570、默认soft约.67738，此为冻结机制诊断，不能据此改已冻结选择协议或称独立泛化。

当前完整权重已在新A组装并严格磁盘重载/原输入DEV229重放通过；永久D下载、整SHA/ZIP及第二组轮转CPU原回执还在核验中。不能把addon完成混称完整永久保存或沿用第一组回执。第一组91815三完整weights及七文件加额外资料的轮转CPU原回执全部独立审核通过并保留。

下一设计需要有限预测变化而非只有一阶点积。对参考p0和变化δ=p-p0，平方风险差精确为2(p0-y)δ+δ²；条件期望为2ρ(x)δ+δ²，其中ρ(x)=p0-E[y|x]。ρ̂能只用TRAIN拟合，δ由标签无关冻结模型计算。准确J不能消除未知ρ，有限二阶δ²也不能省略。这只是一项候选设计依赖，必须先冻结预算与匹配控制，再真实检查标签/梯度/二阶路径/初始化/订单/全模型重放后才启动，不为填卡跳依赖。

证据：Jacobian方向约束三臂正式冻结计划.json、标签无关Jacobian采集与梯度分解独立核验.json、Jacobian三臂100轮联合快照独立核验.json、Jacobian20条件诊断与效用校准独立核验.json、Jacobian残差符号与方向约束独立核验.json。正式源码/根/data/coding/jacobian_aligned_v2_deployment_20261005T1327Z，永久D jacobian_completed_20261005/{a,b,c}。
''',encoding='utf-8')
(o/'有限任务风险反馈下一步设计.md').write_text('''# 有限任务风险反馈候选：先解决依赖再实施

冻结参考p0，任意标签无关反馈产生p=p0+δ。平方损失差L(p,y)-L(p0,y)=2(p0-y)δ+δ²严格成立，不是Taylor近似。若ρ(x)=p0-E[y|x]，条件风险差恰为2ρδ+δ²。候选以TRAIN监督的共享残差ρ̂替代未知ρ，δ每次由同一冻结后半段流/decoder计算，不读取真实推理标签。估计误差为2(ρ̂-ρ)δ；因此|差|≤2|ρ̂-ρ||δ|，准确δ也不保证ρ̂准确。新J约束实验符号约53%提示不得只把问题归因于梯度方向。

同一候选中匹配三臂的初步方案：A固定反馈；B基于真实有限预测变化的标量门对照；C基于冻结decoder有限风险代理的向量更新。B单通道参考收益Û_i=2ρ̂(δ_without-δ_with)+δ_without²-δ_with²；它只是条件任务风险代理，不能命名为语义真值。C可考察信赖域内向量目标 .5||z-F||² + β[2ρ̂δ(z)+δ(z)²]，通过有限步更新反馈后续流；β/尺度/步数必须TRAIN机制与预算先冻结，不能DEV反复改已有协议。对非线性decoder目标不保证凸或全局最优，不套用此前平方hinge近端的闭式/唯一性/非扩张结论。

依赖：第二组整weights/独立CPU保存先收尾；明确共享固定坐标与残差头的TRAIN拟合范围，留出TRAIN内部诊断检查残差符号与过拟合。若引入OOF条件均值教师，要从未看该行标签的模型产生预测，分组/归一化/选模也要隔离；共享已全TRAIN拟合的A编码器不能冒称整个流程无标签泄漏或无偏crossfit。OOF教师的向量坐标不得直接当成A坐标，要只转移任务输出/标量条件均值。任何跨折教师训练要先确定TRAIN行/视频分组映射、预算和选模集合，不能直接占GPU启动。

C实现若反向穿过向量优化步，需真实二阶autograd、梯度隔离/有限差分、标签替换、零信号控制、数值信赖域、峰值显存、20更新预算与全原始输入模型重放检查。主任务不得训练ρ̂/J代理，辅助不得训练供体/冻结模型；若采用停止整个更新方向，要明确实际梯度与只优化控制的差别。三臂同容量、初始化、订单、共同warmup与官方TRAIN1281/dev229，100轮选择协议重新完整冻结。未完成这些依赖不启动正式新训练，不造填卡任务。
''',encoding='utf-8')
bad=Path('D:/CodexBackups/selective_flow_20261003_1105/jacobian_snapshots_20261005T1328Z');rejection={}
for n in 'abc':
 p=bad/n/'snapshot.zip';row={'path':str(p),'used_as_verified_evidence':False}
 if p.exists():
  row['bytes']=p.stat().st_size;row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
  try:
   with zipfile.ZipFile(p) as z:row['crc_ok']=z.testzip() is None
  except zipfile.BadZipFile:row['crc_ok']=False
 rejection[n]=row
(o/'Jacobian联合快照下载时序与拒收记录.json').write_text(json.dumps({'utc':now.isoformat(),'reason':'Initial downloads began before capture completion; initialA receipt missing and partial snapshot retained. None of initial three used as verified evidence. Waited realCAPTURE_COMPLETE, downloaded all three to new_verified directory, allSHA/member auditspassed. Atomicv3 publishes final ZIP onlyafterclosed; sourcev2 retained.','initial_downloads':rejection,'accepted_live_audit':'Jacobian三臂1328联合快照独立核验.json','accepted_complete_audit':'Jacobian三臂100轮联合快照独立核验.json'},ensure_ascii=False,indent=2),encoding='utf-8')
print('JACOBIAN_REPORT_NEXT_DESIGN_AND_REJECTION_RECORD_SAVED')
