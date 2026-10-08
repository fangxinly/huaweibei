from pathlib import Path
import json,datetime
o=Path(__file__).parent.parent/'outputs';now=datetime.datetime.now(datetime.timezone.utc).isoformat();old=o/'有限任务风险实验分析.md';archive=o/'有限任务风险实验分析_C2完成前_20261005T1634Z.md'
if not archive.exists():archive.write_bytes(old.read_bytes())
diag=json.loads((o/'有限任务风险两对照及C2三组20条件诊断与100完成独立核验.json').read_text(encoding='utf-8'));control=json.loads((o/'有限任务风险控制代理与实际风险及信赖域分析.json').read_text(encoding='utf-8'));full=json.loads((o/'有限任务风险C2完整权重本地独立核验.json').read_text(encoding='utf-8'))
t=f'''# 有限任务风险实验分析

最新真实UTC{now}。seed91817固定A、有限风险标量门B与解析单token修订向量C2各已100轮，并完成完整模型重载/官方DEV229原输入重放、永久D及异训练/组装节点CPU保存、预定20条件冻结干预诊断。当前向量候选没有优于两条对照。原C v1只有10轮完成且第11轮自然失败，原源/日志/失败材料仍保留；不能把C2回填成原三臂协议全成功。骨干/流/reader/decoder参数冻结，仅520506供体与匹配头训练，不是端到端。

|正式完成臂|100最小DEV选中epoch|DEV229 MAE|MSE229|非零二分类准确率|
|---|---:|---:|---:|---:|
|A fixed 原v1|40|0.59844172|0.67633808|0.86111111|
|B finite_scalar 原v1|11|0.59862399|0.67539716|0.86574074|
|C2 finite_vector 修订v2|37|0.59932536|0.67760897|0.86111111|

选模规则是作者128/101两批MSE均值的100轮最早严格最小，表中MSE按229样本另算，两个加权方式不同。C2两指标均比A/B更差；单seed、已用于选模的DEV与不同选中epoch限制解释，不宣称稳定提升、SOTA或共享/补充/干扰语义真值。

C2采用同seed91817、相同初始化/after20/100订单和前10fixed共享期整SHA；只对1有效token时的centered cosine构造解析常数0，其他样本仍原计算。只读shared10原实现失败witness、1281TRAIN旧新前向0、41批供体梯度、200freshAdam、400fixed共享期张量0、原输入含行620三模式重放/换标签0、真实double方向有限差误差6.8323e-11/HVP/主辅梯度隔离全部在正式前完成原回执独立核验。这个修正解决原二阶零范数数值失败，没有更改β、截断内层主梯度或恢复原optimizer。

C2真实UTC16:02:12启动、16:19:38自然exit0，wrapper3571/child3572均退休。新完整weight746207712bytes/461张量在新A组装，严格磁盘整模型重载后官方DEV229原输入最大误差5.96e-7，400旧非供体张量相等。完整SHA {full['receipt']['full_checkpoint_sha256']}。永久D finite_c2_completed_20261005/c，七required加12extra全部全SHA/ZIP CRC核验；C训练、A组装/原输入重放、B独立CPU三者不同。B实际TorchCPU全461张量有限且供体/头对应选中addon、CUDA未初始化，原CPU回执已下载并独立审核。小addon不能替代完整weight，large-file快照引用也不等于再次下载。

## 冻结干预结果与误差分解

20主条件为默认、全关、六方向最终消息关断、六方向换供体、三策略、三接收者消息同时关断；另7次raw校准前向区分原始供体与最终控制消息。所有default重放/换标签/上下文重放0，模型参数SHA前后不变。关断发生在控制后的实际消息上，不重新优化其余通道；换供体只在官方128/101批次内循环移位单方向供体表示再无标签控制。DEV只用于冻结诊断，无新目标拟合。

A/B诊断v1/v2分别被float32风险恒等式相减误差2.265e-6/2.734e-6拒收，原日志/输出保留。v3把同一float32模型预测转float64，用更严1e-12验证代数恒等式，不改变预测/utility数组/干预或参数；C2沿用精度核验的v4源。三份原数组/receipt与源码逐SHA独立审核通过，C2恒等式误差6.245e-16。

给定参考p0与同一标签y，任意标签无关反馈p=p0+δ有精确平方风险差2(p0−y)δ+δ²。使用残差预测ρ̂时，代理与实际差为2(ρ̂−(p0−y))δ；比较控制后与raw固定策略时，误差为2(ρ̂−(p0−y))(p_control−p_raw)。独立229数组核对误差<1e-12。它说明准确计算δ仍不足以识别真实条件风险；不能把observed标签残差称未知条件ρ本身。

|臂|相对自身raw固定策略的代理风险均值变化|实际平方风险均值变化|代理下降样本比例|实际下降样本比例|残差符号准确率|
|---|---:|---:|---:|---:|---:|
'''
for n,x in control['rows'].items():t+=f"|{n.upper()}|{x['proxy_control_vs_raw_mean']:+.8f}|{x['actual_control_vs_raw_mean']:+.8f}|{x['proxy_improved_fraction']:.4f}|{x['actual_improved_fraction']:.4f}|{x['estimated_residual_sign_accuracy']:.4f}|\n"
t+='''
C2相对自身未控制raw策略平均平方风险确实降低约.001686，但其raw策略MSE.679295，本身比两个已选对照更差，最终MSE.677609仍没有超过A/B。C2所有样本代理目标下降，实际收益方向只有约50.22%%一致；不能把优化代理成功当作可靠任务效用判断。B代理下降约98.25%%，实际下降约46.29%%且平均实际风险略上升，进一步支持需要验证残差估计与收益校准。六方向最终关断收益与代理符号一致率：A45.41%%–48.03%%、B44.54%%–46.29%%、C243.67%%–48.47%%，Spearman多数为负。

C2信赖域变化最小7.11e-6、均值.006936、最大.097672；.25边界命中0，不应将失败归因于已验证的信赖域饱和。每步代理目标下降不保证非凸终端的全局最优。全关条件MAE.589626虽然低于三臂，但MSE.692251更高；单通道收益也不等于共同关断收益，不能据此给出信息共享/互补真值。

教师全TRAIN拟合/DEV选模；1153残差headfit/128headholdout不是全流程crossfit。当前训练残差RMS.220689与DEV残差绝对误差约.6提示应检查教师拟合与应用分布差异，但不能从DEV单次观测认定唯一失败原因。下一先检验隔离的TRAIN内教师目标，再决定新正式匹配实验；不为填卡启动重复训练。

## 保存与后续依赖

三节点capture14实际UTC16:30:53–16:31:19组合覆盖91815/91816/原有限风险A/B完成与C失败、C2正式源/100订单/selection/20原数组、新A完整模型及新B原CPU回执、单token二阶与52视频组映射。原子CAPTURE_COMPLETE/exit0后先receipt后ZIP下载永久D finite_c2_completed_snapshots_20261005T1627Z；目录1627仅ID，实际UTC内证。全SHA/ZIP/成员唯一、旧新协议、七加extra manifest与原回执分别独立通过。所有训练/诊断/组装/CPU验证已自然退出，没有健康任务被停止。

实际TRAIN1281条覆盖52视频；按视频标签无关分组的三外折hold433/424/424，行顺序标签与现有TRAINcache匹配，分组互斥/完整覆盖已独立审核。仅完成分组先决条件，未拟合新教师或产生OOF预测。全教师、归一化、内层选模及初始化仍需隔离核验与预算冻结；现有全TRAIN拟合的A编码器不能冒称全流程crossfit。详情下一份按视频分组TRAIN内教师隔离设计.md。

24h期限按首次实际Oct5UTC12:08:17仅估Oct6UTC12:08:17，平台未核；Oct6UTC08:08/10:08/11:38仍须强化真实动态保存。旧C最终缺失与旧14:30未捕获缺口继续保留，不回填。全部研究及未来租期保存继续，不能称整体完成。

证据：有限任务风险C2完整权重本地独立核验.json；有限任务风险C2七文件异训练与组装CPU保存独立核验.json；有限任务风险两对照及C2三组20条件诊断与100完成独立核验.json；有限任务风险控制代理与实际风险及信赖域分析.json；有限任务风险C2完整模型独立CPU及最终联合快照核验.json；TRAIN视频组映射与三外折先决核验.json。前期拒收/失败详情另存历史，不删除或改旧冻结源。
'''.replace('%%','%')
old.write_text(t,encoding='utf-8');(o/'有限任务风险C2修订实验分析.md').write_text(t,encoding='utf-8');print('FINAL_FINITE_REPORT_WRITTEN')
