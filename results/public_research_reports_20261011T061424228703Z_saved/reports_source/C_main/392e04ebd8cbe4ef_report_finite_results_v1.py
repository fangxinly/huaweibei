from pathlib import Path
import json,datetime,numpy as np
root=Path('D:/CodexBackups/selective_flow_20261003_1105/finite_risk_completed_20261005');cache=np.load('D:/CodexBackups/selective_flow_20261003_1105/soft_teacher_cache_20261005T1246Z/teacher_cache_v1/dev_cache.npz');rho=cache['reference_prediction']-cache['y'];rows={}
for node in ['a','b']:
 with np.load(root/node/'predictions.npz') as z:
  assert np.array_equal(z['valid_y'],cache['y']);r=z['estimated_residual'];rows[node]=dict(residual_mae=float(np.mean(abs(r-rho))),residual_mse=float(np.mean((r-rho)**2)),zero_residual_baseline_mse=float(np.mean(rho**2)),residual_sign_agreement=float(np.mean((r>=0)==(rho>=0))),residual_mean=float(r.mean()),true_dev_residual_positive_fraction=float(np.mean(rho>=0)),weights_mean=z['actual_weights'].mean(0).tolist())
Path('outputs/有限任务风险两完成臂DEV残差冻结数组分析.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,scope='DEV diagnostic of already selected checkpoints only. No training target or new structure choice. A40/B11 head learning duration differs. Not calibrated true semantic utility; actual channel interventions pending.'),ensure_ascii=False,indent=2),encoding='utf-8')
report='''# 有限任务风险实验分析

截至真实UTC{utc}，seed91817有限风险v1的固定反馈A、有限风险标量门B各完成100轮；C向量臂在第11轮出现非有限供体梯度自然失败，只完成10轮。这个结果不是完整三臂成功实验，也不是端到端训练：教师骨干、流、关系读出、decoder参数冻结，只训练供体与匹配残差头520506参数。

|完成臂|100轮最小DEV选择|DEV229 MAE|DEV229 MSE|非零二分类准确率|
|---|---:|---:|---:|---:|
|A fixed|40|0.59844172|0.67633808|0.86111111|
|B finite_scalar|11|0.59862399|0.67539716|0.86574074|

选择按作者两批128/101 MSE均值的100轮最早严格最小，指标按官方DEV229数组另算。B相对A平均平方风险变化约−0.000940885，逐样本较好比例0.50655；MAE略高。单seed、已用于选模的DEV、选中不同epoch及C失败都限制结论；不能宣称稳定提升/SOTA或共享、补充、干扰真值。剩余20条件干预诊断还未执行，不能把预测代理效用当实际通道收益。

两新完整模型各746207712bytes、461张量，统一在新A组装。严格整模型磁盘重载与原输入DEV229重放误差A4.77e-7/B1.44e-6，400旧非供体张量完全相等；永久D finite_risk_completed_20261005/a,b全SHA/ZIPCRC/100最小选模与原数组独立核验通过。A到B、B到C七required加6extra CPU保存三者角色清楚，两目标均异训练且异组装新A；两原CPU回执已下载独立审核，不混旧91815/91816。源、TRAIN尺度、初始化、100订单、共同10fixed、残差头标签fit1153/holdout128与主辅梯度隔离都在正式前冻结。教师全TRAIN拟合/DEV选模，head holdout不是全流程crossfit。

C原失败UTC14:56:29 exit1，安全断言在非有限梯度进入更新前触发；无法从不完整日志断言原失败的准确batch或整个epoch11没有任何更新。只读从shared10固定权重、不做optimizer更新，在epoch11订单batch13复现，原失败/异常栈/数组永久D finite_risk_preflight_20261005T1434Z/c。DivBackward0的二阶异常追溯冻结关系读出centered cosine_similarity；原源码保留。真实TRAIN行620只有1有效时间步，3模态×8slots中心化范数均零。

单有效token时注意力只能取该token，因此每模态8slots相同、centered=0，几何余弦为常数0、q=sigmoid(−0.3/0.2)。新finite_single_token_reader_v1.py只对这个情形直接构造0，其他行仍原余弦；参数与旧科学源码不改。该解析分支避免构造零范数double backward，而不是降低β、改TRAIN标签或截断整个控制器梯度。共享10权重上的1281TRAIN前向与旧实现最大误差0、41梯度批次及失败witness均有限；另200TRAIN连续更新预检完成且原回执独立审核。这个fresh Adam pilot不是旧optimizer恢复或新100轮。

后续依赖为初始化/20更新/100订单/400warmup完全对应、全模型原输入含行620重放、二阶/主辅隔离与预算/新源修订冻结。两项最新检查以原完成回执为准，尚不把候选预检冒称正式C100或整体研究完成。任何正式修正版用独立源版本与新根，原C v1失败、A/B完成协议保持。只TRAIN/dev，不访问TEST挑结构。

最新核心证据：有限任务风险两完整权重本地核验.json；有限任务风险两臂七文件异训练与组装CPU保存核验.json；单token解析分支TRAIN机制独立核验.json；有限任务风险两完成臂DEV残差冻结数组分析.json；work/new_p4_checks/finite_snapshot6_current_independent_audit.json。旧C最终恢复和旧14:30保存缺口仍保留，不由新资源回填。
'''.format(utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
Path('outputs/有限任务风险实验分析.md').write_text(report,encoding='utf-8');print(json.dumps(rows))
