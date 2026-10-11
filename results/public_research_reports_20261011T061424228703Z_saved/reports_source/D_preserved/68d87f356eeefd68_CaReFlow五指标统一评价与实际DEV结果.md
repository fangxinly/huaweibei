CaReFlow五指标统一评价与实际DEV结果

实际本地CPU计算UTC 2026-10-06T18:38:19.047663+00:00。既有官方DEV229，13个真值0；Acc2/F1用216非零行。预固定旧A/B/C2的default预测与CaReFlow seed128缓存，未更新模型、读取TEST或挑seed/读出。原数组及原回执SHA、作者Acc7函数、独立F1与Pearson核验全部通过；原10成员ZIP真实D保存，非新GPU或异节点CPU。

| 既有DEV229 | Acc7↑ | Acc2↑ | F1↑ | MAE↓ | Corr↑ |
|---|---:|---:|---:|---:|---:|
| CaReFlow复现seed128 | 45.8515% | 84.7222% | 84.5969% | 0.64678940 | 0.84272600 |
| 旧A | 48.0349% | 86.1111% | 86.0670% | 0.59844172 | 0.86721719 |
| 旧B | 48.9083% | 86.5741% | 86.5192% | 0.59862397 | 0.86731029 |
| 旧C2 | 48.0349% | 86.1111% | 86.0670% | 0.59932545 | 0.86699381 |

三旧臂分别对该预固定CaReFlow缓存DEV五项全部更好；这是已选且反复探索DEV的描述性观察，不是相同seed/训练容量预算的公平因果比较、跨seed稳定性、论文TEST超过或新完整流100成绩。A的MAE最低；B其余四项较高；不据此拼接成不存在的单模型结果。新完整流使用FIT695/30与INNER153/4，不能把它的INNER选模数与这229行DEV或论文Test横比。

评价口径与作者发布代码一致：Acc7裁剪两数组到[-3,3]后NumPy取整（半整数到偶数）；Acc2/F1排除真值恰0，预测≥0为正，F1按真实类别支持数加权；MAE/Corr用全部原始未裁剪回归值，Corr是Pearson。额外Has0指标、混淆矩阵与未定义处理见JSON。用float64聚合，原float32微小舍入差单列，不算模型变化。来源：https://github.com/TmacMai/CaReFlow/blob/main/train_reflow_new.py 。

作者论文Table1仅外部目标参照：MOSI Acc7/Acc2/F1/MAE/Corr=50.6%/89.8%/89.7%/.616/.858；MOSEI=55.7%/87.9%/88.0%/.504/.799。来源：https://arxiv.org/html/2602.19140v1#S4.T1 。DEV与Test不同不能直接判胜。

人类目标是同一模型、同一合格评价协议五项同时超过CaReFlow：四项更高、MAE更低，按未舍入数判定；不平均掉失败项，不为各指标单独选checkpoint/阈值。当前冻结100源及INNER视频等权MSE earliest-strict-min选模保持。结束完整D/异节点CPU/严格重放保存通过后，给该唯一best补五项INNER选择用途指标；更后续development/head角色协议先冻，五项统一报告。正式论文比较另需同官方数据划分/标签尺度/训练与选模协议，锁定方案后一次最终Test评价；禁TEST挑结构。尚未实施新fullTRAIN/matched消融/MOSEI/finalTest。

原D证据：D:\CodexBackups\selective_flow_20261003_1105\five_metric_DEV_actual_20261006T183811Z；frozen plan SHA 096304fce68eab40f33e09da6214170f3a8934e025c777852fd835774bcc20fa。
