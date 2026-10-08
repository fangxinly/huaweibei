# MSE 单变量完整训练与评分接续

真实100轮/4000更新已于2026-10-08 UTC19:31:58自然0。仅Huber主任务损失改MSE；结构、初始化、学习率、context惩罚及共同订单保持。官方TRAIN1281、VAL229、TEST685，seed128；按VAL128/101两批MSE均值严格最小且最早选中第65轮，选模值0.70459340474218。选中模型SHA b1af7ce9de4e53d6c06d0cf2d8381227491f0fa59ebbf94003509c9057aca61e。

完整checkpoint 2960812651字节，SHA67326cc52a0e90c677afc96393dbbdff86f72b8078f85c97ca1b271f6b91ccf2。完整归档3543666247字节，SHA27c9dd730f8d774ecd5163a11bdb377ef6676a0f6f8c7a9a6923262379a053b1，170个原始成员SHA/CRC/唯一性通过；D盘完整副本通过，GitHub Release三分片均实际digest通过。没有删除旧冻结原件。

[原件Release](https://github.com/fangxinly/huaweibei/releases/tag/autonomous-flow-health-20261008)。B首次公开HTTP请求在下载0字节时RemoteDisconnected，child1570自然1，完整失败原件保留；未开始CPU审计，不是GPU或权限故障。恢复方案是将已验证D原件SFTP传B，在新根重新执行相同完整CPU审计；来源明确为D-SFTP，不能声称公开下载成功。

真实TRAIN日志全SHA核过：40步消息任务梯度极小，400步已明显更新；2600步消息任务梯度RMS约1.707e-5，扣rounded衰减更新约2.413e-5。不能称永久惰性。第65轮到100轮TRAIN目标从0.01988下降到0.01470，VAL选模MSE从0.70459升到0.72004，有过拟合迹象；TRAIN是更新过程目标含context惩罚，不能当端点TRAIN纯MSE或建立因果。逐批梯度也不能证明消息提供了有用新信息。

当前尚无本次最终VAL/TEST五项结果。完整B审计自然0后，只评价第65轮的官方VAL与TEST消息ON/OFF；预测原件先保存，然后一次评分、B独立重算。旧VAL MAE0.597256537、TEST0.650616015，CaReFlow TEST0.619535294是历史结果；不将VAL选模MSE当MAE，也不以TEST改结构。TEST有历史访问，不声称新盲测。
