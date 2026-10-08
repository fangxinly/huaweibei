# 原流 MSE 单变量训练与真实健康诊断

本轮只将原流主任务损失从 Huber 改为 MSE，保留结构、初始化、其他惩罚、参数分组学习率、官方 TRAIN1281/VAL229/TEST685、seed128、100轮和4000更新及原训练订单。目的在于检验真实任务梯度和更新，尚无新的最终 VAL/TEST 五项结果。原流旧 VAL MAE0.597256537/TEST0.650616015、CaReFlow TEST0.619535294保持不变。

真实16步预检 child1149自然0；首失败 child895在第一次opt.step前发生，原因是诊断程序的float32大参数抽样索引舍入越界，原失败证据保留，已使用整数索引修复。A/B真实CPU合成参数轨迹和CUDA大索引取样均通过。首学习率0保持，流fields/role/message第3步、reader第4步出现任务梯度；第16步消息任务梯度RMS1.654e-14，扣除rounded纯权重衰减后的更新RMS8.234e-12，固定32TRAIN输入的消息开关输出差0。它支持“初期消息更新非常弱”，不支持“永久没有消息作用”。

完整16步状态2220624858字节SHA `2d1e08145a80f4dec044ebbee4e7436974325dff610d3da453b115820acec4ad` 已保存到 [完整原件Release](https://github.com/fangxinly/huaweibei/releases/tag/autonomous-flow-health-20261008)。完整ZIP1180925397字节SHA `ffd24c655f5a507094f7897e2696a490a2b2fade05bc65e66e4cfd7d1c3bb0e4` 含完整模型/Adam/scheduler/RNG/订单/源码/日志。B下载新原件，全部ZIP成员SHA/CRC/唯一、341个Adam状态全16步、模型SHA、RNG和订单联合通过，CPU没有编码器模型前向。大文件不落本地满D盘，不删除旧冻结原件。

完整续训入口为 `work/autonomous_mse_prefix16_v1/resume_official_mse100_v1.py`，在模型/优化器构造、数据处理后严格恢复完整状态和随机数，从17步继续，16步计入总4000。A/B原流模块CPU16→40断点续训的预测/参数/Adam/调度器/PythonNumPyTorch RNG与不中断轨迹完全相同。实际完整编码器续训已于UTC18:30:59启动，第1轮40步完成。第400步会保留完整断点并记录任务/惩罚梯度和扣衰减更新。

官方VAL只按作者128/101两个批的MSE均值严格最小、相同时取最早epoch选模。训练完成后评价同一个选中checkpoint的VAL和TEST全部五项；不拼指标、不用TEST改结构。TEST有历史访问，不能声称新的盲测。当前100轮仍在运行，源码资格和健康诊断不是性能提升证据。
