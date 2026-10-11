# 正式 CaReFlow 比较：来源与预算本地准备

这份材料只核查源码和原协议，不是新训练、模型选择、201重评分或最终TEST授权。

既有指定baseline采用TRAIN1281/DEV229、100轮、batch32、drop_last=True，因此每轮40批、4000次更新；DEV选择是batch128的批MSE算术均值。当前新fold参考只有FIT695/INNER153、2200次更新和视频等权选择。两者不能直接称相同训练预算或选模协议；新fullTRAIN协议必须明确并统一这些规则，不能把fold模型当正式fullTRAIN模型。

原baseline训练源SHA d0266a55931fbae4329c2123b3b0f0d475799f31b8e202fa393ecef406ce841e；旧五指标归档时使用另一已固定指标源SHA d88739d1d1a224a2f159f29fa723cc24c347eadcf921e93bf57bd0e2cdaa3cb7。原runner来源与protocol pin匹配；不把两份版本冒同源码。缓存不是作者CLI全默认，也不冒论文同配方。

缓存的原runner在训练及DEV选模完成后曾执行一次TEST；本轮未打开任何预测NPZ、pickle或TEST标签。主聊天未读不抹去历史TEST访问。现有本地缓存目录没有best.pt，不能只凭protocol/预测缓存冒拥有可完整重放的baseline checkpoint。未来可核实永久原件位置；没有合格整模型就保持未通过，不因某个低DEV结果选baseline。

下一准备应固定单方法（不根据201九读出挑赢家）、共同官方角色/数据尺度/预训练资产/预算/选模规则、实际完整源与安全标签守卫。保持TEST不可访问；先做实际资源与保存门。不同架构即使同seed也不是共享全部初始权重，容量和训练成本须如实列出。当前未选择常数/U/W作为部署方法、未启动fullTRAIN/MOSEI或重新跑baseline。

第16建议turn在remote compact阶段因usage limit实际failed，只有部分commentary，无完整报告或独立源审阅。主线独立准备继续；不重复发送同批或把部分建议冒完成，不把该失败当GPU/服务器故障。

补充实际clock 2026-10-07 00:02:34 UTC：已从原保存manifest定位C的指定seed128 epoch72完整模型，fresh SHA/ZIP CRC/333唯一成员通过，原742279993字节已本地完整复制D并复核；原缓存小目录无best.pt并不代表原完整模型缺失。此为字节保存门，未Torch加载/张量或模型重放，不冒正式缓存可复用。train/dev-only守卫仅合成检查，未接入完整runner。续读正式CaReFlow原整模型与标签守卫准备接续.md/json。新目录TZ后缀为命名变换错误，实际clock字段为准。
