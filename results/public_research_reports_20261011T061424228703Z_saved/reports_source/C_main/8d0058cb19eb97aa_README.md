# best20 / best36 逐样本复核数据

这四份 CSV 来自已保存预测。每个模型 FIT1494条/64视频，INNER264条/11视频；FIT 与 INNER 按视频隔离。新旧行ID、顺序与真标签完全一致，每一浮点数和ID已逐行回读核验。

- y：真实情绪标签，原连续尺度。
- b：原始特征池化后经共享 decoder 的预测；它本身也使用多模态特征，**不是消息全关 p0**。
- f：两步流轨迹共享 decoder 读出，再加 role head。
- p：b + tanh(gain) × (f − b)。best20 gain≈−0.795142945；best36≈−0.805566777。它不是凸组合。
- p0 / p1：仅 best36 提供。同一个固定模型、同一流轨迹/reader/role head/decoder/gain，六 donor feedback 输出全关/全开；p1逐值等于p。p0仍有其它跨模态路径。
- row_id：原片段ID；video：ID中第一个 `[` 前的视频ID。
- 弱/强分组：|y|≤1 / |y|>1，只用于描述与FIT训练；推理不使用真标签。

这些是全 TRAIN/DEV/TEST 合并后的第0折 FIT/INNER。官方 TEST 的部分行已进入训练，**不能把这些分数称为官方保留TEST、全新独立确认、完整五折或正式超越CaReFlow**。两检查点均由这份 INNER 选出，INNER 是探索性评价。

selected state SHA256：
- best20：35fa4c4eae77b1f505f5c717669d533e05a6810fde5cbdb39fffd36b8450209e
- best36：ad9c35a4b7900dbd8806f11a2466bbdc771211f52c44d09729b725fd10245a24

校准仅用FIT标签拟合 α、β；INNER只评价。主分析片段等权；视频等权只是固定敏感性分析，均报告，不据分数挑读出。bootstrap按11个完整视频成对重采样10000次，seed20261008，条件于固定检查点与拟合参数，不重拟合或重选epoch。

包内 JSON 保存所有未舍入五项、MSE、弱/强分组、参数、结构相关性和配对区间。source_refs保留原件路径与SHA；合成测试只验证计算，不代表真实任务效果。报告见REPORT.md。
