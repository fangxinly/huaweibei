# 多模态流与任务残差消息控制研究

本仓库保存截至 2026-10-08 的研究源码、冻结版本、训练和评估流程、机制诊断及小规模代码检查。入口目录见下表，完整历史目录见 [CODE_INDEX.md](CODE_INDEX.md)，逐文件来源、原始及发布 SHA256、脱敏记录见 [source_manifest.json](source_manifest.json)。

## 主要代码入口

|目录|用途|实际状态|
|---|---|---|
|[官方划分原流改进](work/official_anchored_upgrade_20261008T053429Z/)|原编码器与 100 维两步流；六方向 donor-minus-zero 消息；TRAIN 训练、VAL 选模、TEST 评价|100 轮 / 4000 更新完成；官方 VAL/TEST 五项已评分，TEST 未超过 CaReFlow|
|[原流逐方向残差控制](work/residual_direction_v2_frozen_20261008T114041Z/bundle/)|同流消息 OFF / 六单方向 / ALL 八个实际干预；视频隔离残差估计、校准和效用选择|原服务器 CPU 合成代码检查通过；完整真实 OOF 训练及新 VAL/TEST 尚未完成|
|[线性校准与增量对照](work/linear_control_addendum_v1/)|无约束融合、仿射校准、正交化增量|本地合成检查通过；真实官方双方 OOF 拟合尚未完成|
|[单调非线性校准](work/flexible_official_calibration_v1/)|固定五锚点单调分段线性校准；对完整同一非线性基空间残差化增量|本地合成检查通过；没有新校准后的官方 VAL/TEST 分数|
|[极性与强度候选审阅](docs/polarity_intensity_review.md)|在原流内允许消息同时纠正极性和强度，增加辅助监督|数学与文献审阅，尚无实现冻结或训练结果|
|[归档独有源码](archive_code/)|本地 ZIP 中未被其他源码覆盖的历史版本|归档号与原包内路径的对应关系在 manifest；不把旧实验当现行方法|

`work/` 保留原本研究版本目录；`outputs/` 仅包含源码，不包含原始实验输出。源码归档没有按结果优劣删除旧方案。

## 当前官方 VAL / TEST 结果

数据角色：官方 TRAIN 1281 条、VAL 229 条、TEST 685 条。原方案以 VAL 规则选择第 30 轮；双方均用同一个各自选定检查点报告五项，不能跨检查点拼接指标。

|划分|模型|Acc7 (%)|Acc2 (%)|F1 (%)|MAE|Corr|
|---|---|---:|---:|---:|---:|---:|
|VAL|原流改进|50.2183|87.5000|87.5085|0.597256537|0.865830422|
|VAL|CaReFlow|49.7817|87.9630|87.9248|0.604523826|0.859493847|
|TEST|原流改进|46.1314|87.1756|87.1488|0.650616015|0.825181614|
|TEST|CaReFlow|50.6569|87.6336|87.6112|0.619535294|0.851188468|

**当前没有达到五项全面超过 CaReFlow。** VAL 约 0.59 的 MAE 不能写成 TEST 成绩。详细结果、消息开关与局限见 [官方结果说明](docs/official_VAL_TEST_results.md)。双方数据角色、更新订单、更新数和选模规则对齐，但模型容量及损失并不完全相同，不声称严格容量匹配。历史合并 TRAIN/DEV/TEST 训练及 INNER 诊断不作为此表的泛化结果；完整合并数据五折尚未完成。已有 TEST 结果已经查看，后续同一 TEST 的结果不能声称首次独立确认。

统一指标实现见 `sentiment_metrics_careflow_v1.py`：Acc7 对所有行 clip 后 round；Acc2/F1 排除真实值为 0 的行，预测值 >=0 视为正类，F1 使用 support-weighted；MAE/Pearson 使用原始连续值。

## 研究目标与边界

研究问题是同流跨模态消息能否提供超出一维校准的样本特异预测信息，并通过任务残差控制消息效用。对真实最终输出，平方误差改善恒等式为：

\[
U=(y-p_0)^2-(y-p_1)^2=2(y-p_0)(p_1-p_0)-(p_1-p_0)^2.
\]

`p0` 必须是同一流和同一读出的消息关闭预测。效用控制版本要求完整任务前驱的视频隔离 OOF，以及独立的 TRAIN 内校准；普通任务损失门控、灵活校准和实际消息开关仍须共同对照。当前实现检查不等于真实数据收益，不提供风险保证、PID 语义真值或时间控制结论。

## 环境与小成本代码检查

正式服务器环境为 Linux、PyTorch 2.1.0+cu121；其余实际依赖版本记录在 [requirements-research-lock.txt](requirements-research-lock.txt)。历史脚本可能有其他依赖。该锁文件记录原运行环境，不代表所有历史代码均在同一环境验证。

```bash
python -m pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements-research-lock.txt

# 合成数据上的模块契约，不是数据集性能测量
python work/official_anchored_upgrade_20261008T053429Z/official_upgrade.py
python work/residual_direction_v2_frozen_20261008T114041Z/bundle/native_contract.py
mkdir -p scratch
python work/flexible_official_calibration_v1/contract_check.py scratch/flexible_contract.json
```

训练入口为 `train_official.py`，预测入口为 `infer_official.py`，评分入口为 `score_official.py`。正式训练需要外部公共预训练资产及 MOSI 数据，且运行计划包含资产 SHA、源 SHA、ID、预算与一次执行守卫。历史协议的机器、路径和到期时间不可直接套用于新机器；重新执行应先生成完整新计划并冻结来源。不要通过删除守卫绕过来源或标签隔离。

本仓库含不涉及标签的固定训练位置订单文件，以保留原预算的顺序依据。**数据集、逐样本标签/预测、原始日志、模型权重/Adam/RNG 检查点、离线 wheel 包、凭据不在 Git 中**，这些原件仍保留于原研究备份位置。脱敏只改发布副本，原冻结证据不改；发生脱敏的历史文件及其配置不能冒为原字节完全一致的可执行冻结包。

## 参考与来源

部分兼容模块沿用 CaReFlow 作者组件与原项目接口，源码内保留原注释和来源命名；本次上传未为第三方代码重新声明许可证。极性/强度多任务及乘性交互也有既有研究，不能单独当作本项目的新创新：

- [Polarity and Intensity: the Two Aspects of Sentiment Analysis](https://aclanthology.org/W18-3306/)
- [Tensor Fusion Network for Multimodal Sentiment Analysis](https://aclanthology.org/D17-1115/)

方法限制与核对见 [残差几何审阅](docs/coupling_residual_math_review.md) 和 [非线性校准说明](docs/flexible_calibration_review.md)。
