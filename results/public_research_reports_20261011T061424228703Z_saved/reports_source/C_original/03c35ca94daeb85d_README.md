# 选择性共享流：研究原型与实验

第四版入口：[流内重叠与循环协议](v4_design/流内重叠与循环实验协议.md)、[模型结构图](figures/v4_flow_overlap_cycle.png)、`train_v4_overlap.py`、[第四轮结果](v4_overlap_run/结果汇总.md)。三个完整模态流经共同参照软匹配，读取三路共有、三种两路共有和三种单模态特有信息；中段与末端关系反馈更新同一组流。

第三版入口：[不确定性循环流协议](v3_design/不确定性循环流实验协议.md)、[结构图](figures/v3_uncertainty_cyclic_flow.png)、`train_v3_uncertainty.py`。完整表示直接经条件流分配为三组坐标，S/R关系与预测方差反馈到第二次分解；使用配对的普通循环对照检验不确定性路由。结果位于 [第三轮结果汇总](v3_uncertainty_run/结果汇总.md)。

第二版入口：[方法与实验协议](v2方法与协议.md)、[结果汇总](v2/结果汇总.md)、`train_v2.py`。第二版保留时序，先冻结编码器，再用可逆坐标变换学习分支分配；验证集选择与测试评估由两条命令分别执行。完整指标、预测和检查点位于 `v2` 子文件夹。下文记录首版 `train.py` 的设置，首版结果单独保留。

这是一个用于检验研究假设的小规模实验：完整文本、语音和视觉特征先形成共享、特有、其余信息三组坐标，**仅共享坐标**经两步 rectified-flow 风格的速度场传输至三模态共识。特有坐标提供预测修正；其余信息坐标只参加重建与弱对抗约束，不直接参与预测。

当前实现是探索性原型，**不是** CaReFlow、FUSE-Net 或 TSD 的复现，也尚未证明三个坐标具有严格可识别的语义。当前分解层是线性投影，尚非可逆流分解；使用流的是共享分支的对齐步骤。

## 已实现的对照

| 名称 | 输入与结构 |
| --- | --- |
| `text_only` | 仅文本；两层 MLP |
| `concat` | 三模态拼接；同样的 MLP |
| `factorized` | 共享／特有／其余信息分支，无流 |
| `selective_flow` | 与 `factorized` 相同，但仅共享分支加速度场传输 |
| `selective_flow_decor` | 在 `selective_flow` 上加入噪声坐标与标签的批内线性相关惩罚；事后诊断实验 |

`factorized` 与 `selective_flow` 的区别仅在是否使用共享分支的流及其训练损失，因而是本轮最直接的对照。模型用验证集 MAE 早停，测试集只在模型选择结束后评价。记录 MAE、二分类准确率、相关系数，以及共享支单独预测的 MAE。附加诊断包括：标准化音频或视觉特征加入幅度为 0.5 的高斯扰动，观察 MAE；在固定表征上训练线性探针，测量各分支仍含有多少标签信息。

第一轮结果揭示噪声支的标签泄漏后，额外设计了 `selective_flow_decor`。它对每个噪声坐标与标准化标签的批内相关系数平方取平均，再乘固定权重 5.0 加入训练损失。该版本是在**看过第一轮测试结果后**提出的探索性诊断，不能作为独立确认实验。完整数字与逐种子记录见 [实验记录](实验记录.md)、[首轮 JSON](results_v1.json) 和 [去相关 JSON](results_decor.json)。

## 数据

使用 [MMSA](https://github.com/thuiar/MMSA) 发布的 CMU-MOSI `aligned_50.pkl` 特征。下载镜像：[Hugging Face 数据文件](https://huggingface.co/datasets/tamb2203579/CMU-MOSI/blob/main/Processed/aligned_50.pkl)。原文件的 SHA-256 必须是：

```text
d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9
```

`preprocess_mosi.py` 在确认哈希后才反序列化 pickle。它按官方 train/valid/test 划分，用文本 attention mask 及语音、视觉非零帧掩码计算每段的均值和标准差，生成约 13 MB 的 `mosi_pooled.npz`。这种汇聚保留了三个模态，但丢失时间顺序；首轮结论只适用于该简化表征。原数据与压缩数据不随代码打包。

## 运行

需要 Python、NumPy、PyTorch。先预处理，再训练：

```bash
python preprocess_mosi.py /path/to/aligned_50.pkl /path/to/mosi_pooled.npz
python train.py --data /path/to/mosi_pooled.npz --out results.json
python train.py --data /path/to/mosi_pooled.npz --models selective_flow_decor --out results_decor.json
```

远程实验目录为 `/data/coding/selective_flow`。默认使用随机种子 0、1、2，每个模型最多训练 100 轮，验证集 MAE 连续 15 轮未改善时早停。

## 结果解释边界

- CMU-MOSI 是情感强度任务，不等于离散情绪识别；后续需在 IEMOCAP、MELD 等数据上验证。
- 三分支名称是训练目标赋予的**操作性定义**，不能从网络结构本身推断其已正确分离信息。
- 去相关约束仅压低线性可读的标签信息；非线性探针或分布偏移下的信息泄漏仍有可能。
- 高斯扰动仅是特征空间压力测试，不代表真实麦克风或摄像头噪声；线性探针只能检查线性可读的标签信息。
- `without_private_mae` 是查看共享支单独输出的诊断值，不能等同于重新训练一个没有特有支的模型。
- 本轮对照是同一数据、同一划分上的原型消融，不能与论文表格中使用不同特征、编码器或训练流程的数字直接比较。
- 若 `selective_flow` 没有稳定优于 `factorized`，应先分析表征和约束，再决定是否开发可逆流分解与更精细的模态交互。

## 相关工作

- [CaReFlow：整体模态分布传输与循环信息保持](https://arxiv.org/html/2602.19140)
- [FUSE-Net：共享、特有、噪声分解](https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_Factorize_Reconstruct_Enhance_A_Unified_Framework_for_Multimodal_Sentiment_Analysis_CVPR_2026_paper.pdf)
- [TSD：全局共享、双模态共享和特有空间](https://arxiv.org/abs/2602.19585)
