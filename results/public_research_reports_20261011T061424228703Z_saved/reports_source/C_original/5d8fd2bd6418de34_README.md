# CaReFlow官方实现复现

作者仓库：[TmacMai/CaReFlow](https://github.com/TmacMai/CaReFlow)。已下载官方源码并锁定commit `5c9f9c7a0bb3f1202ebb3258052da2b99710c565`，源码快照在 `official` 子目录，文件哈希在 `official_manifest.json`。

本目录用于复现强基线。此前V1–V4原型保存在 `../selective_flow_prototype`，其预测结果不能作为CaReFlow复现结果。

## 设置差异

| 项目 | 此前V4 | CaReFlow作者设置 |
| --- | --- | --- |
| 文本入口 | 预提取BERT768维特征 | 微调DeBERTa-v3-base |
| 语音维度 | 5 | 74 |
| 视觉维度 | 20 | 47 |
| 训练／验证／测试样本 | 1284／229／686 | 1281／229／685 |
| 流 | 条件仿射耦合＋关系反馈 | 带时间的rectified-flow速度场＋反向流 |
| 对齐单位 | 64维向量重排的局部块 | 词级对齐输入，经各模态编码器汇聚 |

数据从官方 `datasets/download_datasets.sh` 中的MOSI Dropbox地址下载。SHA256：`5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b`。不同数据与编码器设置带来的收益必须与新增结构收益分别验证。

预训练模型为 `microsoft/deberta-v3-base`，Hugging Face revision：`8ccc9b6f36199bec6961081d44eb72fb3f7353f3`。权重与分词器文件均已下载、校验并传到SSH环境。

## 运行方式

远程工作目录：`/data/coding/selective_flow/strong_baselines`。隔离虚拟环境 `.venv` 复用现有PyTorch，仅在该环境补充依赖，未升级原有训练环境。

`run_careflow.py` 导入未修改的作者源码。设置沿用README的MOSI命令：100epochs、batch32、lr1e-5、ratio4、share_dim100、inter_dim150、Transformer3层、Euler2步、前向权重0.2、反向权重0.1、eps1e-3；seed先用作者默认128。验证与测试batch128，默认不开启attention-mask选项，保留作者逐批min-max归一化。

区别：作者每轮评估测试集，本脚本只依据验证集的batch平均MSE选择检查点，训练完成后才评估测试集一次。这不改变模型、训练目标或选择规则。当前为单种子首轮复现，尚未宣称恢复论文结果或多种子稳定性。

检查命令：

```bash
.venv/bin/python run_careflow.py --stage check --repo CaReFlow \
  --backbone deberta-v3-base --data mosi.pkl --out careflow_check
```

正式运行：

```bash
.venv/bin/python run_careflow.py --stage run --repo CaReFlow \
  --backbone deberta-v3-base --data mosi.pkl --out careflow_seed128 \
  --epochs 100 --seed 128
```

检查阶段逐张量核对预训练编码器权重，验证文本、语音投影、前向流、反向流梯度，并执行一次优化器更新与原尺寸验证批次。正式输出包括源码／模型／数据哈希、逐轮训练记录、验证选择文件、最佳模型检查点和最终预测。

## 结果状态

首轮官方代码训练与验证诊断均已完成：100轮、seed128、最佳检查点第72轮。测试含零MAE **0.6450**、去零MAE **0.6532**、去零Acc2 **87.18%**。当前仅单种子，尚未恢复论文报告水平。结果、曲线和解释见[首轮结果汇总](careflow_seed128/结果汇总.md)。

源码／协议／结果包／最佳模型哈希及保存预测复算全部通过，核验见 `careflow_seed128/local_verification.json`；本地最佳模型为 `best_seed128.pt`。验证诊断显示当前预测主要依赖文本，音视频表示置乱几乎不影响预测，因此不能把相较旧原型的性能改善归因于流。

以 `careflow_seed128/history.json` 查看训练进度。只有存在 `results.json` 且日志出现 `RUN_COMPLETE` 时，才表示正式训练及最终测试均已完成。源码下载和单步检查通过均不等同于完成复现。

最近进度快照记录在 `status_snapshot.json`，该文件不是实时状态。验证集诊断已接到本轮训练完成后的流程中，单次入口为 `finish_validation_diagnosis.py`。诊断只读取选定检查点的验证集表示，不再评估测试集；原始模型、训练入口和训练设置均保持本轮锁定版本。

## 实现审计与后续研究

[官方实现审计与后续实验](官方实现审计与后续实验.md) 记录流的计算位置、前后向梯度路径、代码与论文表述差异，以及用户提出的流内信息读取与反馈如何开展同条件比较。

`diagnose_validation.py` 已正式执行成功，状态为COMPLETE；输出为验证预测复算核对、表示替换／置乱指标、映射几何及原始验证表示。完整报告见[验证集诊断](careflow_seed128/验证集诊断.md)。

## 同条件对照

`run_control_baseline.py` 复用锁定的作者编码器、分词、归一化、优化器设置和选择规则；先构造完整作者模型，再移除流。`plain` 将未映射的三模态表示输入原融合头，`text_only` 进一步移除音视频编码器，其融合输入位置置零。两组均从预训练模型重新训练，使用100轮、seed128，训练完成后测试选定模型一次。

移除流改变了随机数消耗，因此相同种子不等于所有轮次具有相同的batch顺序或dropout抽样；对照记录这一限制。首轮比较后仍需多种子确认。输出分别位于远程 `control_baselines/plain_seed128` 与 `control_baselines/text_only_seed128`，当前是否开始或完成以 `control_baselines/status.json` 为准。
