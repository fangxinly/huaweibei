# 下载与复现

[已发布的模型与 MOSI 输入](https://github.com/fangxinly/huaweibei/releases/tag/reproducibility-20261008)，共 9 个附件，4,832,691,234 字节。每个附件的远端 SHA256 已与上传字节流核对；大小、下载 URL、分片位置和原完整文件 SHA 见 [附件清单](../results/release_asset_manifest.json)。

|附件|字节数|SHA256|
|---|---:|---|
|[mosi.pkl](https://github.com/fangxinly/huaweibei/releases/download/reproducibility-20261008/mosi.pkl)|14158784|`5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b`|
|[deberta-v3-base-pytorch_model.bin](https://github.com/fangxinly/huaweibei/releases/download/reproducibility-20261008/deberta-v3-base-pytorch_model.bin)|371146213|`691d48a2800b926a19e3051def466fc2cca4f59a15e42ce4a0cf7f1b380b5e33`|
|[deberta-v3-base-config.json](https://github.com/fangxinly/huaweibei/releases/download/reproducibility-20261008/deberta-v3-base-config.json)|579|`649f6a1ec33c6bdd9a6486d5c66019d461139e54957073eafe9bbc2d34c75b0b`|
|[deberta-v3-base-spm.model](https://github.com/fangxinly/huaweibei/releases/download/reproducibility-20261008/deberta-v3-base-spm.model)|2464616|`c679fbf93643d19aab7ee10c0b99e460bdbc02fedf34b92b05af343b4af586fd`|
|[deberta-v3-base-tokenizer_config.json](https://github.com/fangxinly/huaweibei/releases/download/reproducibility-20261008/deberta-v3-base-tokenizer_config.json)|52|`3f3978e0c036f2c2588cac34a6047cbb0af0b0dc1814254e291028529805496d`|
|[careflow-official-best93.pt](https://github.com/fangxinly/huaweibei/releases/download/reproducibility-20261008/careflow-official-best93.pt)|742340015|`9890b4fd2138e49a6500c52e72b0cbcc58151dc26e4b927656b39b6aa9e280f7`|
|[minimal-F-official-best89.pt](https://github.com/fangxinly/huaweibei/releases/download/reproducibility-20261008/minimal-F-official-best89.pt)|741809156|`7ec2173145bee6a80d6e7367e03f98f49cf87bf5d9eee5eb53558dbf81c839e8`|
|[official-upgrade-complete-final-and-selected.pt.part00](https://github.com/fangxinly/huaweibei/releases/download/reproducibility-20261008/official-upgrade-complete-final-and-selected.pt.part00)|1600000000|`d3c2e8cea8c1fd5d27544b42f88e4f3bfdadb9d8a71e552eecd7e662ff4fcaa5`|
|[official-upgrade-complete-final-and-selected.pt.part01](https://github.com/fangxinly/huaweibei/releases/download/reproducibility-20261008/official-upgrade-complete-final-and-selected.pt.part01)|1360771819|`171a95fb87e4d084fda820bf2c10ffd996a360e848d9313db8d9108dd98d8c94`|

对应官方结果：原流改进的 VAL MAE 为 0.597256537、TEST MAE 为 0.650616015；CaReFlow 的 TEST MAE 为 0.619535294；历史固定 F 的 TEST MAE 为 0.643699787。五项完整结果在 `results/official_upgrade_VAL_TEST.json` 与 `results/fixed_models_VAL_TEST_five_metrics.json`。此次上传没有产生新训练成绩。

## 先复核已保存的分数

克隆仓库并安装 NumPy 后，在仓库根目录运行：

```bash
python work/reproduce_published_VAL_TEST_v1.py
```

脚本按原五项口径复核原流改进 ON/OFF 的 VAL/TEST，以及 CaReFlow 和历史 F 的 TEST，误差容许值为 1e-12。CSV 便于阅读；原始 float32 预测值从 NPZ 读取，避免 CSV 字符串少量舍入造成差异。这里是既有预测的算术复现，不是重新选模型或新的独立实验。CaReFlow 的 VAL 点值保留在 JSON 中，此脚本没有据此声称重跑了它的 VAL 模型前向。

## 还原原流完整状态

下载 `official-upgrade-complete-final-and-selected.pt.part00`、`.part01` 与 `release_asset_manifest.json`，放到同一目录。需要额外至少 3.46GB 空间保存合并文件；脚本拒绝覆盖已有文件。

```bash
python work/reassemble_release_checkpoint_v1.py downloads/release_asset_manifest.json downloads official-upgrade-complete-final-and-selected.pt downloads/complete_final_and_selected.pt
```

原文件大小为 2,960,771,819 字节，SHA256 为 `9f670b5a2835be0d5d4638c34427024eadb326fc6e52ebff73a291ae2f2d34cf`。完整状态含最后第100轮、VAL选中的第30轮、优化器、调度器和RNG；**报告成绩对应 `selected_model`，不要用 `model` 的第100轮替代**。

原环境为 PyTorch 2.1.0+cu121；正式推理加载方式见 `work/official_anchored_upgrade_20261008T053429Z/infer_official.py`。归档协议含历史路径和机器资格，复现时需要生成新环境对应的资产/ID/源/预算计划，不能直接把旧租期计划当新执行资格。

`careflow-official-best93.pt` 对应官方 CaReFlow 第93轮；`minimal-F-official-best89.pt` 对应旧 F 第89轮。它们是已经选中的整模型，不是新的方法变体。输入资产包括当前使用的 MOSI 预处理 pickle、DeBERTa v3 base 权重、配置及分词模型。MOSI 文件包含原官方 TRAIN/DEV/TEST；训练仍应只索引 TRAIN，VAL 用于原选模，TEST 不能据此反复选结构。数据与公共预训练资产沿用上游来源和使用条件，本项目不宣称其所有权。

下载后的公共资产目录应按原接口放置（以下路径相对于传给 `--assets` 的根目录）：

|下载文件|资产目录中的位置|
|---|---|
|mosi.pkl|assets/mosi.pkl|
|deberta-v3-base-pytorch_model.bin|assets/deberta-v3-base/pytorch_model.bin|
|deberta-v3-base-config.json|assets/deberta-v3-base/config.json|
|deberta-v3-base-spm.model|assets/deberta-v3-base/spm.model|
|deberta-v3-base-tokenizer_config.json|assets/deberta-v3-base/tokenizer_config.json|

服务器密码未发布。本地原始权重、数据和冻结证据未因这次 GitHub 上传而删除。此前压缩后 SHA 无法核验的旧中间状态不在这些附件中；新 MSE 候选尚未完成真实训练。
