# 引用仓库的 raw 输入来源关联

本次仅读取公开源码，固定了各自实际提交，并保存 HTTP 回执与文件 SHA；没有下载数据、运行外部源码、连接租用节点或重新诊断/训练/评分。

已保存的 CaReFlow 下载脚本，与两个致谢仓库的下载脚本逐字节相同。这建立了共同资源位置的来源关联，不能证明那些位置当前或历史返回的数据与本任务官方 pickle SHA 相同。

检查 MAG-BERT notebook 的代码单元，仅看到加载、交互查看和词对齐/形状断言，没有构造或归一化配方。未运行代码、未查看 notebook 输出。检查范围为所选源文件及文件名清单，不是全仓库全文或历史穷尽审计。

逐片段中心化仍未得到源码证明，通道语义/单位、有限缺失哨兵、精度、填充及上游拟合/预训练曝光也未闭合。原全部视频、折、lambda、预测和负结果保持；不能借此改尺度下限或按 TEST 救分。

固定主源：

- [MAG-BERT 下载脚本](https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/dc7876fc30f7ef362999200911e3d4d8a2bca107/datasets/download_datasets.sh)
- [MAG-BERT notebook](https://github.com/WasifurRahman/BERT_multimodal_transformer/blob/dc7876fc30f7ef362999200911e3d4d8a2bca107/examine.ipynb)
- [ITHP 下载脚本](https://github.com/joshuaxiao98/ITHP/blob/c5414d53e635126cdaf5ad1a5b165c5e7128a412/datasets/download_datasets.sh)
