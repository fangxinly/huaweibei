# 归一化配方的主源证据与适用范围

MAG-BERT 协作者在 issue11 指向 PR10，限定为当时发布的 MOSEI。合并 PR 的说明包含清洗/归一化示例；mean/std 没有 axis 参数，按写法归约整个数组，不能推出每个通道的时间均值为零。AST 静态核对通过，未执行该源码。

MISA 的 MOSI 生成源码明确按实例、按通道沿时间轴中心化和缩放，EPS为1e-6；它先用 SDK 词对齐平均并删除 sp，保存分开的 split pickle 和含 actual_words 的四元素模态 tuple。它是对照配方，不是本次官方三元素 tuple/pickle 的生成身份证据，不能移植它的 epsilon 或过滤步骤。

因此，中心化是已有方法的真实操作，但本次输入是否如此生成仍未知。两来源的不同归一化轴尤需区分；共同下载链接、格式和类似数值现象都不替代源版本和文件 SHA 的绑定。通道语义/单位、哨兵、精度、填充与曝光范围未闭合。

只抓取公开源码/API回执，PR补丁包含公开 notebook 元数据；没有执行 notebook 或任何下载的源码，没有读取本任务数据数组/标签，没有新训练/诊断/求解/评分或 SSH。全部原视频、折、lambda、预测和结果保持。

主源：
- [MAG issue11 协作者说明](https://github.com/WasifurRahman/BERT_multimodal_transformer/issues/11#issuecomment-737673762)
- [MAG PR10](https://github.com/WasifurRahman/BERT_multimodal_transformer/pull/10)
- [固定 MISA MOSI 生成代码](https://github.com/declare-lab/MISA/blob/ec42faddde0d210cf7368aebf2118fe9570e7102/src/create_dataset.py#L197)
