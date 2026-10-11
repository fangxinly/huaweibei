双方新fullTRAIN100完整保存及固定best一次DEV五项实际结果

实际评分子进程 28316 UTC 2026-10-07T03:50:17.953086+00:00 自然0。双方先完成GPU原训练/整selected及resume D保存/异节点CPU/真实fresh重放capture与联合门，再固定评分来源和两方父SHA，一次全五项同时评分，不选择新checkpoint或读出。

| 固定模型 | epoch | Acc7↑ % | Acc2↑ % | F1↑ % | MAE↓ | Corr↑ |
|---|---:|---:|---:|---:|---:|---:|
| minimal_fixed_F | 89 | 49.344978 | 87.500000 | 87.424393 | 0.621011489124 | 0.857556416296 |
| careflow | 93 | 49.781659 | 87.962963 | 87.924750 | 0.604523826276 | 0.859493846929 |

F在全部五项点值上均差于本次新CaReFlow；当前版本未达到五项全面超过。此DEV229已探索并用于预固定batch MSE选模，Acc2/F1排除13个0真值后216行，不是独立确认或论文TEST。不同架构的参数与实耗仍不相同；名义共同seed128/100轮/batch32/drop1/4000实际更新/共同订单/同批MSE严格earliest选模。旧弱缓存不能取代本次指定CaReFlow。

全部7415841507字节完整模型在D+异节点B原CPU字节/SHA/ZIP CRC/唯一通过。F CPU child31944/C31863自然0，374/329模型state、364/323 Adam状态及各4000 step、scheduler、全部RNG、原100轮数组/预测冻结先于label/订单核过，独立DEV选择MSE误差0。不是CPU模型前向。原CPU31642/31714自然1是JSON list/Torch tuple尾批字段比较的表示错误，原失败与严格只修此比较视图的补充源/协议已完整保存；没有删改训练/旧冻结源/模型。

F fresh child3173/C1162自然0；独立公共实例整selected重放、dummy0/7、自己的统计及全部参数/RNG不变均过。各38成员真实capture，CPU C208/F206成员，GPU各156成员；actual UTC和完整argv均由原件关联，不用目录后缀当实际时间。后续静态本地总封装不是新的remote capture。

本轮Windows文本读取消去CRLF导致回执传递SHA门失败，发生在fresh根/模型子进程创建前，改为原始字节传递且原失败保留。所有本批SSH/SFTP均明确exit/bye实际0关闭，ID禁复用。

停止当前minimal_fixed_F版本基于该DEV的seed/feature/readout救分或扩矩阵；仅独立审阅反证与合理后续成本，不外推否定所有flow/消息/残差。最终TEST流程尚未冻结/执行，历史旧baseline TEST访问不能抹去；正式五项目标未完成。09:30/11:30/13:00 UTC三机动态强化保存仍待实际执行；租期保守13:30非平台确认。

D原件：D:\CodexBackups\selective_flow_20261003_1105\paired_fulltrain100_complete_actual_20261007T030311Z
评分plan SHA a23ba23e62b49534b739e77d0eeee8a0ae7c9582868dceadabcea9c840b089cd
评分result SHA 6e7e0d62de1aa933c7ffa1b5b374d09149c965c4552104ea47e7f569fa50cf87
全记录 SHA 60f835f75562189a3f119fc68b001f20d9b30b9e6376f1a50ce0fe2a9a2ae741
