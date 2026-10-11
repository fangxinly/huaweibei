# 用户授权：先集中保存到4060Ti

用户在本聊天明确要求所有项目文件先保存在4060Ti。已核实目标可用55.19GB，三个来源当前项目共25.75GB，足够先保存。该服务器保守到期仍为北京时间2026-10-04 00:55；训练继续，原始文件、本地已有24份完整权重以及P4临时副本均保留。

目标根：`/data/coding/selective_flow/central_backup_20261003T1040Z`，各源分别位于 `original/strong_baselines`、`instant/strong_baselines`、`memory/strong_baselines`。4060Ti自身的活跃项目保持原位。

三路SCP -3递归传输已认证并真正开始，尚未完成：原服务器会话64700，instant63642，memory67329。通过本地网络流中转，避免在本地写大文件。启动清单 `central_transfer_launch_20261003T1042Z.json` 不含凭据。先前SCP URI写法不兼容；使用命令专用的 `relay_connections_v1.conf`，不修改全局SSH设置。

各源 `central_backup_records_20261003T1040Z/source_manifest.json` 已生成，逐项记录大小与SHA。来源清单分别8159/8108/8089项；21,632,688,930 / 2,802,091,766 / 1,316,532,951字节。全目录包含代码、数据、环境、预训练模型、已有与当前模型权重、失败尝试及研究记录。

目标CPU核验器PID2337已启动，当前 `WAITING_FOR_INCOMING_TREES`；状态为根目录 `verification_status.json`，launch为`verification_launch.json`，日志`verification.log`。它等文件到达后校验稳定文件SHA，不增加GPU任务。必须核对SCP退出码、源清单、目标文件及SHA之后，才报告稳定内容已保存。动态日志/状态/当前best checkpoint需补同步；这不是一个暂停训练后的原子全进程快照。

三路全目录传输不得重复启动。失败保留已有目录和记录，先查源、目标、资源与实际错误；新修复另版本目录。暂存服务器也会到期，独有的新结果应在短租到期前迁往长租P4或耐久存储。现有来源仍保留，禁止删除权重或研究文件。

本地十分钟监管已更新覆盖这三路传输、核验器、四台训练、同硬件控制排队和各租期窗口，不把启动或排队报告为完成。
