# 正式比较原baseline整文件与角色守卫准备

实际clock开始 2026-10-07 00:02:08 UTC。在已存manifest中定位指定seed128的原selected best（epoch72），原C整文件742279993字节的fresh完整SHA与selection精确一致，ZIP完整CRC及唯一成员通过。原D顶层同名路径当前不存在；这只描述该路径，不推断所有D归档缺失。原C字节已本地复制至 D:\CodexBackups\selective_flow_20261003_1105\official_baseline_artifact_guard_preparation_20261007T000208TZ\original_selected_best.pt 并重新核完整SHA/CRC；未远程重下载、未重训。

这通过整文件字节保存门，尚未加载Torch/state张量或运行完整模型重放；不据此宣称正式缓存已可复用。原protocol、源码、历史一次TEST访问边界继续保留。正式共同订单/尾批/更新/DEV选模尚未冻结；新共同订单生成器不能冒与旧缓存实际订单一致。

新增train/dev-only角色守卫只用合成哨兵核验：禁止角色在接触底层mapping前拒绝，合法角色核官方行数与唯一ID，TRAIN/DEV重叠拒绝。没有打开pickle/NPZ/真实标签、导入模型或执行GPU。该守卫尚未接入完整正式runner，不能冒运行时端到端标签隔离。
