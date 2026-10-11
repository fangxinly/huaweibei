完整流fixed v2构造源、角色守卫及订单本地接续

实际UTC 2026-10-06T12:38:04.458206+00:00。新目录C:/Users/21234/Documents/Codex/2026-10-05/ni/work/minimal_fixed_runtime_v1_local_20261006T1228Z，D原件D:/CodexBackups/selective_flow_20261003_1105/minimal_fixed_runtime_local_candidate_20261006T123804Z。构造源SHA e86ac4e2f8243fe1e2ef356af650062c318e75781b4c0722a1222afa7817e84f，candidate plan SHA e4d0c71a73186ae9073dc0d88ccdd4fd753a4d81a967eb458a52a0669437fa63。这是本地源候选，不是完整Torch模型已构造、GPU通过、正式100源/资产/预算联合冻结或新分数。

完整encoder+最小flow的构造路径已写成候选：公共DeBERTa与新任务seed91819、FIT-only统计、预先删除作者8个unused流模块及pooler、直接挂载fixed flow、删除decoder_modules重复注册、新优化器完整唯一覆盖和空状态检查。构造器临时创建作者optimizer/scheduler后丢弃；零步骤记录准确。clean initial克隆模型与Torch/CUDA RNG，但实际checkpoint保存/另实例严格重放及预检runner尚未实现。资产来源与helper依赖沿原已核pin，实际新环境可达/资产未在本轮验证。

角色守卫仅接收TRAIN1281，FIT695/30、INNER153/4、OUTER433/18互斥；输入张量化全部dummy标签，FIT监督单独请求。INNER选模标签只在预测文件SHA、完整行序、有限形状和model-state SHA标记核对后读取，不能据此称独立确认。OUTER、DEV、TEST输入在此预检候选直接拒绝，未实现正式完成后的OUTER授权。可信pickle反序列化会存在其它split字节，只索引TRAIN，不冒物理未加载其它split。

无标签新订单使用独立NumPy PCG64 seed91819，每轮FIT全695行，100×695；每轮21个32行批及23尾批，共2200更新，前10轮完整可供未来匹配。订单不消耗模型初始化/Dropout RNG。新构造随机数消费与旧版不同，不冒旧init等价。真正控制候选的同容量/init/order/shared10比较协议仍未冻结。

本地AST编译及合成毒化标签守卫通过：拒绝OUTER、INNER冒FIT、浮点行号、重复行、负行、未冻结预测、错误SHA及错误角色预测行序。独立检查100轮完整FIT排列、尾批和视频互斥；不导入Torch/构造optimizer，不读真实数据/标签，没有模型前向/反向/HVP、磁盘重放、显存时间或成绩。正常32/含454、505、620混合单token候选/尾23只是未来行号提案，原输入真实mask仍须核。

下一步先补预检runner的两真步全保留有限非None梯度、尾23仅前后向计时、lambda无关的本参考真实单token边界、完整INNER原输入换标签0、精确FIT统计缓冲、整新pt另实例strict重放和原回执审核。正式100 runner/最早INNER最小选择/自然退出/实际预算与至少2h保存余量、完整D+异节点CPU另需真实通过。当前租期未核且余量失败，不执行新GPU预检/100，也不重试未经平台核对的A/B连接。

原完成源/权重/正负分数均保留。本轮无新研究结论，不重发建议聊天，仅按用户静默规则保存准备证据。
