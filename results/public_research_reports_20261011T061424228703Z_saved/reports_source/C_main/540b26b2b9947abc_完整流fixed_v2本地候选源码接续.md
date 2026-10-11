完整流fixed v2本地候选源码接续

实际UTC 2026-10-06T12:21:48.370161+00:00。原完成科学源/协议均未改；新flow子模块独立候选目录C:/Users/21234/Documents/Codex/2026-10-05/ni/work/minimal_fixed_flow_v2_local_20261006T1214Z，永久D D:/CodexBackups/selective_flow_20261003_1105/minimal_fixed_v2_local_candidate_20261006T122148Z。主源码SHA 0f64dea25c522c6c8415297cc329e97cf484b6cbba499fe4e22b9db2c39a5a8c，候选计划SHA f1c49e68ddd6e842bafdeaae86d5820fb4c2553e9b73529960e566e8af3e25b7。

直接构造有明确预测/辅助作用的forward/backward fields、解析单token reader、role/unimodal heads及六donor；从构造依赖链删除旧feedback、pair/utility/predicted gate，不执行10条参考终端。不是系数置零继续计算。新任务随机数消费因此不同，不能沿用旧init等价或task权重，未来须公共预训练/新clean initial单独存证。

一pass两Euler：第一步使用零context，第一步后根据stage slots更新context，第二次update返回原context。fixed权重.5、selected再乘.5、context再乘.5，因此各供体从零context的实际系数.125。新辅助目标严格MSE+.02FM+.01cycle+.05unimodal+.01variance；unimodal为source和stage1 observer各.5。cycle终端/context未detach，第二阶段FM context也未detach，不能称只训练backward；donor特征及原目标detach边界保留。

本地AST编译、声明模块/调用删减核验与NumPy独立上下文恒等式差0、synthetic双步/单token mask日程检查完成exit0。不导入或运行Torch模型，不读真实数据/标签，不核GPU、真实参数梯度、HVP、完整初始化、原输入或整pt重载，不产生任何分数。解析单token旧源码只是精确复制，旧验证不冒新整模型验证。

尚未实现完整encoder runtime/角色守卫与预检训练器；公共预训练资产、FIT-only统计、INNER原输入dummy标签严格重放、优化器覆盖/唯一性、正常两真实步骤/23尾批/单token梯度、clean initial与预检分离、完整100订单/选模/实际显存时间与完整D+异节点CPU/至少2h余量仍全待实现与实核。这不是正式源/预算冻结，不允许启动训练。A key和B认证限制未恢复，平台释放未核，租期估计经过也不能填卡或续租。

只本地准备按用户静默规则，不向建议聊天重发、不通知新成绩。当前真实标量/消息正负分数、最新三远端11:39保存、C12:06只读和A/B失败状态不变。下一步本地可准备完整runtime和角色守卫，数值检查须待可验证环境，不能把静态图当真实autograd。
