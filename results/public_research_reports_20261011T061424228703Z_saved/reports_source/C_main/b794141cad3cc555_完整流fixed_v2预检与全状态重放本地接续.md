完整流fixed v2预检与全状态重放本地源接续

实际UTC 2026-10-06T12:55:57.322073+00:00。source-only目录C:/Users/21234/Documents/Codex/2026-10-05/ni/work/minimal_fixed_precheck_local_20261006T1240Z，D原件D:/CodexBackups/selective_flow_20261003_1105/minimal_fixed_precheck_local_candidate_20261006T125557Z。新runner v2 SHA 647a98c14a7218df48443af9ed65630b4a8d642e36c21cd94feacd67e7ee1d54，candidate plan v2 SHA 88b32e03e162af8cd9d2afec3efcb34b72f2e7aef00575632d8f7e5d3bba8d5c。本轮未导入Torch/连GPU/读真实输入或标签，没有新分数或真实预算、参数梯度、磁盘重放证据。

候选预检先检查可信人类/平台来源、确证租期、预检20min+2h保存余量、5min内UUID/空compute/完整argv/主机身份/源资产完成空间证据，再导入Torch。JSON字段不验证来源或替代授权；合成正门控不是真实GPU门控。当前平台未核且估计已过，执行门槛关闭。空间方案远端4GiB、永久D6GiB用于两份整新状态及归档，未在真实GPU执行环境验证；6GiB Torch分配/保留峰值不冒硬件全局显存测量。

正常32和mixed-singleton32各1真optimizer/scheduler步；23尾批只前后向计时，参数SHA必须不变，禁止第三步。实际主/辅目标分别求探针梯度：backward字段应仅辅助有路径，decoder仅任务有路径；最终聚合所有保留参数finite非None，零值准确记录。不用总损失减MSE制造抵消路径，不保留不用utility/pair/反事实decoder。所有这些都是尚未执行的源码断言。

clean initial全参数/缓冲在步前保存，Torch/CUDA和Python/NumPy RNG分别记录；两步状态另整pt保存，原INNER153零和7 dummy标签重放0，参数/FIT统计SHA不变，自己的新pt文件/张量SHA与source/fold/seed/scope/steps核对后另公共初始化实例strict加载，原输入差<=1e-6。全过程不用真实INNER标签，不采OUTER。正式训练重置全部状态/RNG的runner仍未实现。

v1从未Torch/GPU运行，原源/plan/AST回执全部保留。局部源码审阅发现terms局部变量可能在删除首模型后仍保留autograd图，新增v2每步释放terms/names，并在fresh重载后释放不用clean CPU克隆；不是实测OOM或GPU故障，也未改旧完成源/模型/目标。

本地AST检查gate在Torch导入前、仅两步循环、tail无额外step、strict=True、无INNER真标签调用；JSON fixtures拒绝14项不合法条件及clean initial冒两步状态。正fixture只是合成输入；真实执行门控未过。原输入单token454/505/620 mask、主辅梯度、实际时间显存/完整重载全待真实验证。HVP/完整encoder二阶未实现，未来控制求解另验，不冒固定参考预检已覆盖二阶。

下一依赖：自然退出wrapper、独立CPU整新checkpoint/数组/原回执审核、实际capture新源/root/argv覆盖、真实资产和预算/保存，以及formal100与OUTER完成授权。当前不启动GPU/100，不重试A/B或向建议聊天发送小准备，10min静默规则维持；整体研究/租期未完成。
