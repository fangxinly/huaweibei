actualclock 2026-10-07 01:57:08 UTC：新双方官方MOSI fullTRAIN100已真实启动并健康更新，禁止重复启动或停止。整体五项超过目标、100完成/最终保存/最终TEST仍未完成。

冻结源 /data/coding/paired_official_fulltrain_v2_20261007T011543Z，plan SHA 8dc4093ae64a61018fee0781d250b40e0ecf26fe11b6c9feb8529fd4ef1a95eb。与完成的fold100、FIT232/201及v1失败不同的新实验：原fixed F直接terminal vs指定作者固定CaReFlow配置seed128；双方官方TRAIN1281/DEV229、同100×40×32/drop尾1=4000更新、同物理订单、同DEV128/101批MSE float64算术均值strictmin/平局最早。无201读出选择或残差头、无TEST。baseline非CLI全默认/已核论文同配方；双方text nonreentrant gradient checkpoint/RNG保留和dtype-only marker0适配已预声明，未声称旧baseline数值等价或参数容量相等。

F root /data/coding/paired_fulltrain_train100_minimal_fixed_F_20261007T015004Z，wrapper2784/child2785；actual cold start01:52:56.248981。CaReFlow root /data/coding/paired_fulltrain_train100_careflow_20261007T015004Z，wrapper806/child807；actual cold start01:52:58.656119。双方各自完整model state/all RNG与原clean预检精确一致，Adam空/scheduler0/step0，新构造绝不继承预检2步。

原观察UTC01:55:02.781718/01:55:02.777681：F125与CaReFlow127次更新，前三轮已完成；实际累计reserved峰值4525654016/4481613824B低于原6GiB，无peakreset、无Traceback。此时的DEV MSE仅选模过程，不是最终五项成绩或TEST。NVML compute PID与容器child PID不同命名空间，不冒二者相同。不是此报告写入时刻的实时三机健康。

预检全部保存门已过：D D:\CodexBackups\selective_flow_20261003_1105\paired_official_prechecks_v2_actual_20261007T012116Z，四整件5931787667B真实D/B，全SHA/ZIPCRC/唯一/完整参数Adam两矩step2/scheduler/RNG/数组/source/fullargv/两原capture通过，原B TorchCPU自然0非CPU模型前向。原capture F46/61、CaReFlow46/61，actual receipts与source全部关联。先读《正式双方v2完整预检与保存实际接续》。

第一次formal派发误用paired_fulltrain100_根前缀，冻结wrapper在TRAIN子进程/模型/输出目录前拒绝，完整原9文件/源/fullargv/fresh检查/Traceback与原manifest真实D保存，退出码没有原wait记录所以保持未知；不冒已训练或新审批拒绝。corrected fresh root采用paired_fulltrain_train100_，模型/runtime/训练源与协议不改，原失败目录不改不删，所有三机fresh物理/全source-assets/空间/人类保守13:30与执行3h+至少2h保存余量重新通过后启动。

本轮启动原件 D D:\CodexBackups\selective_flow_20261003_1105\paired_formal_actual_dispatch_v3_20261007T015004Z；actual startup关联SHA 99931ec944e31b0c97ad9e9d9e08bf6c9b02197a367b11060d1502ea31d67667。只启动证据，不冒最终自然COMPLETE捕获或已保存final weights。自然完成后立即完整resume+selected真D下载、异节点原CPU、双实际COMPLETE/natural0捕获、fresh公共实例整selected229 dummy/state/allRNG重放及D关联，再另冻最终一次TEST协议。原10分钟观察，健康不停止/不重启，结束/新失败及时处理。09:30/11:30/13:00 UTC强化动态保存待做；保守13:30不是平台确认。整体研究与租期保存未完成。
