# 正式TRAIN/DEV来源与共同预算：局部适配准备

从固定CaReFlow训练源提取候选组件，保留tokenization、数据转换、模型/AdamW/scheduler构造、每批训练、batch-minmax和DEV批MSE函数的原AST；排除原pickle加载器、每轮TEST函数/训练总控/main及TEST CLI参数。原冻结源码未改。此候选没有启动入口或完整runner，编译只检查AST，未导入Torch或加载模型。

共同预算候选按既有固定baseline：TRAIN1281、batch32、drop_last=True、100轮×40正式更新=4000；每轮完整1281索引排列后丢尾1。DEV按固定顺序128/101两批MSE算术平均严格改善选模且平局最早。纯合成检查通过重复/短/非整数排列拒绝、尾批数量、错误DEV批结构/非有限MSE拒绝、严格平局保持。未生成真实官方ID/共同订单，也未把候选预算冒双方实际训练预算。

下一必须完成单方法固定与官方TRAIN/DEV-only角色/ID/尺度/公共资产，完整双方runner/共有实际订单/全状态/CPU与capture协议及5分钟资源和2h保存门；模型DEV真值与dummy替换不变须实际验证。未将201九读出排名用于方法选择，未读取pickle/NPZ/真实标签，没有新分数或正式TEST。旧baseline完整字节D保存通过不替代新协议下的完整模型/订单资格。
