# C恢复锚点与最低材料

已核对14:10新snapshot及相关成员SHA。C正式predicted运行截至81轮，best-so-far已为72轮；31轮临时副本的best30已不是最后观察的最优。这里没有取得72轮权重内容或100轮最终选模。

最后观察81轮内最小作者批次等权DEV MSE为0.6408416629，仅用作恢复历史核验锚点，不能作为完成实验的指标。

新取回history必须连续1..100且前81条逐字段完全等于已保存历史；protocol、100订单文件及shared_phase必须与锚点逐字节一致。每轮best记录及100轮全局最小选模重新计算；选中阶段<=10时恢复fixed的规则仍保留。当前锚点下最终best只能是72或82..100，不能回填为30。

最低工件为七核心文件best.pt、protocol.json、batch_orders.npy、history.json、selection.json、results.json、predictions.npz；shared_phase.json额外必须保留。完整best.pt>700MB，全文件SHA绑定selection，并独立核验Torch ZIP完整CRC；229预测/标签、六方向u/权重、own、fixed参考及all-off全部核验。文件体积相等不证明权重相同，旧31轮与最后观察best.pt恰好同体积，不能据此复用。

工具work/verify_recovered_counterfactual_artifacts_v1.py已在A的真实完整工件上通过；它仅证明CPU工件一致性，不证明原训练进程完成时刻、当前GPU身份或PyTorch权重重放。当前C临时目录必须拒绝为最终材料。

取回材料后还需要真实目标GPU UUID及来源记录、冻结权重原方法229重放、预定29条件诊断和独立节点七文件CPU SHA。可用新节点恢复时应单独记录实际目标UUID；不能冒充原C UUID，不能修改已冻结训练源。当前无有效新节点信息，不执行任何连接、重训或新训练。

13:40/14:10保存证明保持；14:30真实capture仍缺失。当前恢复也不能补记该期限。A独立CPU原回执文件如可取得应原样下载并与保存的终端证据比对，不能由摘要重建原回执。

现有材料会永久保留，所有新材料使用新目录/文件名。官方TRAIN1281/dev229，不访问TEST选结构，不称五seed稳定/SOTA/共享补充干扰真值。
