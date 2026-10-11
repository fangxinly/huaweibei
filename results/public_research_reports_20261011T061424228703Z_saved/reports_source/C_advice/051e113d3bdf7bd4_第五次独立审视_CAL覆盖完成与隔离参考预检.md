# 第五次独立审视：覆盖检查结束，先收敛参考目标与参数清单

2026-10-06。本轮读取CAL机制报告、原回执/审核、保存联结及完整流参考草案，并只读审阅旧流/损失/编码适配/单token代码。核新NPZ SHA及无y成员，用fold0角色和长度元数据核单token见证；没有新GPU、模型运行、原始任务标签读取、EVAL指标计算、主源修改、训练、子代理或跨聊天发送。

**结论：原生半径限定检查可以结束，不再补这条旧参考上的收益实验。下一步支持实现单折隔离流参考的最小预检，但先明确目标和参数清单。fixed全序列流作为新参考合理；utility与pair并非必不可少的整体，应分别处理。建议最小版移除预测不使用的utility校准及其反事实分支，保留明确选定的基础流辅助，pair是否保留在新源执行前冻结。预检通过也不等于参考100/学生100已获科学依据。**

## 1. CAL覆盖确已关闭，边界也应随证据关闭

新CAL418行/18视频的F与旧native重放最大误差2.38e−7；OT cap有效唯一候选含F均值1.8325→8.8852，合法且输出移动超过1e−6的非F覆盖96→418行。旧native最大幅度候选396行过cap、22行仍被过滤。新档案SHA与GPU/CPU/保存联结一致且没有y成员。它证实粗网格覆盖缺陷与原生尺度修订的实际可行性，前次八臂负结果及停100决定保留。

274行有负CAL代理候选只说明评分窗口覆盖，不说明纠错收益。全418行半径非零，未测合成零半径或新选择器全拒绝/零cap拼接重放；将来控制器需要这些边界，当前隔离参考尚未使用候选选择器，无需为它重跑旧CAL补部署证书。

## 2. fixed参考及utility/pair目标：可行，但需作一个明确选择

已读`counterfactual_flow_model.py`：fixed使用恒定0.5消息权重，utility_heads输出的predicted_weights不参与fixed预测；其输入features与反事实标签目标均detach。utility校准只训练utility_heads，不回传到供体、流或编码器。训练时仍为它计算1个固定参考、6个单通道移除和3个联合移除的终端预测，共10条额外参考终端支路；它们共享第一阶段状态，不是10次DeBERTa原输入前向。

pair_sentiment则直接由未detach的pooled槽表示产生，能回传pair_heads、reader/流/源表示；它是FIT标签的额外表征监督，会改变参考训练目标。FIT-only执行时，两种辅助目标都没有天然外折泄漏，但“合法”不等于“最小参考必须保留”。

建议只选一个版本冻结，不在本轮跑多个损失配方择优：

| 版本 | 明确行为与适用范围 |
|---|---|
| **推荐：最小fixed流参考** | 保留旧fixed预测路径、整序列两步Euler、reader、六供体、decoder；任务MSE加预先选定的基础FM/cycle/unimodal/variance目标。移除utility_heads、utility_calibration及10参考支路；pair可默认删去以减少辅助标签路径，或明确以原0.025保留，但不能运行后据结果决定。命名为新自定义fixed参考，不称原旧完整训练目标逐项复现。 |
| 旧fixed辅助目标复现 | 原pair与utility都保留、权重/归约/目标detach均冻结，所有反事实仅FIT。它是合理的复现参考，但utility不改善fixed预测路径、增加代码与生成成本；需承担更大预检/保存清单。不能用“所有保留张量都有梯度”作为保留理由。 |

删utility不能只是把系数置0：旧代码仍计算10个参考且构建头。也不能删属性后继续调用会访问它的旧forward。应在新源重组fixed模块与损失，旧科学文件不改。若移除基础cycle目标，backward_fields也需移除或明确冻结；若移除unimodal目标，相应独立头不能继续作为“全参数微调”统计。基础FM/cycle等虽在训练中使用、预测时不用，仍是有梯度路径的训练模块，不应同always-unused模块混删。

固定参考的“完整流”应明确是**全序列状态流**。此旧counterfactual类继承`BaseFlow('mid')`，实际一pass/两步Euler；`CONFIG['passes']=2`不决定该路径。不要因为“完整”而无意改成cycle两pass/四Euler。与旧参考架构/目标的差异逐项列出，当前几步预检只验证所选版本能执行。

## 3. 先核数据、初始化和训练图；不要以全部NonNone替代语义审计

**数据入口：** 固定fold0 FIT695/INNER153/OUTER433，冻结行/视频SHA和角色互斥。FIT真实标签用于任务及被选辅助目标；INNER输入推理直接构造dummy label，不从真实INNER y转换/取shape后冒无标签入口。OUTER、DEV、TEST、CAL/EVAL标签均无预检入口。仅容器存在其它标签不等于训练使用，但报告应准确区分打开容器、转换哪个角色的标签和真正进入loss的行。

`encoder_adapter.fit_statistics`按传入dataset的内容mask选有效token；若误传完整TRAIN再切FIT，就是全TRAIN统计。必须先选FIT原输入，再double统计音/视频mean/std/active，固定CLS/SEP/padding排除规则。记录统计行/有效token数与SHA；INNER毒化标签和禁止角色请求均不得改变统计或预测。缺失/常量通道、位置dtype缓冲初始化也要记录，不能把旧统计/缓存放进新组装。

本轮角色元数据核到454、505、620均是FIT0单token候选，454/620不是OUTER0。只读了旧档案的输入长度与角色，不使用其拟合特征；新源仍应从原token mask重建这些见证，不能将旧C2缓存当原输入。

**初始化证据：**

1. 公共DeBERTa文件SHA＋保留张量名称映射逐一核对，单列合法缺失/额外键；完整任务checkpoint、A/C2/T0权重和旧优化器/缓存没有加载路径。T0是已拟合教师，即使同折合法，也不等于公共预训练初始化。
2. 固定seed91819、构造顺序和新配置；任务模块采用新初始化。供体输出层等按原结构零初始化可以保留，但要写明，不要求每个任务元素都随机非零。
3. 完成unused模块裁剪、reader分支安装、FIT统计注册、dtype占位缓冲显式初始化后，记录public子集SHA、任务参数SHA、统计缓冲SHA和完整clean initial SHA。跨折比较非统计子集；统计差异是合法折差异。
4. 两步预检后的状态与clean initial分别保存。未来正式阶段从clean initial新进程/恢复初始化，并重置优化器、scheduler、随机状态与采样序列；不能只把权重换回去却沿用预检后的Adam或随机流。

旧`run_counterfactual_v5.py`会构造全TRAIN/dev、拟合全TRAIN统计、截掉尾行，并先建立old_optimizer再删除旧模块。新预检不调用其训练入口，只复用核验后的类/函数。若使用旧构造助手，要确认临时optimizer参数引用已释放，否则`delattr`不一定释放被引用的未用模块；最好先在CPU完成裁剪再迁GPU，且全程峰值仍从构造前计量。

**参数清单与梯度判据：** 先列保留/移除/冻结三类及每类理由，优化器成员应恰好覆盖唯一的声明trainable参数。正常多token批与含单token的FIT批都要求loss、有效梯度有限；全部声明训练参数必须有可解释的路径。记录None、全零和非零分别计数，按编码器/流/reader/供体/decoder/辅助头聚合。

“每个保留参数每批均NonNone”可作为拟合版本中活跃路径的检查，但不能先据这个条件反向设计损失。固定权重、detach、零输出层初始化或特殊单token分支可能使某些模块合法零梯度；零不等于未使用。只在某个合法分支没有路径的参数，其预期None必须在执行前列出，并在代表性正常批证明路径，不能事后给None找借口。always-unused或被整个目标删去的头应裁剪/冻结。禁止用`0*parameter.sum()`伪造NonNone，也不要因单token批零梯度删掉多token需要的reader参数。

主任务与各辅助的梯度范围按模块审计即可，不必永久复制多份全DeBERTa梯度数组。utility若保留，范围应限utility头；pair应有表征路径。某些预期模块只得辅助梯度应如实称训练辅助，不称直接任务端到端梯度。

## 4. 最小执行次序与应削减项

**先做三道门，前一道失败即停止后续：**

1. **只读/CPU组装门：** 角色、输入mask、FIT统计、公共权重和任务初始化、已选fixed结构/损失、optimizer清单和clean initial保存清单全部冻结。新源需要实际实现才能谈GPU可行。
2. **新原输入GPU门：** 正常FIT批＋含单token的FIT批，两次明确optimizer步骤，核全部声明训练梯度和实际学习率/更新。第一次Adam step才分配状态，必须计入峰值。单token解析reader需在新随机全模型上重新核完整前向/反向有限，不把旧冻结终端的HVP证据移入。预检固定LR/schedule并日志化；若formal warmup首步LR为0，不要求每个参数两步都改动，但须区分优化器调用与实际权重更新。
3. **INNER与保存门：** 步骤后模型eval，153行INNER原输入零标签预测写盘；完整state保存，按同一新配置/MethodType安装/统计缓冲重建另一个实例，再strict加载、eval和原输入重放≤1e−6。state_dict不保存monkeypatch、mode、epoch、dropout设置或reader安装行为，必须有代码/配置清单。清理优化器、梯度与临时实例后再重载能降低并存峰值，计量不重置以掩盖早期峰值。

本次应削减：utility反事实训练分支（若选最小版）、全套旧控制诊断、新HVP求解器/Oracle/候选接受器、OUTER/CAL/EVAL性能、多折/多seed以及多损失版本。不削减原输入整流前向、真实FIT梯度、完整INNER零标签磁盘重放、clean initial与步骤后全权重保存。未来学生主辅门控、零半径和选择器拼接仍另设预检，当前参考没有这些模块，无须提前塞进这20分钟。

## 5. 20分钟/6GiB/8GiB都只是待实测上限

695行batch32为21个完整批＋23行尾批，每轮22次更新，100轮2200次；不能复用旧TRAIN1280的40批订单，也不能只测两个全批就把尾批和100次INNER验证当零成本。预算至少包含：构造、统计/数据传输、首次Adam状态分配、稳定完整批与尾批、完整153行INNER、序列化/磁盘重载、实际D下载与独立审核、最终保存余量。

两步计时只给初步估计。先用冷启动与稳态分开保守计算训练量；尾批23行至少验证完整行序/前后向及时间，必要时可额外前后向但不加optimizer步骤，并准确记录。约式可写为100×(21*t32+t23+tINNER)加构造/保存等固定成本，t32取代表性保守值而非最短批。数据、步骤与预检本身的墙钟也须纳入截止时间。

若保留约1.85亿参数并用FP32 AdamW，参数＋梯度＋两个moment的基本量约16P字节，约2.75GiB；这不含激活、临时张量、公共权重加载副本或重载实例。因此6GiB是需验证的上限，不是由旧53MB缓存终端峰值或教师推理峰值证明。完整反向和首次optimizer step过后才谈可行。

batch32若不可行，可另冻结microbatch/累计版本，但需要新参数/目标说明。尤其variance_floor用批内std，utility正负组平衡也是批内非线性归约，逐microbatch loss简单平均不等于原batch32目标；dropout/流目标随机时刻与尾批权重也变化。普通MSE/pair的按行加权可加性不能扩展到整个损失。不以“effective batch32”冒原数值/目标等价。

8GiB空间要按文件账本核，而非仅比较单个pt：clean initial、post-step、原包/ZIP、未来best/last、可能的Adam恢复状态、异节点CPU原件和原子下载临时文件。可以让公共权重经合法引用/恢复流程复用，任务状态保存策略需明确；不要盲目每轮存整权重，也不能省掉本次完整clean/post原件而冒严格保存通过。磁盘空间在实际写入前fresh核验，不删旧材料换空间。

本轮读时UTC约07:36，估计到12:08约4.5小时，扣至少2小时保存余量后不是完整4.5小时训练预算；平台期限仍未核。08:08强化保存也不能被新预检遮蔽。若实测预算不足，停在已保存预检和下一阶段协议；无需为了本租期完成训练而改变轮数/隔离边界。预检通过只支持可行性，正式100仍须独立订单/选择/保存计划，不由几步训练loss授权。

## 6. 停止条件与真正读过的材料

初始化混入任务拟合权重或FIT外统计/标签、optimizer成员与声明不符、未解释的None或任何非有限梯度、normal/singleton新整模型不通过、strict磁盘重放失败、空间/时间不足，均停止正式启动，保留新源/回执与失败。修复只在新版本；不切换最好折、损失权重或seed求过关。

实际读取：

- `C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs/原生位移半径CAL零标签机制实际结果.md`
- `C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs/视频隔离完整流参考可行性协议草案.md`
- `D:/CodexBackups/selective_flow_20261003_1105/native_radius_cal_mechanism_actual_20261006T072031Z/`下`joint_preservation_audit.json`、`mechanism/receipt.json`、`mechanism/independent_audit.json`；nonF覆盖JSON由程序读取all汇总及边界字段，未冒全文重新独立重建所有每视频数值。
- 同D根`mechanism/cal_candidate_arrays.npz`：只核SHA、成员无y，不重复GPU或收益统计；SHA `f6685f7797f0deca4ceaf12858b2ba26831c9ca6179313f8497a1cf61d6ec8b6`。
- `C:/Users/21234/Documents/Codex/2026-10-05/ni/work/inflow_counterfactual_v5_deployment_20261005T0520Z/`下`counterfactual_flow_model.py`、`legacy_flow_model.py`、`encoder_adapter.py`、`run_counterfactual_v5.py`，全文只读审阅；辅助梯度范围是根据detach/前向路径的源码判断，尚无新参考GPU实测。
- `C:/Users/21234/Documents/Codex/2026-10-05/ni/work/finite_single_token_reader_v1.py`全文读取。
- `C:/Users/21234/Documents/Codex/2026-10-05/ni/work/group_teacher_plan_20261005T1650Z/{fit_0,inner_0}.npy`核原SHA与角色；旧匹配档案只取row/valid_lengths元数据核见证，未取任务y或拟合特征作新训练输入。

自己的只读核验源码/记录：`C:/Users/21234/Documents/Codex/2026-10-06/multimodal-flow-research-review/work/audit_fifth_coverage_and_fit_roles.py`与`work/fifth_coverage_and_fit_roles_audit.json`（同工作目录）。新参考源、GPU预检、参数规模/活跃梯度、实际资源和永久保存尚待实现执行；本轮未查文献，不主张贡献或泛化收益。
