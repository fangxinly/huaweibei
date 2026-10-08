"""Build a forwardable report from saved results; no model scoring or fitting."""
import hashlib, json, shutil, sys, zipfile
from pathlib import Path
import numpy as np

clock = sys.argv[1]
out = Path('outputs')
base = Path('D:/CodexBackups/selective_flow_20261003_1105')
read = lambda p: json.loads(p.read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
results_path = out / '所有固定模型VAL_TEST五项实际对齐结果.json'
counts_path = out / '强弱情绪比例原标签实际统计.json'
diagnosis_path = out / '固定双方TRAIN_DEV退化诊断.json'
results, counts, diagnosis = read(results_path), read(counts_path), read(diagnosis_path)
models = results['models']

# Count the actual retained training rows using the original common 100 orders.
full = base / 'paired_fulltrain100_complete_actual_20261007T030311Z'
orders = []
for method in ('minimal_fixed_F', 'careflow'):
    orders.append(np.load(full / method / 'a/run/out/original_shared_TRAIN_orders.npy', allow_pickle=False))
assert orders[0].shape == orders[1].shape == (100, 1281)
assert np.array_equal(orders[0], orders[1])
assert all(np.array_equal(np.sort(row), np.arange(1281)) for row in orders[0])
y = np.load(Path(counts['inputs']['careflow_TRAIN']['path']), allow_pickle=False).reshape(-1)
used = y[orders[0][:, :1280].astype(np.int64)]
exposure = dict(actualclock_UTC=clock, rows_per_epoch_used=1280, epochs=100, batches=4000,
    total_row_exposures=int(used.size), weak_row_exposures=int((np.abs(used)<=1).sum()),
    strong_row_exposures=int((np.abs(used)>1).sum()), weak_percent=float(100*(np.abs(used)<=1).mean()),
    both_models_orders_exactly_identical=True,
    order_source_SHA=sha(full/'careflow/a/run/out/original_shared_TRAIN_orders.npy'),
    usage_rule_source_SHA=sha(full/'careflow/a/source/common_budget_selection_candidate.py'),
    rule='original train_batches: first40x32 entries, last1 omitted each epoch; repeated exposures are not independent samples')
(out/'正式训练强弱样本实际暴露统计.json').write_text(json.dumps(exposure,ensure_ascii=False,indent=2),encoding='utf8')

def metric_row(label, role, metric):
    return f"| {label} | {role} | {100*metric['Acc7']:.4f} | {100*metric['Acc2']:.4f} | {100*metric['F1']:.4f} | {metric['MAE']:.6f} | {metric['Corr']:.6f} |"

md = ['# 多模态流方法与 CaReFlow：实验结果、退化诊断及升级建议', '',
      f'报告日期：2026年10月7日（北京时间）。证据汇总时刻：{clock}。', '',
      '当前结论：已完成所有指定固定模型的 VAL 和 TEST 五项评价。官方划分、共同训练预算的固定 F 与固定 CaReFlow 比较中，F 在 TEST 的五项指标均落后。旧模型 VAL 约0.59的 MAE是真实结果，但 TEST 为0.643–0.646；新模型在合并数据后原TEST角色上的0.544不是保留测试集结果。尚未证明五项全面超过 CaReFlow，完整视频五折尚未完成。', '',
      '## 1. 比较协议与指标口径', '',
      '正式固定比较使用作者公共 MOSI 资产：TRAIN1281条、VAL（DEV）229条、TEST685条。双方使用相同样本身份、标签尺度、seed128、100轮、batch32及固定训练订单；每轮丢弃尾部1条，共4000次更新。VAL按128/101两个批次的MSE算术均值选一个checkpoint，严格最小、平局取最早；F选第89轮，CaReFlow选第93轮。测试五项共用这个模型，不为每个指标分别选模型。', '',
      '这是本项目的固定配置复现比较；不声称与论文原训练配方、全部资产清洗版本或多seed平均完全等价。双方网络和辅助目标仍不同，因此该比较不能单独定位某个组件的因果作用。', '',
      'Acc7：预测与真值clip到[-3,3]后round，使用全部样本。Acc2/F1：排除真值0，以预测≥0判正，F1使用类别支持数加权。MAE/Pearson Corr：使用全部原值。VAL的Acc2/F1为216条，TEST为655条。前三项下表以百分比表示；MAE越低越好，其余越高越好。', '',
      '## 2. 官方划分下已经测出的 VAL / TEST 分数', '',
      '| 模型 | 集合 | Acc7 % | Acc2 % | F1 % | MAE | Corr |',
      '|---|---|---:|---:|---:|---:|---:|']
for key,label in [('careflow','CaReFlow best93'),('minimal_fixed_F','我们固定 F best89')]:
    for role in ('VAL','TEST'): md.append(metric_row(label,role,models[key][role]))
f,c=models['minimal_fixed_F']['TEST'],models['careflow']['TEST']
md += ['', f"F的TEST MAE高{f['MAE']-c['MAE']:.6f}，Acc7低{100*(c['Acc7']-f['Acc7']):.4f}个百分点；Acc2、F1、Corr也均低。当前固定版本未达到目标。这些TEST分数已经实际计算并保存，不再是‘预测完成但未评分’。", '',
       '旧 A/B/C2 使用此前开发得到的模型，容量、冻结范围、seed及教师成本与上述正式比较没有完全匹配。其实际分数如下；可作历史结果对照，不能充当受控消融。', '',
       '| 模型 | 集合 | Acc7 % | Acc2 % | F1 % | MAE | Corr |',
       '|---|---|---:|---:|---:|---:|---:|']
for key,label in [('old_A','旧 A best40'),('old_B','旧 B best11'),('old_C2','旧 C2 best37')]:
    for role in ('VAL','TEST'):md.append(metric_row(label,role,models[key][role]))
md += ['', '旧 A/C2 的 TEST Acc2、F1高于当前固定 CaReFlow，但 Acc7、MAE、Corr更差；没有一个旧模型同时赢五项。旧0.598的意义是开发集上减少了一部分误差，并未证明测试泛化提升。另一个0.591578来自201条开发子集，也不是完整VAL229或TEST685，不能连接成同一条优化曲线。', '',
       '## 3. 新方案与“合并数据”的含义', '',
       '按用户指定，将原TRAIN、VAL、TEST合并为2195条、93个视频，再按视频划分五折。每折的视频训练集与外折测试集隔离；训练部分再拆FIT和INNER，INNER用于选模。这种协议下，原官方TEST不再整体保留，必须以每条样本未参与其对应模型训练或选模的外折预测汇总OOF指标。', '',
       '目前完成的是新 anchored_flow 的第0折40轮开发pilot：FIT1494、INNER264、OUTER437，1880次更新；预定INNER MSE选择第20轮。选中模型INNER五项为 Acc7 48.1061%、Acc2 86.4000%、F1 86.3860%、MAE 0.679452、Corr 0.818677。', '',
       '该模型重新按原角色统计得到VAL MAE0.542107、TEST MAE0.544225，但原VAL含FIT135/INNER34/OUTER60条，原TEST含FIT436/INNER88/OUTER161条。因此这些汇总包含训练和选模样本，不能作为独立TEST胜出证据。部分OUTER标签已进入用户要求的原角色描述性评分；全部437条外折汇总及完整五折仍未完成。', '',
       '## 4. 强情绪与弱情绪的实际比例', '',
       '固定定义：弱情绪为 |y|≤1，包括中性0和边界±1；强情绪为 |y|>1。以下来自已保存的原标签数组，双方正式比较的数据分布相同。', '',
       '| 数据角色 | 总条数 | 弱情绪条数 | 弱占比 | 强情绪条数 | 强占比 | 中性0条数 |',
       '|---|---:|---:|---:|---:|---:|---:|']
for key,label in [('careflow_TRAIN','正式TRAIN'),('careflow_VAL','正式VAL'),('careflow_TEST','正式TEST'),('pilot_fit','新pilot FIT'),('pilot_inner','新pilot INNER')]:
    v=counts['counts'][key]
    md.append(f"| {label} | {v['rows']} | {v['weak_abs_le1']} | {v['weak_percent']:.2f}% | {v['strong_abs_gt1']} | {v['strong_percent']:.2f}% | {v['neutral_y0']} |")
md += ['',f"核对实际训练订单后，4000个batch共{exposure['total_row_exposures']}次样本暴露，其中弱{exposure['weak_row_exposures']}次（{exposure['weak_percent']:.4f}%），强{exposure['strong_row_exposures']}次。两方法完全一致；反复训练的暴露次数不是独立样本数。", '',
       '比例约为强60%、弱40%，不是极端不平衡。强弱比例变化不能单独解释同一数据上F与CaReFlow的差距；MSE的梯度贡献还依赖实际误差。后验标签统计还显示，非零样本中正情绪占比TRAIN55.13%、VAL57.41%、TEST42.29%，存在符号分布变化，但尚未证明它导致了测试损失。', '',
       '## 5. 退化集中在哪里，哪些解释有证据', '',
       '| 正式VAL诊断 | 固定 F | CaReFlow |',
       '|---|---:|---:|',
       '| 弱情绪84条MAE | 0.729502 | 0.659727 |',
       '| 强情绪145条MAE | 0.558162 | 0.572544 |',
       '| 全体预测均值减真值均值 | +0.089870 | −0.012549 |',
       '| 弱情绪预测标准差 | 1.099953 | 1.018466 |',
       '| 弱情绪非零样本符号错误 | 21 | 18 |', '',
       '弱情绪真值标准差只有0.581583；F在弱样本上更容易放大幅度、偏正并产生符号错误。按条数拆解，弱情绪损失使整体MAE增加约0.02559，强情绪改善抵消约0.00911，净增加0.01649。已完成新pilot中，流修正也表现为帮助强情绪、损害弱情绪，且FIT和INNER同时存在这一现象；不能只归因于过拟合。', '',
       '这些是保存预测可确认的现象。尚无受控消融证明是流路径、辅助损失、归一化、骨干更新或反馈中的哪项导致退化。两方法训练总objective含不同辅助项，其数值大小不能直接证明欠拟合或过拟合。新模型的流前b也不是另行训练的无流基线；负的全局gain同样不是退化原因的证明。', '',
       '## 6. 已验证的直接修正与待验证的训练候选', '',
       '已经做过一个固定推理门控对照：q=b+[b²/(1+b²)](p−b)。b是流前预测，p是原最终预测，门控只读模型输出。固定尺度1，未扫描阈值或参数。', '',
       '| 新pilot INNER264 | 原最终p | 门控q |',
       '|---|---:|---:|',
       '| 弱情绪111条MAE | 0.638971 | 0.557799 |',
       '| 强情绪153条MAE | 0.708821 | 0.815040 |',
       '| 全体MAE | 0.679452 | 0.706882 |',
       '| Acc7 | 48.1061% | 42.0455% |',
       '| Acc2 | 86.4000% | 86.0000% |',
       '| F1 | 86.3860% | 85.9809% |',
       '| Corr | 0.818677 | 0.819591 |', '',
       '结果：弱情绪改善被强情绪损失超过，五项中四项变差，拒绝这个固定门控。强情绪中51/153条的|b|也≤1，说明预测幅度不能可靠区分真实强弱；简单缩小修正会删掉对强情绪有用的变化。没有为这个门控补测TEST或扫描尺度。', '',
       '现有单损失候选已完成源码准备，尚未训练：在FIT弱情绪样本上增加 mean ReLU((p−y)²−(stop_gradient(b)−y)²)，系数固定1，原推理路径不变。它只惩罚相对b增大误差的修正；真标签仅在训练损失中使用。stop_gradient仅阻断该损失对比较项的直接梯度，并没有冻结b对应的网络；b仍可能随共同训练漂移。有限惩罚不能保证弱样本泛化不退化，也不能声称已获得新VAL/TEST收益。原服务器Torch合成验证尚未完成。', '',
       '## 7. 从流的角度升级：建议优先验证什么', '',
       'CaReFlow原论文通过跨模态分布映射、基于训练标签差异的自适应对齐以及循环重建来减少模态差异、保留模态信息。这为路径与信息保持设计提供依据，但模态对齐和重建本身不保证弱情绪预测风险降低。[CaReFlow原论文](https://arxiv.org/html/2602.19140v1)', '',
       '本项目新pilot使用速度场和两步Euler特征传输、供体反馈及流前预测锚，但其当前训练目标是任务损失，已移除旧FM/cycle项；不能仅因存在速度场就把它等同于论文的rectified flow matching。建议将下一次结构升级明确为“对情绪任务友好的流路径”，而不是只改变最终输出幅度。以下仍是方法设计，未实现或验证收益。', '',
       '1. **优先：路径中的情绪保持监督。** 对已有中间态和终态使用同一情绪读出，训练时检查弱样本的误差、近零符号及跨阈值变化；将约束施加到速度场走过的路径，而不只在终点收缩输出。比较基准需要避免随共同训练漂移；若使用独立冻结参考，它必须只用该折FIT训练，并把额外教师成本计入双方预算。此设计旨在保留强样本的有效位移、限制弱样本的有害位移，尚不是风险保证。', '',
       '2. **后续可考虑：条件速度场。** 让速度依赖各模态一致性、可靠性和流时间，学习何时以及往哪个方向修正。推理不输入真实强弱标签。当前所有样本共享一个gain的做法不足以表达样本之间的修正需求，但需真实受控比较才能确认条件速度有用。', '',
       '3. **辅助替代：训练内强弱组平衡和有序分类约束。** 仅在FIT固定分组均衡损失贡献，同时用近零符号/情绪等级阈值约束预测。它能直接对准弱样本和Acc2/Acc7的敏感位置，但可能损害强样本或MAE；VAL与TEST保持原分布，禁止用其重采样或逐项拼模型。', '',
       '实施顺序建议：先验证已准备的单损失候选及比较项漂移问题；后续只固定一套路径升级方法，与同一CaReFlow在相同数据、订单、预算、选模规则下比较。每个训练完成的新checkpoint都报告VAL和TEST五项，并补训练/验证强弱分组诊断；方法和系数的开发依据保持为TRAIN/VAL，不能根据TEST挑结构。既有TEST已被查看，后续报告应保留这一历史，不冒全新的独立确认。', '',
       '## 8. 当前完成范围与请同行重点审阅的问题', '',
       '已完成：正式双方100轮训练和VAL/TEST五项、旧A/B/C2的VAL/TEST五项、新第0折40轮pilot、强弱比例统计、固定弱门控负结果、完成模型与训练全状态保存。三机到期前动态快照已在实际13:07UTC采集并通过D盘核验，没有回填为13:00。', '',
       '未完成：弱情绪新损失训练、新路径升级实验、双方完整视频五折OOF、MOSEI、多seed稳定性，以及五项全面超过CaReFlow。当前剩余租期和约2.45GB本地保存空间不足新完整训练状态及原定保存余量；源码准备不能算新训练成功。', '',
       '请同行优先评估：弱样本幅度放大/偏正是否应通过路径监督解决；如何构造合法且不漂移的比较参考；条件速度场是否比预测幅度门控合理；如何同时保护MAE和近零符号/等级；哪一项最小受控比较能区分优化目标问题与流结构问题。', '',
       '## 附：结果来源', '',
       '完整未舍入指标与诊断见随报告ZIP中的JSON。报告只汇总既有结果并新增训练订单暴露统计，没有重跑旧训练、模型预测或TEST评分。', '',
       '- 所有固定模型VAL_TEST五项实际对齐结果.json',
       '- 固定双方TRAIN_DEV退化诊断.json',
       '- 新方案40轮实际完成与完整保存接续.json',
       '- 强弱情绪比例原标签实际统计.json',
       '- 正式训练强弱样本实际暴露统计.json',
       '- 弱情绪直接修正对照与新训练方案实际结果.json',
       '- 完整流匹配U_W头201一次开发五指标实际结果.json（201仅开发子集）',
       '- 候选源码仅含anchored_flow.py和candidate_protocol.json，明确执行disabled、未训练。', '']

report_path = out / '多模态流与CaReFlow实验总结报告_20261007.md'
report_path.write_text('\n'.join(md),encoding='utf8')
stamp=clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
dest=base/('share_research_report_'+stamp)
dest.mkdir()
evidence=[results_path,counts_path,diagnosis_path,out/'新方案40轮实际完成与完整保存接续.json',
    out/'正式训练强弱样本实际暴露统计.json',out/'弱情绪直接修正对照与新训练方案实际结果.json',
    out/'完整流匹配U_W头201一次开发五指标实际结果.json',out/'第二租期13点真实动态保存实际接续.json']
candidate=base/'weak_retention_training_candidate_20261007T124321Z/source'
for p in [report_path,Path(__file__),*evidence]:shutil.copy2(p,dest/p.name)
(dest/'candidate_source').mkdir()
for name in ('anchored_flow.py','candidate_protocol.json'):shutil.copy2(candidate/name,dest/'candidate_source'/name)
manifest=dict(actualclock_UTC=clock,report_source='saved_results_no_new_scoring',
    member_sha256={p.relative_to(dest).as_posix():sha(p) for p in dest.rglob('*') if p.is_file()})
(dest/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
archive=out/'多模态流与CaReFlow总结报告及原始指标_20261007.zip'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for p in sorted(dest.rglob('*')):
        if p.is_file():z.write(p,p.relative_to(dest).as_posix())
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,h in manifest['member_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
shutil.copy2(archive,dest/archive.name)
print(json.dumps(dict(report=str(report_path),archive=str(archive),D=str(dest),report_SHA=sha(report_path),zip_SHA=sha(archive),exposure=exposure),ensure_ascii=False))
