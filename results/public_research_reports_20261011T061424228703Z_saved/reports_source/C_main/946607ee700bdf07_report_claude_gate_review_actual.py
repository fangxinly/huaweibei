import argparse,datetime,hashlib,json,shutil,zipfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--clock',required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
D=Path('D:/CodexBackups/selective_flow_20261003_1105');out=Path('outputs')
gate=Path(read('work/inner_lovo_output_gate_current.json')['D']);gr=read(gate/'actual/actual_stage_receipt.json')
donor=Path(read('work/fixed_weak_donor_mask_current.json')['D']);dr=read(donor/'A_original_capsule/original/actual_stage_receipt.json')
train=D/'weak_retention_train40_20261007T145201Z';tr=read(train/'A_original/out/actual_stage_receipt.json');joint=read(train/'training_D_B_CPU_joint.json');assert joint['status']=='PILOT40_D_ORIGINAL_CPU_CAPTURE_COMPLETE'
assert read(donor/'A_actual_D_audit.json')['natural_exit']==read(donor/'B_actual_D_audit.json')['natural_exit']==0
br=read(donor/'B_original_capsule/original/actual_stage_receipt.json');assert br['saved_metrics_exact_equal'] and br['original_receipt_sha256']==sha(donor/'A_original_capsule/original/actual_stage_receipt.json')
boot=Path(read('work/formal_video_paired_bootstrap_current.json')['D']);bur=read(boot/'actual/actual_stage_receipt.json')
record=dict(status='ACTUAL_CLAUDE_SUGGESTION_OUTPUT_GATE_DONOR_MASK_WEAK_TRAIN_FORMAL_BOOTSTRAP_D_SAVED',actual_record_UTC=a.clock,
 output_gate=gr,donor_mask=dr,donor_native_CPU=br,formal_fixed_video_bootstrap=bur,
 weak_train_complete=dict(original=tr,D_B_CPU_joint_SHA=sha(train/'training_D_B_CPU_joint.json')),
 artifacts={k:dict(D=str(path),sha256=sha(path)) for k,path in [('output_gate_capture',gate/'complete_local_CPU_capture.zip'),('donor_A_capture',donor/'A_complete_capture.zip'),('donor_B_capture',donor/'B_complete_capture.zip'),('formal_bootstrap_capture',boot/'complete_local_CPU_capture.zip')]},
 overall_all5_superiority_or_full5fold_complete=False,
 decisions={'oracle_diagnostic':'Completed; keep finite fixed-prediction scope.',
 'low_dimensional_gate':'Reject this fixed three-feature ridge.01 LOVO gate on explored INNER; no feature/lambda sweep or deployment.',
 'MC_dropout':'Deferred; signal-to-noise shrinkage heuristic is not a calibrated risk guarantee and does not remove stable bias.',
 'loss_geometry':'Separate prospective Huber delta1 ablation is a candidate, not implemented/trained; existing objective already includes .25 L1 and .5 source MSE, so result would support rather than identify a sole cause.',
 'new_controller_inputs':'Prioritize actual same-flow off/on and single-direction state information for video-OOF residual supervision; first mechanism prototype can freeze public encoder features. Not implemented residual controller yet.',
 'weak_harm_loss':'Already completed one matched40 candidate, no expansion; FIT weak harm largely removed but INNER weak harm remains. Does not settle OOF-comparator question.'},
 source_limitations=['Claude referenced /mnt script was not available in this Windows workspace; own fixed implementation used three stated features, equal-video weights and ridge.01, no hyperparameter search.',
 'LOVO excludes each heldout video from gate fitting, but checkpoint was selected using every INNER video; not nested independent OOF.',
 'Output b/p oracle selected20 and donor off/on oracle selected36 belong to different fixed models and prediction contrasts; do not equate them.',
 'Inference encoder caching does not mean encoders were frozen in training; all347 parameter tensors were trainable.',
 'Donor feedback masking retains crossmodal reader/role head and same Euler/readout, and probes a model trained with messages; functional intervention only.',
 'Merged fold contains135 original VAL and436 original TEST training rows; descriptive original-role scores cannot replace official retained TEST scores.',
 'All bootstrap intervals condition on fixed predictions, training seeds and selection. No simultaneous five-metric significance or posterior claim.'],
 storage=dict(D_free_bytes=shutil.disk_usage(D).free,no_additional_frozen_files_deleted=True,conservative_lease_end_UTC='2026-10-08T14:00:00+00:00',platform_expiry_verified=False))
name='Claude建议核验与残差消息诊断实际结果_20261008';jp=out/(name+'.json');mp=out/(name+'.md');assert not jp.exists() and not mp.exists();write(jp,record)
lines=['# Claude建议核验与实际诊断', '',f'实际记录：{a.clock}。以下均来自保存预测或本轮固定模型执行，非合成数据效果。', '',
 '结论：诊断优先级合理。现有输出幅度与符号三特征门失败；保持损失单候选带来开发集改善；真实 donor 反馈仍伤害整体 MAE/MSE。应先补实任务残差监督与同流消息控制，而不是继续扫幅度门或直接扩大训练。', '',
 '## 1. 已保存 selected20 的输出层门诊断', '',
 'INNER 264条、11视频。特征为 |b|、|p−b|、sign(b)sign(p−b)，训练视频等权、视频内行等权；标准化只用本轮留一训练视频，ridge=.01，仅惩罚三斜率，截距自由。每个视频只接受其他视频拟合的门；没有搜索读出/特征/正则。原 checkpoint 已由全部 INNER 选出，因此这是探索诊断，不能称新独立交叉验证。', '',
 '|固定预测/门|整体MAE|弱MAE|强MAE|整体MSE|','|---|---:|---:|---:|---:|']
s=gr['inner']['summary']
for label,key in [('原始b','b'),('原始p','p'),('事后连续最优门','continuous_oracle')]:lines.append(f"|{label}|{s['all'][key]['MAE']:.6f}|{s['weak'][key]['MAE']:.6f}|{s['strong'][key]['MAE']:.6f}|{s['all'][key]['MSE']:.6f}|")
for label,key in [('留一常数门','constant'),('留一特征门','feature')]:
 v=gr['inner']['gate_scores'][key];lines.append(f"|{label}|{v['all']['MAE']:.6f}|{v['weak']['MAE']:.6f}|{v['strong']['MAE']:.6f}|{v['all']['MSE']:.6f}|")
lines+=['',f"事后连续门 MSE 从 {s['all']['p']['MSE']:.6f} 降至 {s['all']['continuous_oracle']['MSE']:.6f}，有限样本降幅18.781%。它用真标签，只是这组输出线段内的事后空间，不是可实现收益、全部消息控制上界或泛化界。弱/强 corr(Δ,ρ) 分别 {s['weak']['corr_delta_residual']:.6f}/{s['strong']['corr_delta_residual']:.6f}，U>0比例26.13%/86.93%。",'',
 '特征门整体 MAE−p=+0.009114，10000次按视频配对 bootstrap 条件95%区间[+0.002892,+0.016752]；弱样本略好、强样本变差。区间不重新拟合门，也不重新选择 epoch，不能消除选模影响。停止此固定三特征门，不选201/TEST读出救分。', '',
 '## 2. 弱情绪保持损失40轮与同流 donor 控制', '',
 '新旧40轮具有相同分组、初态、随机状态、1880订单与更新预算、INNER MSE选模规则；只增加固定系数1的 FIT弱样本相对 stopgrad(b) 平方伤害惩罚。新best36相对旧best20，INNER五项点值均改善：Acc7 48.1061%→50.3788%，Acc2 86.4%→86.8%，F1 86.3860%→86.8057%，MAE .679452→.640391，Corr .818677→.841168。两者都是开发选择结果，非新保留TEST或完整五折。', '',
 '弱MAE .638971→.601410；强MAE .708821→.668671，两组都改善。FIT弱样本 b/p=.117906/.115053，但INNER b/p=.484453/.601410：保持损失没有消除留出视频上的弱样本伤害。也不能把源b漂移排除为原因。', '',
 '同一个固定best36、同一编码器/两步Euler/decoder/全局gain：仅把六个donor反馈MLP输出乘固定0或1。p0仍保留 reader 与 role head，因此只判断这些 donor 反馈的功能效应，不能代表所有跨模态路径或真实语义因果。', '',
 '|同流固定mask|整体MAE|弱MAE|强MAE|整体MSE|','|---|---:|---:|---:|---:|']
m=dr['donor_mask_diagnostic']['inner']['regions']
for label,key in [('六方向全关p0','p0'),('六方向全开p1','p1')]:lines.append(f"|{label}|{m['all'][key]['MAE']:.6f}|{m['weak'][key]['MAE']:.6f}|{m['strong'][key]['MAE']:.6f}|{m['all'][key]['MSE']:.6f}|")
lines+=['',
 '开启 donor 后 Acc7略差、Acc2/F1略好、Corr近乎相同，不能按 MAE 选择部署读出。INNER二选一标签oracle至多在本固定预测上降低p1 MSE约4.954%；这与上节 b/p 的18.781%不是同一模型或同一对照。六单方向完整五项表见JSON；单方向效果不可直接相加，输出加性闭合残差RMS=.007117。', '',
 f"当前全局 gain=tanh(parameter)={dr['gain_tanh']:.9f}，p=b+gain(f−b) 是反向外插，并非 b 与 f 的凸组合。训练编码器全部微调；本诊断只复用每批冻结检查点的推理激活，all-on严格重放原预测、dummy0/7不变、全state/RNG/stat不变。A882自然0，B23719只复核保存数组自然0，没有CPU模型前向。完整状态2966459315字节已D/B原CPU核过，347Adam状态/1880步、40历史/选择误差0。", '',
 '## 3. 固定正式VAL/TEST的按视频配对区间', '',
 '固定正式 F best89 与 CaReFlow best93，双方官方TRAIN训练、VAL选模、原TEST685已一次评分。此次只使用已存预测；VAL10视频、TEST31视频。10000次整视频成对重采样，保留每视频所有行，五项共享同一抽样，沿用官方逐片段指标。', '',
 '|TEST指标|F−CaReFlow点差|条件95%区间|','|---|---:|---:|']
for metric in ['Acc7','Acc2','F1','MAE','Corr']:
 v=bur['roles']['test']['F_minus_C'][metric];scale=100 if metric in ['Acc7','Acc2','F1'] else 1;unit='个百分点' if scale==100 else ''
 lines.append(f"|{metric}|{v['point']*scale:+.6f}{unit}|[{v['percentile95'][0]*scale:+.6f}, {v['percentile95'][1]*scale:+.6f}]|")
lines+=['',
 '正式 F TEST MAE=.643699787，CaReFlow=.619535294，差+.024164494。Acc7/MAE/Corr各自条件区间未跨0；Acc2/F1跨0。VAL五个区间均跨0。Acc2/F1抽样差高度相关；不能把五项当五个独立证据，或宣称五项同时显著/稳定多种子。区间不包含训练随机性或epoch选择不确定性，不修正VAL赢家诅咒。', '',
 '新合并训练模型的原角色 VAL/TEST五项也已完成，完整数值见JSON，但TRAIN含135条原VAL和436条原TEST；即使原TEST MAE=.403923，也不能替代上述正式保留TEST成绩。', '',
 '## 4. 后续方向与边界', '',
 '优先设计冻结公共编码特征的机制实验：同流 p0/p1，按视频隔离折外残差监督，输出层校准，再与同容量、只用任务损失的普通学习门比较。必须正视OOF模型到最终模型的错配；仅主张逐样本/逐方向，当前单次注入不主张时间自适应。机制源码尚未实现，不冒核心创新已落地。', '',
 'Huber(δ=1)可作为独立单变量候选；现目标已有 .25 L1 与 .5源MSE，不能说它是纯MSE实验，也不能因弱伤害消失就认定唯一原因。MC-dropout缩放是启发式，样本间预测方差不等于MC不确定性，稳定偏差可能完全保留；先不启动多种子或dropout矩阵。当前弱保持单候选结果保留，不继续扩容；若再研究比较损失，优先折外比较对象。', '',
 '剩余D空间约2.6GB，不具备再保存一份约3GB完整新状态及余量；本轮没有再删除冻结资料。整体五项超过CaReFlow、完整五折均未完成。', '',
 f"逐样本CSV：{gate/'actual/inner_y_b_p_video.csv'}。完整源/参数协议/原预测/自然退出/ZIP SHA/CRC均在JSON指向的D证据目录。"]
mp.write_text('\n'.join(lines)+'\n',encoding='utf8')
# Dynamic pointers only; every earlier frozen scientific original remains unchanged.
prefix=f"最新实际 {a.clock}：先读《{name}.md/json》。弱保持40轮1880步/选36已自然0、完整2966459315B D+异节点B原CPU347Adam/40历史选择联合通过；INNER五项相对同预算旧20均点值改善、MAE .679452→.640391。输出三特征LOVO门实际变差、全体MAE+.009114；同流donor全关/全开INNER MAE .632093/.640391，弱伤害仍在，原始b不是p0。A882/B23719自然0/双真实capture全SHA CRC唯一/source argv过；B只保存数组核验非模型前向。正式固定VAL10/TEST31视频10000配对bootstrap已D全原件，TEST MAE差CI[+.004720,+.043621]；非多种子/同时五项显著。原角色合并训练TEST含FIT436不能冒独立。全五指标超过与完整五折未完成；D约2.6GB，未额外删原件；仅新三机/保守Oct8UTC14:00平台未核。以下历史。\n\n"
state=out/'研究接续状态.md';state.write_text(prefix+state.read_text(encoding='utf8'),encoding='utf8')
activation=out/'第三租期恢复与弱情绪损失原生检查实际接续.json';x=read(activation);x['status']='THIRD_LEASE_WEAK_TRAIN40_COMPLETE_D_B_DONOR_DIAGNOSTIC_FORMAL_BOOTSTRAP_COMPLETE';x['latest_actual_record_UTC']=a.clock;x['latest_result_json']=str(jp.resolve());x['latest_training_complete_D_B_CPU_joint']=str(train/'training_D_B_CPU_joint.json');write(activation,x)
stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z';seal=D/('claude_gate_review_actual_'+stamp);assert not seal.exists();seal.mkdir()
for source in [mp,jp,Path(__file__),Path('work/audit_fixed_donor_mask_capsule_D.py')]:shutil.copy2(source,seal/source.name)
write(seal/'manifest.json',dict(actual_UTC=a.clock,member_sha256={q.name:sha(q) for q in seal.iterdir() if q.is_file()}))
with zipfile.ZipFile(seal/'complete_report_source_refs.zip','x',zipfile.ZIP_DEFLATED) as z:
 for q in seal.iterdir():
  if q.is_file() and q.suffix!='.zip':z.write(q,q.name)
with zipfile.ZipFile(seal/'complete_report_source_refs.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
write(out/'残差消息最新诊断接续.json',dict(status=record['status'],actual_UTC=a.clock,report=str(mp.resolve()),result=str(jp.resolve()),D_seal=str(seal),seal_SHA=sha(seal/'complete_report_source_refs.zip'),overall_complete=False))
print(json.dumps(dict(report=str(mp.resolve()),result=str(jp.resolve()),D=str(seal),seal_SHA=sha(seal/'complete_report_source_refs.zip'))))
