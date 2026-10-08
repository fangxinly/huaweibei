"""Preserve actual native evidence and a bounded next-stage development plan."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
p=argparse.ArgumentParser();p.add_argument('--actualclock',required=True);a=p.parse_args()
ws=Path(__file__).resolve().parent.parent;v=read(ws/'work/residual_direction_v2_current.json');d=Path(v['D'])
native={n:read(d/(n+'_native')/'actual_D_verification.json') for n in ('A','B')}
for n,q in native.items():
    assert q['all_original_SHA_CRC_unique'] and q['original_natural_exit']==0
    assert q['result']['original_full_flow_maxerror']==0 and q['result']['global_state_RNG_unchanged']
closure=read(ws/'work/residual_direction_v2_closed_20261008T114758Z.json')
assert all(c['result']['status']=='fulfilled' and c['result']['value']['exit_code']==0 for c in closure['sessions'])
resource=dict(actualclock_UTC=a.actualclock,D_free=shutil.disk_usage(d).free,C_free=shutil.disk_usage(ws).free,
    conservative_lease_end_UTC='2026-10-08 14:00 UTC',platform_confirmed=False,
    required_save_reserve_seconds=7200,new_complete_crossfit_execution_budget_qualified=False)
report=ws/'outputs/下一步原流方向残差控制与实际原机验证.md'
text='''下一阶段要回答的是：在相同流和读出上，真正排除留出视频得到的残差，能否帮助逐样本选择有用的消息方向。

最近官方对齐实验 TEST MAE 0.650616，CaReFlow 0.619535，五项点值均差。此前真实新息原型残差估计器元训练相关0.4946、CAL仅0.0043，校准斜率0.0180，已经说明复杂门容易记住元训练数据。下一版采用低容量估计、完整前驱视频隔离，而非继续扩大这两个失败版本。

这次已经完成的实现与实际检查：

1. 保留原100维、两步Euler、reader、解码器和六方向消息。第一步状态和读出共享，消息开关在第二步context注入前生效；p0严格为同流全关消息，原始池化b不再冒充p0。
2. 固定八种干预：全关、六个单方向、全开。每种干预独立计算完整非线性流输出；不把单方向Δ相加作为全开预测。
3. 固定20维输入：第一步三个模态slots的公共随机投影12维、p0一维、七个真实干预Δ。使用含截距的21参数视频等权ridge，归一化人口目标λ=1，无参数扫描。它是候选容量选择，尚无真实数据收益。
4. 监督目标为ρ=y−p0，数据必须来自视频隔离的OOF前驱。编码器、归一化、流、消息、特征变换都要排除该行视频和独立CAL视频；仅把头称OOF而沿用见过所有TRAIN视频的任务编码器，会被拒绝。JSON集合检查不能证明外部来源真实性，真实运行仍需原SHA、argv与capture联结。
5. 在预先分离的TRAIN内CAL视频、且最终候选前驱同样排除CAL的条件下拟合非负斜率s。用2sρhatΔ−Δ²选干预，平局取全关；未拟合、未校准或s=0时严格退回p0。非负预测效用不是真实风险界，也不能保证五指标改善。

A/B原Torch2.1.0+cu121 CPU合成检查均自然0，A child3302、B child38866；原件完整D保存，SHA/ZIP CRC/唯一成员/源argv/PID/UUID/预算与runtime核过。全开与原流最大误差0，效用恒等式误差5.55e−17，dummy标签、padding、批次排列和参数/RNG不变检查通过；合成的编码器泄漏、归一化CAL泄漏、视频拆行和角色越界输入被拒绝。四个新SSH/SFTP均明确exit/bye0关闭。此次没有真实新OOF训练、编码器/任务权重前向或新VAL/TEST分数。

下一步真实实验按以下次序落实：

1. 仅在官方TRAIN内预先按视频分成拟合、CAL和机制确认角色；拟合角色再做完整前驱的五折OOF。确认视频不得进入统计、基线、流、控制器或校准。已训练于全TRAIN的当前任务检查点不能作为这些留出视频的OOF前驱。
2. 先核消息关闭基线的留出五项与逐视频误差，再核残差相关、校准、预测效用与实际效用；同时报告相对仅依赖p0的标量重标定对照，避免把重标定当消息收益。普通端到端门必须作为对照；它的同输入、同干预与容量匹配实现及预算仍需在真实执行前冻结，本次未冒充已完成该对照。
3. 固定方法后才进入新的CaReFlow配对实验。双方实际训练行、订单、更新、选模、CAL用途和五指标必须共同冻结。上面机制实验因保留TRAIN内CAL而少用部分训练行，不能直接借用全TRAIN CaReFlow成绩声称公平胜出；它也不是用户此前要求的全数据合并五折。两类完整五折均未完成。
4. 继续使用同一检查点报告VAL/TEST全部五项；历史TEST已访问，这些点值无法恢复成新盲测，也不得拿来反复选门、特征或模型。

当前D仅约0.35GB可用，保守租期Oct8 UTC14:00/BJ22:00，平台未确认；完整任务模型视频OOF的训练状态、队列耗时和至少2小时保存余量尚未通过。因此本轮没有启动新的完整训练。冻结编码器的小机制模型与原完整任务微调预算不同，若采用必须明确标成机制验证，不能再次冒充官方性能升级。下一完整训练须先落实可信执行预算与长期保存容量。
'''
text+=f'\n实际接续时钟：{a.actualclock}。源码冻结计划SHA {v["plan_SHA"]}；源码ZIP SHA {v["ZIP_SHA"]}。\n'
report.write_text(text,encoding='utf8')
value=dict(status='CORE_DIRECTION_INTERFACE_AND_LOW_CAPACITY_OOF_SELECTOR_NATIVE_COMPLETE_REAL_CROSSFIT_PENDING',actualclock_UTC=a.actualclock,
    source=v,native=native,report=str(report),report_SHA=sha(report),resources=resource,client_closure=closure,
    real_OOF_training_complete=False,new_VAL_TEST_scores=False,ordinary_capacity_matched_control_complete=False,
    full_OFFICIAL_TRAIN_internal_fivefold_complete=False,full_merged_dataset_fivefold_complete=False,whole_goal_complete=False)
control=ws/'outputs/下一步原流方向残差控制实际接续.json';control.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
saved=d/'final_delivery';saved.mkdir()
for f in (report,control,Path(__file__),ws/'work/residual_direction_v2_closed_20261008T114758Z.json'):shutil.copy2(f,saved/f.name)
members={f.name:sha(f) for f in saved.iterdir() if f.is_file()}
zp=d/'complete_delivery.zip'
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED) as z:
    for n in members:z.write(saved/n,n)
with zipfile.ZipFile(zp) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
joint=dict(actualclock_UTC=a.actualclock,source_ZIP_SHA=v['ZIP_SHA'],plan_SHA=v['plan_SHA'],native=native,
    delivery_ZIP_SHA=sha(zp),members=members,all_SHA_CRC_unique=True,real_OOF_training_complete=False,whole_goal_complete=False)
raw=json.dumps(joint,ensure_ascii=False,indent=2).encode('utf8');(d/'actual_joint.json').write_bytes(raw);jointsha=hashlib.sha256(raw).hexdigest()
state=ws/'outputs/研究接续状态.md'
short=f'最新实际 {a.actualclock}：先读《下一步原流方向残差控制与实际原机验证.md/json接续》。原100维两Euler六方向开关+同流p0+8个真实非线性干预、20维/21参数视频等权ridge λ1、完整编码器等前驱OOF与独立CAL守卫、效用mask选择已源码冻结native计划{v["plan_SHA"]}，A3302/B38866原TorchCPU合成均自然0，全开与原流误差0/状态RNG不变/泄漏fixture拒绝，双原capture完整D SHA CRC唯一源argv/PID/UUID/余量通过，joint{jointsha}。仅合成代码资格，无真实新OOF训练/新VALTEST/完整效用收益或风险界；普通容量匹配对照仍待冻结。不得重复旧100/FIT232/201评分或旧失败新息矩阵，不据TEST救分。完整OOF新任务编码器需排除留出视频，旧全TRAIN任务模型不能冒OOF；当前D约0.35GB与保守UTC14:00剩余不足完整队列+2h保存，未启动。B52751/A71900/SFTP97156/77649全部明确exit/bye0，C未fresh，无续租/删原件，原10min保持。上一官方TEST负结果.650616/C.619535完整保存不抹；完整合并五折/正式超过/整体目标未完成。以下历史。\n\n'
state.write_text(short+state.read_text(encoding='utf8'),encoding='utf8')
print(json.dumps(dict(report=str(report),control=str(control),joint=str(d/'actual_joint.json'),joint_SHA=jointsha,new_real_scores=False),ensure_ascii=False))
