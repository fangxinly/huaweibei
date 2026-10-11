"""Seal completed official comparison, clear stale control state, retain original evidence."""
import argparse, hashlib, json, shutil, zipfile
from pathlib import Path
import numpy as np

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def write(p,v):
    with Path(p).open('x',encoding='utf8') as f: json.dump(v,f,ensure_ascii=False,indent=2)

p=argparse.ArgumentParser();p.add_argument('--actualclock',required=True);a=p.parse_args()
ws=Path(__file__).resolve().parent.parent
pointer=read(ws/'work/official_upgrade_pointer.json');d=Path(pointer['D'])
seal=read(d/'final_report_seal.json');report=read(seal['JSON'])
assert sha(seal['report'])==seal['report_SHA'] and sha(seal['JSON'])==seal['JSON_SHA']
assert sha(d/'final_report_complete.zip')==seal['ZIP_SHA']
stamp=a.actualclock.replace('-','').replace(':','').replace(' UTC','Z').replace(' ','T')
local=ws/'outputs'/('official_upgrade_completion_'+stamp);local.mkdir()
saved=d/('complete_joint_'+stamp);saved.mkdir()
closure=read(ws/'work/official_upgrade_current_clients_closed_20261008T111910Z.json')
assert all(x['tool_result']['status']=='fulfilled' and x['tool_result']['value']['exit_code']==0 for x in closure['current_sessions'])
qualification={}
for stage in ('B_native','B_audit','A_infer','B_score'):
    path=d/stage/'actual_D_verification.json';v=read(path)
    assert v['natural_exit']==0 and all(v[k] for k in ('source_and_argv_passed','physical_original_not_local_capture','all_original_member_SHA','ZIPCRC','unique'))
    assert sha(d/stage/'complete_actual_capture.zip')==v['archive_SHA']
    assert sha(d/stage/'actual_capture_receipt.json')==v['receipt_SHA']
    qualification[stage]=dict(path=str(path),SHA=sha(path),qualification=v)
trainpath=d/'A_train/actual_split_D_C_verification.json';trainqual=read(trainpath)
assert trainqual['natural_exit']==0 and trainqual['full_complete_archive_byte_stream_verified']
assert all(trainqual[k] for k in ('source_and_argv_passed','physical_original_not_local_capture','all_original_member_SHA','ZIPCRC','unique'))
qualification['A_train']=dict(path=str(trainpath),SHA=sha(trainpath),qualification=trainqual,large_parts_previously_byte_verified_not_rehashed_this_finalization=True)
trainlabels=d/'A_train/extracted_small/out/TRAIN_targets.npy';y=np.load(trainlabels,allow_pickle=False)
assert len(y)==1281
ratios={'TRAIN':dict(rows=len(y),weak=int((abs(y)<=1).sum()),strong=int((abs(y)>1).sum()),weak_fraction=float((abs(y)<=1).mean()),source=str(trainlabels),SHA=sha(trainlabels))}
for role,v in report['roles'].items():
    ratios[role]=dict(rows=v['actual']['rows'],weak=v['groups']['weak']['rows'],strong=v['groups']['strong']['rows'],weak_fraction=v['groups']['weak']['fraction'])
oldpath=Path('D:/CodexBackups/selective_flow_20261003_1105/paired_final_TEST_prediction_actual_20261007T043535Z/score_pair/run/out/joint_targets_and_fixed_predictions.npz')
old=np.load(oldpath,allow_pickle=False)
oldmae=float(np.abs(old['minimal_fixed_F'].astype(float)-old['labels'].astype(float)).mean())
assert abs(oldmae-.64369978747)<1e-9
diagnosis={'result':'Official aligned TEST all five point values worse than CaReFlow; this candidate did not meet the optimization goal.',
    'old_fixed_F_TEST_MAE':oldmae,'old_source':str(oldpath),'old_source_SHA':sha(oldpath),
    'new_minus_old_TEST_MAE':report['roles']['TEST']['actual']['new']['MAE']-oldmae,
    'message_functional_effect':{role:v['paired_vs_messages_off'] for role,v in report['roles'].items()},
    'not_identified':'Architecture and objective changed together; this experiment cannot causally assign the total change to Huber or donor structure alone.',
    'core_controller_pending':'Current gain is still global; full video-OOF residual estimator, calibrated utility and risk control are not yet implemented.',
    'next_independent_work':'Continue the previously specified TRAIN-only video-isolated residual/message diagnostics, verify sample-specific donor content beyond scalar calibration, and qualify the OOF controller before further full-budget experiments.',
    'no_TEST_driven_candidate_selection_or_new_training_this_finalization':True}
lines=['本次官方对齐实验已经完成，结果为负：新模型 TEST 五项点值全部低于 CaReFlow，不能宣称优化成功。',
    '新模型仅在官方 TRAIN1281 上监督训练100轮/4000次更新，以 VAL229 的既定批MSE选第30轮，固定后评价 TEST685。复用的 CaReFlow 原完整第93轮已核验；双方订单、更新数、选模和评价规则一致。损失、参数量和实测耗时不同，不称同配方同容量。',
    '', '|划分|模型|Acc7 ↑|Acc2 ↑|F1 ↑|MAE ↓|Corr ↑|','|---|---|---:|---:|---:|---:|---:|']
for role in ('VAL','TEST'):
    for label,key in [('CaReFlow','careflow'),('原方案改进','new'),('改进版同流关消息','messages_off')]:
        v=report['roles'][role]['actual'][key]
        lines.append(f'|{role}|{label}|{v["Acc7"]*100:.4f}%|{v["Acc2"]*100:.4f}%|{v["F1"]*100:.4f}%|{v["MAE"]:.6f}|{v["Corr"]:.6f}|')
lines += ['',f'此前约0.59是验证集成绩：这一版 VAL MAE={report["roles"]["VAL"]["actual"]["new"]["MAE"]:.6f}，TEST={report["roles"]["TEST"]["actual"]["new"]["MAE"]:.6f}。此前官方固定F TEST={oldmae:.6f}，新版本比它也差{diagnosis["new_minus_old_TEST_MAE"]:.6f}。',
    '消息本身的功能性贡献很小：TEST同流关消息/开消息 MAE=0.652700/0.650616，差值95%视频配对区间[-0.008129,+0.004266]跨零。弱情绪MAE从0.601387变为0.604981，强情绪从0.683129降为0.677679；不能声称解决了弱样本伤害。该开关是同一已训练模型的功能消融。',
    '改进版相对CaReFlow的 TEST MAE差=+0.031081，31视频、10000次配对重采样95%区间[-0.006043,+0.070205]。Acc7差=-4.5255个百分点，区间[-7.2785,-1.4354]个百分点；其他完整区间见原结果JSON。五项相关，不能把五项点值落后写成五个独立显著结论。',
    '', '|官方划分|弱情绪（标签绝对值≤1）|强情绪（标签绝对值>1）|弱情绪比例|','|---|---:|---:|---:|']
for role,v in ratios.items():lines.append(f'|{role}|{v["weak"]}|{v["strong"]}|{100*v["weak_fraction"]:.4f}%|')
lines += ['', '本次继续原AnchoredFlow的100维、两步Euler、reader与共同读出；六方向供体消息改为低秩 donor-minus-zero，并用Huberδ1和固定context惩罚。完整编码器与任务组件从公共初始化联合训练。本次同时修改消息和损失，不能区分两者各自的因果贡献。全局gain仍存在，完整的折外残差、校准与风险效用控制尚未实现。',
    '后续沿原定方向在TRAIN内部按视频隔离核验消息的样本特异内容、折外残差的泛化与校准，再决定是否投入下一完整训练。现有负结果不足以支持扩大这版训练矩阵。完整五折尚未完成。',
    '最终/最佳状态、Adam/scheduler/RNG、100轮验证预测、源码和失败原件均已完整保存D，两份大分片整流SHA/CRC/唯一成员核验通过；B原状态CPU完整流重放通过，未做CPU编码器前向。固定VAL/TEST预测全参数与RNG不变、dummy标签与重放通过。五项独立算术误差≤1.12e-16。四个当前SSH/SFTP均明确exit/bye自然0。',
    '历史TEST已经访问，不能称新盲测；此前合并模型VAL0.35449/TEST0.36633包含训练重叠，不纳入官方比较。CSV舍入导致本地报告算术断言失败的原件保留，报告已改用原始SHA合格NPZ，无训练、模型前向或原评分重跑。',
    f'完整统计原报告：{seal["report"]}；逐项结果：{seal["JSON"]}。本次实际封存时钟：{a.actualclock}。']
markdown=local/'官方对齐改进完成总结.md';markdown.write_text('\n\n'.join(lines[:2])+'\n\n'+'\n'.join(lines[2:]),encoding='utf8')
value=dict(actualclock_UTC=a.actualclock,phase='OFFICIAL_ALIGNED_UPGRADE_TRAIN_VAL_TEST_FIVE_AND_PRESERVATION_COMPLETE_NEGATIVE_RESULT',
    current_upgrade_official_VAL_TEST_five_complete=True,full_state_D_other_node_CPU_complete=True,
    training=dict(epochs=100,updates=4000,best_epoch=30,checkpoint_SHA=report['training']['checkpoint_SHA'],plan_SHA=pointer['active_training_plan_SHA']),
    roles={role:v['actual'] for role,v in report['roles'].items()},ratios=ratios,diagnosis=diagnosis,
    report_seal=seal,completion_report=str(markdown),completion_report_SHA=sha(markdown),original_stage_qualification=qualification,
    current_client_closure=closure,active_current_connections=[],previous_clients_unavailable_not_claimed_exit0=pointer['previous_client_sessions_unavailable_not_claimed_exit0'],
    whole_goal_complete=False,full_fivefold_complete=False,all_five_strict_superiority_achieved=False,
    lease=dict(conservative_end_UTC='2026-10-08 14:00 UTC',platform_confirmed=False,renewed=False),
    source_pointer=str(ws/'work/official_upgrade_pointer.json'),source_pointer_SHA=sha(ws/'work/official_upgrade_pointer.json'),
    capacity=dict(D_free=shutil.disk_usage(d).free,C_free=shutil.disk_usage(ws).free),no_additional_deletion=True,
    clock_is_local_seal_not_remote_capture=True)
control=local/'官方对齐完成实际接续.json';write(control,value)
for src in (markdown,control,Path(__file__),ws/'work/official_upgrade_current_clients_closed_20261008T111910Z.json',d/'final_report_seal.json'):
    shutil.copy2(src,saved/src.name)
members={f.name:sha(f) for f in saved.iterdir() if f.is_file()}
zpath=saved/'complete_joint_evidence.zip'
with zipfile.ZipFile(zpath,'x',zipfile.ZIP_DEFLATED) as z:
    for name in members:z.write(saved/name,name)
with zipfile.ZipFile(zpath) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,h in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
joint=dict(actualclock_UTC=a.actualclock,members=members,archive=str(zpath),archive_SHA=sha(zpath),all_small_SHA_CRC_unique=True,
    complete_original_large_parts_referenced_not_redownloaded=True,original_qualifications=qualification,
    original_report_seal=seal,whole_goal_complete=False)
jointbytes=json.dumps(joint,ensure_ascii=False,indent=2).encode('utf8')
jointsha=hashlib.sha256(jointbytes).hexdigest();(saved/'actual_complete_joint.json').write_bytes(jointbytes)
controlpointer=dict(value,complete_joint=str(saved/'actual_complete_joint.json'),complete_joint_SHA=jointsha)
(ws/'outputs/官方划分原方案消息改进与CaReFlow对齐实际接续.json').write_text(json.dumps(controlpointer,ensure_ascii=False,indent=2),encoding='utf8')
state=ws/'outputs/研究接续状态.md'
short=f'最新实际 {a.actualclock}：官方TRAIN-only原方案改进100轮4000更新/VAL选30已自然0；VAL五项50.21834%/87.5%/87.50848%/.597256537/.865830422，TEST46.13139%/87.17557%/87.14876%/.650616015/.825181614。TEST全部点值差于CaReFlow50.65693%/87.63359%/87.61119%/.619535294/.851188468，也差于旧F MAE.643699787；没有优化成功。开消息相对同流关消息TEST MAE仅-.00208360且配对区间跨0，弱样本仍伤害；不据TEST挑结构或扩矩阵。完整2.96GB最终/最佳model/Adam4000/scheduler/RNG/100预测原ZIP1.82GB两D分片+144成员全SHA/CRC/唯一/源argv联结通过，B原CPU341Adam/流重放误差5.4e-6通过，非CPU编码器。914条固定完整前向/零标签dummy重放state RNG不变、B一次五项与独立算术通过；CSV报告断言失败已保留并改用原NPZ，无重跑。先读官方对齐接续JSON及报告 {markdown}；joint {jointsha}。A89189/B39816/SFTP9027/9666全部明确exit/bye自然0，旧四不可用不冒0。第三租期保守Oct8UTC14:00平台未核，无续租/删原件/C实时claim。完整OOF效用控制/五折/五项超过仍未完成，原10min不改。以下历史。\n\n'
state.write_text(short+state.read_text(encoding='utf-8-sig'),encoding='utf8')
print(json.dumps(dict(report=str(markdown),control=str(control),joint=str(saved/'actual_complete_joint.json'),joint_SHA=jointsha,small_ZIP_SHA=joint['archive_SHA'],whole_goal_complete=False),ensure_ascii=False))
