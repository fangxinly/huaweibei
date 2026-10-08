import json,hashlib,zipfile,shutil,sys
from pathlib import Path
clock=sys.argv[1];read=lambda p:json.loads(p.read_text(encoding='utf8'));sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();o=Path('outputs')
d=Path(read(Path('work/weak_gate_current.json'))['D']);g=read(d/'original/actual_stage_receipt.json');a=read(d/'audit_original/actual_stage_receipt.json')
for execution,receipt in [('execution',g),('audit_execution',a)]:
 n=read(d/execution/'natural_exit.json');c=read(d/execution/'actual_child.json');assert n['natural_exit']==0 and n['pid']==c['pid']==receipt['pid'] and n['fullargv']==c['fullargv']==receipt['fullargv']
assert a['original_receipt_sha256']==sha(d/'original/actual_stage_receipt.json')
old=read(o/'所有固定模型VAL_TEST五项实际对齐结果.json')['selected20_same_checkpoint_diagnosis'];cand=read(Path('work/weak_retention_current.json'))
result=dict(status='FIXED_SMOOTH_PREDICTION_GATE_NEGATIVE_DEVELOPMENT_RESULT_LOCAL_CPU_COMPLETE',actualclock_UTC=clock,formula='q=b+(b*b/(1+b*b))*(p-b)',scores=g['scores'],regions=g['regions'],existing_same_checkpoint_b_p={role:{k:old[role][k] for k in ['b','p']} for role in ['fit','inner']},audit=a,primary_pid=g['pid'],D=str(d),new_GPU_training=False,official_VAL_TEST_not_rescored=True,decision='Reject this fixed posthoc gate: INNER weak improves, but strong worsens and 4/5 main metrics worsen. Do not sweep thresholds/scales or deploy q. Does not reject all learned gating.',training_candidate=cand,training_candidate_execution_enabled=False,training_direction='Only add FIT weak-harm excess penalty; original inference unchanged. Native Torch gradient/contract verification pending; full retraining has not begun.',q_affine_OLS_not_executed=True)
(o/'弱情绪直接修正对照与新训练方案实际结果.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
md=['弱情绪直接修正：实际开发对照与下一训练方案','',f'记录 {clock}。仅已保存FIT1494/INNER264数组，新增固定候选一次评价，本地CPU主child11996及独立不同算式child1284自然0。不是新GPU训练/异节点模型前向/正式TEST/五折。','', '| 固定INNER264 | 原修正p | 预测门控q |','|---|---:|---:|']
for k in ['Acc7','Acc2','F1','MAE','Corr','MSE']:md.append(f"| {k} | {old['inner']['p'][k]:.9f} | {g['scores']['inner'][k]:.9f} |")
md+=['','门控公式 q=b+[b²/(1+b²)](p−b)，仅模型流前预测b决定门控，没有真实标签推理门控；固定尺度1，不扫参数。源protocol/predictionSHA在本次目标数组加载前冻结。FIT/INNER此前已经用于训练/选模/诊断，不能称新独立确认。','', 'INNER弱111条 MAE .638970898→.557799316，强153条 .708821345→.815039648，整体 .679452407→.706881781。弱改善对整体贡献约−.03413，强损害约+.06156，净+.02743。Acc7/Acc2/F1/MAE四项均退，Corr略升；拒绝此固定门控，不继续扫门控幅度/阈值救分。','', '机制线索：INNER强情绪51/153条流前预测绝对值≤1，强情绪平均门值只有.4745；预测强弱不是标签强弱。简单门控也压掉对强情绪有用的修正。不能把这个观察当训练因果证明或否定全部门控。','', '下一单候选改训练目标，保留原 p=b+tanh(gain)(flow−b) 推理。仅FIT中|y|≤1的样本加入 mean ReLU((p−y)²−(stop_gradient(b)−y)²)，固定系数1。有害修正才受罚，改善该样本的修正不受此项惩罚；直接梯度不能通过把b变差来降低此惩罚。原四损失、优化器、源架构和推理标签隔离保留。','', '有限惩罚不能保证泛化或弱样本绝不退化，共享表示仍会影响强样本；这是待训练验证假设。新候选源码完整D保存，执行disabled。当前剩余租期不足训练加至少2h保存，D约2.46GB不足新完整训练状态与余量，未启动新训练。原服务器Torch合成梯度验证等待连接；B新SSH一次banner超时不是权限拒绝，原已完成评分/保存不受影响。13:00真实动态保存仍需执行，不能冒已完成。','', f"新训练候选：{cand['D']}。固定门控原件：{d}。OLS对照源未冻结或拟合，不冒执行。"]
(o/'弱情绪直接修正对照与新训练方案实际结果.md').write_text('\n'.join(md),encoding='utf8')
for p in [o/'弱情绪直接修正对照与新训练方案实际结果.json',o/'弱情绪直接修正对照与新训练方案实际结果.md',Path(__file__)]:shutil.copy2(p,d/p.name)
files=[p for p in d.rglob('*') if p.is_file() and p.name!='complete_local_result.zip'];(d/'complete_manifest.json').write_text(json.dumps(dict(actualclock_UTC=clock,member_SHA={p.relative_to(d).as_posix():sha(p) for p in files}),indent=2),encoding='utf8')
with zipfile.ZipFile(d/'complete_local_result.zip','x',zipfile.ZIP_DEFLATED) as z:
 for p in files+[d/'complete_manifest.json']:z.write(p,p.relative_to(d).as_posix())
with zipfile.ZipFile(d/'complete_local_result.zip') as z:assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
print(json.dumps(dict(status=result['status'],seal_SHA=sha(d/'complete_local_result.zip'))))
