from pathlib import Path
import datetime,hashlib,json,shutil,numpy as np
cwd=Path(__file__).parent.parent;out=cwd/'outputs'
d=Path('D:/CodexBackups/selective_flow_20261003_1105/matched_message_pools_actual_v2_20261006T065508Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
j=json.loads((d/'joint_preservation_audit.json').read_text());assert j['status']=='MATCHED_MESSAGE_POOLS_D_ORIGINALS_OTHER_NODE_CPU_AND_ACTUAL_B_CAPTURE_JOINTLY_PASSED'
e=json.loads((d/'evaluation/receipt.json').read_text());g=json.loads((d/'execute/receipt.json').read_text())
z=np.load(d/'execute/predictions_frozen.npz',allow_pickle=False);sel=z['role']=='evaluation';cal=z['role']=='calibration'
assert sel.sum()==863 and cal.sum()==418
shift=np.linalg.norm(z['old_shift'].astype(np.float64).reshape(1281,-1),axis=1)
ratio=shift/z['normalized_radius'].astype(np.float64)
coverage={'prediction_sha256':sha(d/'execute/predictions_frozen.npz'),'labels_read_for_coverage':False,'model_forward':False,'new_fit':False,'radius_fraction_grid':[.125,.25,.5,1.]}
for name,mask in [('EVAL',sel),('CAL',cal)]:
    pv=z['candidate_valid'][mask];delta=z['native_prediction'][mask].astype(np.float64)-z['pf'][mask].astype(np.float64)
    coverage[name]={'rows':int(mask.sum()),'old_native_displacement_to_radius_quantiles':np.quantile(ratio[mask],[0,.25,.5,.75,1]).tolist(),
        'native_output_over_cap_rows':int((np.abs(delta)>z['cap'][mask]).sum()),
        'OT_mean_valid_nominal_including_F':float(pv[:,[0,1,2,3,4,5,6,7,8]].sum(1).mean()),
        'Opm_mean_valid_nominal_including_F':float(pv[:,[0,1,2,3,4,9,10,11,12]].sum(1).mean())}
(d/'coverage_scale_finding.json').write_text(json.dumps(coverage,indent=2))
records=d/'research_records';records.mkdir(exist_ok=True);prior=records/'preceding_state';prior.mkdir(exist_ok=True)
for name in ['研究接续状态.md','研究建议交流接续.json','匹配消息候选短实验执行接续.json']:
    if not (prior/name).exists():shutil.copy2(out/name,prior/name)
report=f'''匹配消息候选对照：实际结果与优化决策

实际更新时间UTC {now}。新CAL视频等权拟合与新的真实GPU有限消息池对照均已执行，原始数组、自然退出回执、永久D、异节点B独立NumPy审核和实际capture19快照联合通过。本实验没有训练学生或更新模型参数。

1. 方法与标签角色

固定C2 best37、原beta65294.20088076077、原TRAIN尺度、3步0.25和消息信赖域0.25。使用旧控制与原OOF教师驱动的三步末端消息位移，取尺度度量中的单位方向；增加反向旧末端位移作匹配对照。原F锚跨三步固定。13个共享真实消息为F及三方向×四幅度1/8、1/4、1/2、1，半径R=0.25||F/scale||。每个候选均经冻结缓存坐标终端真实前向，输出不是直接线性插值。

四候选池为旧方向O（5个）、教师方向T（5个）、O+T（9个）、O+反向O（9个）。接受器分别采用旧目标p0-rho与CAL目标pF+lambda*(mu-pF)，输出上限过滤、float64相对风险Q、最早最小且严格负值才接受，否则精确F。lambda只改变接受器，原mu用于教师路径生成。九对九匹配候选选择机会；教师mu及三步生成增加计算成本，未测单臂全流程相同成本。

CAL固定18视频418行，按视频等权拟合lambda：0 / 0.012728087507237378 / 0.06249098467764248。输出上限由CAL旧原生输出移动95百分位确定：0.030717942863702774 / 0.03076976463198663 / 0.03591611012816428，仅经验幅度限制。第0折CAL接受器必须回退F。八臂全部预测与索引保存并SHA冻结后才读取固定EVAL 34视频863行标签；无CAL重拟合、DEV或TEST访问。

下表仅这个已探索TRAIN内部的固定EVAL角色。参考C2仍全TRAINfit/DEVselected，因此不是全新确认集或全流程crossfit。不能与老师OOF或官方DEV分数直接比较。

|方法|MSE|MAE|视频等权MSE|修改行|修改中有害比例|
|---|---:|---:|---:|---:|---:|
'''
names=['F','native','old_O','old_T','old_OT','old_Opm','cal_O','cal_T','cal_OT','cal_Opm','oracle_O','oracle_T','oracle_OT','oracle_Opm']
for name in names:
    m=e['metrics'][name];h=m['harmful_fraction_accepted'];hs='—' if h is None else f'{100*h:.2f}%'
    report+=f"|{name}|{m['mse']:.9f}|{m['mae']:.9f}|{m['video_equal_mse']:.9f}|{m['acceptance_rows']}|{hs}|\n"
report+=f'''
native是旧原生控制单列基线；oracle只在同上限有限候选池内事后读取标签选择，不是可部署策略、严格上界或用于挑新参数。

2. 结论

CAL O+T略好于CAL O+反向O，但仍差于原F和旧原生控制；修改44/863行，24/44有害。旧接受器的O+T几乎等于O单方向。有限池标签已知Oracle的O+T MSE {e['metrics']['oracle_OT']['mse']:.9f}，反向旧池 {e['metrics']['oracle_Opm']['mse']:.9f}，并未发现整体教师末端方向容量优势。保留负结果，停止扩大这版教师末端方向和接受器，不启动新100轮。

3. 无标签尺度覆盖发现

旧原生末端消息位移相对于R的EVAL中位数为{coverage['EVAL']['old_native_displacement_to_radius_quantiles'][2]:.8f}，最小候选幅度0.125R约为它的6.35倍。O+T池同输出上限下平均只剩{coverage['EVAL']['OT_mean_valid_nominal_including_F']:.6f}个有效名义候选（包含F）；反向旧池为{coverage['EVAL']['Opm_mean_valid_nominal_including_F']:.6f}。旧原生输出本身有{coverage['EVAL']['native_output_over_cap_rows']}行超过当前上限。故本结果限制在这一粗半径网格与输出上限下，不能据此证明任何教师方向普遍无用。

下一优先候选是以旧原生末端位移范数作共同半径，先仅固定CAL行、零标签真实终端检查候选覆盖和旧基线可重现性，复用已保存位移而不重新求解旧路径。它是结构性候选尺度修订，尚未实现或执行。已经读过EVAL不能马上再扫半径并宣称新确认收益。另一必要路线是新FIT-only完整流参考：现有视频教师为plain masked mean-pool fusion，不能当作隔离流参考；编码器/流/供体/读出需各折公共预训练加随机任务初始化和独立新训练协议。

4. 实际机制与保存

预检child14652自然exit0，35行含TRAIN620长度1；标签替换0、单token双精度FD-HVP相对误差4.67e-9/3.98e-8、选中真实消息终端重放0、参数SHA不变、无参数梯度或optimizer。正式wrapper14819/child14820自然exit0于UTC2026-10-06T07:01:20.572764，1281行41批，耗时{g['seconds']:.5f}s、全程峰值{g['actual_peak_allocated_bytes']}bytes。旧/教师路径、原F和旧原生输出重放0，p0误差9.54e-7。这是缓存坐标终端，不冒本轮新DeBERTa原输入整模型重放。

v1预检原始/proc argv因启动竞争为空，独立审核拒绝后保留，未正式执行v1；新wrapperv2只等待实际exec argv，科学源不改。补充覆盖审核v1的绝对1e-6半步断言遭float32原消息舍入1.43e-6拒绝，原拒绝保留；v2按逐坐标推导舍入界核验，科学预测、参数、cap和选择断言未改。全部核心候选/风险/指标由B原NumPy脚本重建，不是CPU模型前向。

GPU新原NPZ实际完整下载51,095,993bytes SHA b6c23484a9fff5d444fa33605f3d8eeee82942411e5fe93cb97f9c1bee7c0c9f。42原文件包实际52,644,533bytes SHA0feb327100782bda3fe1778dd7ba56adcd6c5ed19ca699380c2cef3f63265e8b。B CPU预检/正式/EVAL三原回执与自然exit0已下载；actual B capture UTC{j['utc']}，586成员，全SHA/CRC/唯一成员、42原文件与大NPZ实际D原包联结、524旧科学小文件和10大SHA引用未变联合过。actual C预检06:58:25与完成07:05:06快照均过；A无本轮新材料，不冒新三节点capture。旧完整权重未重传；大权重fresh SHA引用不是再次下载。

永久D：{d.as_posix()}。完整后续研究与租期保存未完成；租期仅估UTC12:08:17/BJ20:08:17，UTC08:08/10:08/11:38强化实际动态保存仍待执行。
'''
name='匹配消息候选对照实际结果与优化决策.md';(out/name).write_text(report,encoding='utf-8');shutil.copy2(out/name,records/name)
(out/'匹配消息候选短实验执行接续.json').write_text(json.dumps({'stage':'ACTUAL_GPU_AND_FIXED_EVAL_COMPLETE_D_OTHER_NODE_CPU_CAPTURE_PASSED','updated_utc':now,'permanent_root':d.as_posix(),'joint_evidence_sha256':sha(d/'joint_preservation_audit.json'),'GPU_prediction_sha256':g['prediction_sha256'],'formal_child':14820,'formal_exit_code':0,'formal_utc':'2026-10-06T07:01:20.572764+00:00','new100_started':False,'decision':'Do not expand this teacher endpoint/grid or launch100. Next only CAL-only zero-label native-radius mechanism feasibility, then separate fully isolated flow-reference protocol.','EVAL_rescue_sweep_authorized_as_confirmation':False},ensure_ascii=False,indent=2),encoding='utf-8')
ledger=json.loads((out/'研究建议交流接续.json').read_text(encoding='utf-8'));ledger['updated_at_utc']=now;ledger['scientific_state']='新匹配GPU八臂与固定EVAL已实际完成且D/B CPU/capture联合通过；教师并集不胜原F/旧原生，有限池Oracle不胜反向旧；未新100。'
for item in ledger['accepted_suggestions']:
    if '候选' in item['suggestion']:item['status']='ACTUAL_COMPLETE_NEGATIVE_WITH_D_CPU_CAPTURE';item['evidence']=(d/'joint_preservation_audit.json').as_posix()
(out/'研究建议交流接续.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'report':(out/name).as_posix(),'coverage':coverage,'joint':j['status']},ensure_ascii=False))
