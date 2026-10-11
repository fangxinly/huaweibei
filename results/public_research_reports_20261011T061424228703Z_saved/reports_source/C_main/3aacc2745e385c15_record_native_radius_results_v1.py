from pathlib import Path
import datetime,hashlib,json,shutil
cwd=Path(__file__).parent.parent;out=cwd/'outputs';d=Path('D:/CodexBackups/selective_flow_20261003_1105/native_radius_cal_mechanism_actual_20261006T072031Z')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();now=datetime.datetime.now(datetime.timezone.utc).isoformat()
ca=json.loads((d/'C_capture_audit.json').read_text());ba=json.loads((d/'B_capture_audit.json').read_text());receipt=json.loads((d/'mechanism/receipt.json').read_text());audit=json.loads((d/'mechanism/independent_audit.json').read_text())
assert ca['status']=='CAL_NATIVE_RADIUS_C_D_ORIGINALS_AND_ACTUAL_CAPTURE_VERIFIED' and ba['status']=='CAL_NATIVE_RADIUS_B_D_ORIGINALS_AND_ACTUAL_CAPTURE_VERIFIED'
assert json.loads((d/'independent_cpu/cpu_array_audit.json').read_text())==audit
joint={'status':'CAL_NATIVE_RADIUS_D_ORIGINAL_GPU_OTHER_NODE_CPU_AND_ACTUAL_C_B_CAPTURE_JOINTLY_VERIFIED','updated_utc':now,'prediction_sha256':receipt['prediction_sha256'],'C_capture_audit_sha256':sha(d/'C_capture_audit.json'),'B_capture_audit_sha256':sha(d/'B_capture_audit.json'),'new_risk_metrics':False,'EVAL_forward_rows':0,'whole_research_complete':False}
(d/'joint_preservation_audit.json').write_text(json.dumps(joint,indent=2))
records=d/'research_records';records.mkdir(exist_ok=True)
report=f'''原生位移半径：CAL零标签真实消息覆盖核验

UTC {now}。这是一项实际GPU机制预检，418行18个固定CAL视频，未读任何真实标签、未重新拟合CAL、未对EVAL前向或计算新指标；没有新学生或100轮训练。

共同半径改为已保存旧原生末端消息位移在TRAIN尺度中的范数。旧方向幅度1直接采用重建原生消息，避免float32归一化来回误差；旧/教师/反向旧三方向共享1/8、1/4、1/2、1四幅度。13个真实消息终端前向，复用旧路径保存的消息和方向，不再求解旧或教师三步路径。输出上限与旧CAL目标未变。只统计候选唯一性、cap有效数和负代理值覆盖，未生成新的接受后控制预测或标签收益。

|CAL机制指标|旧信赖半径网格|原生位移半径|
|---|---:|---:|
|O+T平均cap有效唯一候选（含F）|{audit['coverage_OT_and_Opm'][0]['prior_grid_mean_valid']:.6f}|{audit['coverage_OT_and_Opm'][0]['mean_cap_valid_unique']:.6f}|
|O+反向O平均cap有效唯一候选（含F）|{audit['coverage_OT_and_Opm'][1]['prior_grid_mean_valid']:.6f}|{audit['coverage_OT_and_Opm'][1]['mean_cap_valid_unique']:.6f}|

两个主池名义和实际唯一动作均为9；旧原生幅度1在396/418行通过原输出cap，另外22行被该既定cap过滤，不能因修订半径而称包含基线就保证可选择它。两个主池各有274行存在符合lambda门控的负CAL代理候选，代理改善不等于标签改善。

本地补充只读原数组的非F覆盖核验：O+T有实际cap合法非F动作的行数从96/418增至418/418，反向旧池从100/418增至418/418，旧单方向从95/418增至418/418。以上新合法动作的输出变化均超过1e-6既有重放分辨率，因此不是仅重复F导致均值变大；1e-6只用于描述数值分辨率，不是风险margin。每折/每视频与输出移动分位数保存在nonF_CAL_coverage_analysis.json。该补充是本地已有原数组分析，未宣称另跑异节点CPU这份新补充脚本。

边界：本次418行旧位移半径均非零；没有新的零半径合成GPU边界验证，也没有接受器全拒绝/零cap拼接重放。本轮只候选前向与cap统计，不能把有限CAL通过冒作一般可部署精确零半径回退已过。未来隔离参考控制实现须独立处理精确F、零/非有限方向和真实选择后拼接重放。

原F与旧原生输出重放最大误差均2.384185791015625e-7；CAL行454/620均单有效token，毒化标签替换0，单样本/双样本终端误差1.1920928955078125e-7。原model state SHA前后均13f0d54e10ceafc9d676e03adfc2298aea8b05d11db4aec6ca6e4b405dfbed8b；没有参数梯度、optimizer对象或步骤。此次只有限消息前向，无新的梯度/HVP求解器，不把之前HVP证据称本轮重做。仍是缓存坐标终端，不是新DeBERTa原输入整模型前向。

实际child14988自然exit0，回执UTC{receipt['utc']}，耗时{receipt['seconds']:.6f}s，全程构造/加载/全部前向峰值{receipt['peak_allocated_bytes']}bytes。源SHA {receipt['source_sha256']}，plan SHA {receipt['plan_sha256']}。新完整数组实际下载{receipt['array_bytes']}bytes，SHA {receipt['prediction_sha256']}。本地准备v1的float32消去“完全相等”fixture拒绝保留；v2准备助手按推导舍入界核验，科学源始终直接使用原生消息分支且未改，v1从未部署。

13原文件永久D包16,335,909bytes SHAf4b38dffee589ba21baff62514e7ed61dee5c526d3d932ba07581888c46f875c，没有标签。B异节点原NumPy逐数组/半径/候选/原输出/cap/代理/行序核验与原自然exit0回执已下载，非CPU模型前向。C actual capture19 UTC{ca['utc']}，{ca['members']}成员，B actual capture19 UTC{ba['utc']}，{ba['members']}成员；全SHA/CRC/唯一成员、原源/过程/回执/新完整数组与large引用联结、旧科学成员/旧大SHA未变联合通过。A无本轮新材料，不冒三新快照。旧完整模型没有重传，大SHA引用不是再次下载。

决策：原先候选尺度覆盖缺陷得到机制验证；前次八臂固定EVAL负结果保留，未产生新的收益证据。限定的半径修订预检到此完成，不继续在已读EVAL上扫半径/惩罚或启动新100。下一研究优先建立真正FIT-only流参考的独立可行性协议，准确处理公共预训练随机任务初始化、归一化、参考流/供体/decoder参数与视频隔离角色。现有plain masked fusion教师不能替代流参考，旧fullTRAINfit A/C2不能作隔离初始化。

永久D {d.as_posix()}。整体研究和未来租期保存未完成；仅估UTC12:08:17/BJ20:08:17期限，强化实际动态保存UTC08:08/10:08/11:38仍待执行。
'''
name='原生位移半径CAL零标签机制实际结果.md';(out/name).write_text(report,encoding='utf-8');shutil.copy2(out/name,records/name)
(out/'原生位移半径机制执行接续.json').write_text(json.dumps({'stage':joint['status'],'updated_utc':now,'D':d.as_posix(),'remote':'/data/coding/group_teacher_v1_20261005T1650Z/native_radius_cal_mechanism_v1_20261006T072031Z','joint_sha256':sha(d/'joint_preservation_audit.json'),'selected_predictions_generated':False,'EVAL_scores_generated':False,'new100':False,'decision':'One allowed geometry check done. No adapted EVAL sweep. Next isolated full-flow reference feasibility.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(joint))
