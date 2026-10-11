"""Deliver measured pilot results and immutable original references."""
import argparse,csv,datetime,hashlib,json,shutil,sys,zipfile
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('work/innovation_v1').resolve()))
from contract import read,write,sha
p=argparse.ArgumentParser();p.add_argument('--clock',required=True);a=p.parse_args();stamp=a.clock[:19].replace('-','').replace(':','').replace(' ','T')+'Z'
c=read('work/innovation_v1_current.json');D=Path(c['D_actual']);sp=read('work/innovation_v1_score_current.json');score=read(Path(sp['D'])/'out/actual_scores.json');train=read(D/'A_train/extracted/out/actual_train_receipt.json');audit=read(D/'B_train_audit/extracted/out/actual_cpu_audit.json');diag=read(D/'B_TRAIN_CAL_diagnostic/extracted/out/actual_saved_train_diagnostic.json')
for name in ('A_native','B_native','A_cache','A_train','B_train_audit','B_TRAIN_CAL_diagnostic'):assert read(D/name/'actual_D_verification.json')['natural_exit']==0
labels={'baseline_calibrated':'校准后消息关闭基线','full_message':'完整供体消息','flow_ordinary_gate':'新息流＋普通任务门','flow_residual_gate':'新息流＋折外残差门','regression_residual_gate':'条件回归新息＋残差门'}
rows=['| 方法 | Acc7 % | Acc2 % | F1 % | MAE | Corr | 弱 MAE | 强 MAE |','|---|---:|---:|---:|---:|---:|---:|---:|']
for name in score['all_methods']:
    s=score['all_methods'][name];m=s['all'];rows.append(f"| {labels[name]} | {m['Acc7']*100:.4f} | {m['Acc2']*100:.4f} | {m['F1']*100:.4f} | {m['MAE']:.9f} | {m['Corr']:.9f} | {s['weak']['MAE']:.9f} | {s['strong']['MAE']:.9f} |")
fdiag=score['mechanism_diagnostics']['flow'];fd=diag['calibration_head_diagnostic']['flow'];interval=score['paired_video_bootstrap_delta_vs_calibrated_baseline']['flow_residual_gate']['MAE'];fullci=score['paired_video_bootstrap_delta_vs_calibrated_baseline']['full_message']['MAE']
md=f'''# 新息流与折外残差门：第0折机制实验实际报告

实际交付时间 {a.clock}。本轮真实训练与评价完成，结果为负：没有方法同时改善五项，新息流残差门没有改善校准基线，当前版本不扩为完整五折。新冻结特征基线 INNER MAE 0.865328，也明显没有达到旧 best36 的 0.640391；不能将原型描述为已有性能升级。

{chr(10).join(rows)}

评价为合并角色第0折 INNER264条/11视频，其中弱情绪 {score['weak_count']} 条，强情绪 {score['strong_count']} 条（弱定义为 |y|≤1）。全部五个方法同时一次评分，没有从中选择部署方法。Acc2/F1排除真实0标签，F1按类别支持量加权；MAE/Corr使用未截断预测。

## 已实现的机制

公共 DeBERTa 编码器完全冻结，使用公共预训练权重，未借用旧任务微调检查点。仅训练57,025参数基线、六方向条件流/条件回归、小型消息修正头和门。条件流以CFM学习噪声到供体状态的映射；8个固定反对称噪声样本、8步Euler端点平均估计条件可预测部分，新息为供体减去该估计。它是有限样本近似，不是精确条件期望、PID或语义真值。

每个消息贡献为 h(接收状态,消息)−h(接收状态,0)，六方向贡献平均后形成总Δ。关掉全部消息时修正严格为0，与打开消息共用固定基线，不能新增一条无需消息的解码捷径。五个对照分别为消息关闭、完整消息、流新息普通任务门、流新息残差门、条件回归新息残差门。当前是总Δ的逐样本标量门；逐方向门、流时间门、接收状态轨迹注入尚未实现。

原FIT1494拆为任务TRAIN1201与独立CAL293；CAL整视频排除在所有训练、统计和OOF前驱之外，仅拟合最后的非负斜率。任务TRAIN做5组整流水线视频OOF，内部基线再做5组视频OOF，残差监督来自排除对应视频的模型。训练固定最终步数（基线/分解/修正头160、门200），无INNER选epoch。最后一维校准后门限制在[0,1]，不构成有限样本无伤害保证。

## 结果和退化定位

新修正不再主要是旧的标量缩放：流新息总Δ被基线线性函数解释的方差比例约 {fdiag['delta_explained_by_baseline_linear_R2']*100:.3f}%，但扣除该部分后的任务残差偏相关为 {fdiag['partial_corr_after_baseline_linear']:.6f}。这说明摆脱缩放捷径并不等于产生有用修正。

新息流残差门相对校准基线的MAE差为 +0.000462822，5,000次按视频配对重采样95%区间 [{interval[0]:+.9f}, {interval[1]:+.9f}]，跨0；其实际MSE收益为 {fdiag['actual_MSE_gain']:+.9f}。普通任务门也没有改善，条件回归残差门未显示流的优势。完整消息的MAE伤害 +0.029615900，区间 [{fullci[0]:+.9f}, {fullci[1]:+.9f}]。

保存记录显示基线后40步训练MSE均值 {diag['training_histories']['baseline']['last_40_mean']:.6f}，完整最终栈的原始基线视频OOF MSE {diag['raw_baseline_TRAIN_OOF_MSE']:.6f}/MAE {diag['raw_baseline_TRAIN_OOF_MAE']:.6f}，CAL MAE {diag['CAL_baseline_MAE']:.6f}。这支持明显泛化缺口，不能仅归因为训练步数不够。各训练历史是随机训练批损失均值，和OOF指标不是同一批样本。

残差估计器在自身元训练数据上的ρ相关性为 {fd['meta_training_rhat_corr']:.6f}，CAL上只有 {fd['CAL_rhat_corr']:.6f}；元训练行虽然使用前驱模型的OOF预测，但不是控制器自身的OOF评价。CAL斜率 {fd['CAL_slope']:.9f} 将平均门收缩至 {fdiag['gate_mean']:.6f}；这是近回退，不是性能胜利。最高预测效用分箱的实际平均效用反而为负，效用排序没有通过本次机制检查。

## 实际验证、成本和保存

A真实训练child1583自然exit0，头部训练与重放 {train['elapsed_seconds']:.3f} 秒，累计GPU峰值allocated/reserved {train['max_memory_allocated_bytes']}/{train['max_memory_reserved_bytes']}字节，无peakreset。整个新完整训练态 {train['checkpoint_bytes']} 字节，包含69个训练记录、各模型/Adam/随机状态/训练历史/视频来源、最终分解与门。GPU存盘重放误差0。B原Torch CPU实际小头重放child49545自然exit0，五预测最大误差 {max(audit['CPU_prediction_error'].values()):.3g}，低于事先冻结的1e−4，所有零消息修正严格0；没有CPU编码器前向。

A/B原生合成契约、真实公共编码器缓存、真实训练、B整状态审核及TRAIN/CAL诊断全部完整原capture已D核SHA、ZIP CRC、成员唯一、完整成员SHA、源码/fullargv/PID/自然退出及实际UUID。大公共权重以原SHA联结保存，没有重复下载旧权重。所有新SSH/SFTP明确exit/bye自然0关闭（68578/77889/77440/91056），这些ID禁止复用。本轮未删除旧冻结资料；C没有本轮fresh采集，不冒三机实时。

## 后续选择与比较边界

停止本版本门/新息的大矩阵扩容，先修消息关闭基线的泛化：优先更强正则的低容量读出、保留更有用的冻结特征信息，并只用任务TRAIN的按视频留出验证；不要以继续训练步数解决训练误差已经很低的问题。基线可信后，再做低容量残差估计的TRAIN内视频验证，避免当前57k读出和7k门在少量视频上记忆。具体新的源码/固定预算与选择规则尚未冻结或运行，不能声称下一版本已改善。

这是已探索合并角色的机制原型，公共冻结编码器/1201任务TRAIN/固定160步与旧完整微调/40轮并非共同预算，不能用0.865对0.640做模型公平优劣结论，也不能用它否定所有条件流/残差机制。原官方TEST已有训练数据进入合并角色，原TEST不能继续叫保留测试集。旧正式TEST F MAE0.643699787、CaReFlow0.619535294，旧约0.59仍是DEV表现。本轮没有新官方TEST或CaReFlow训练/评分，没有完整五折，没有五项全面超过。

完整原件根：`{D}`。全部未舍入结果与诊断附JSON，逐样本CSV可供外部独立复核。
'''
out=Path('outputs')/('innovation_pilot_review_'+stamp);out.mkdir();(out/'新息流与折外残差门实际报告.md').write_text(md,encoding='utf8')
combined=dict(score=score,training=train,CPU_audit=audit,TRAIN_CAL_diagnostic=diag,original_D=str(D),plan_SHA=c['execution_plan_SHA'])
write(out/'全部实际结果.json',combined)
q=np.load(D/'A_train/extracted/out/frozen_INNER_predictions.npz',allow_pickle=False);g=np.load(D/'A_train/extracted/out/frozen_INNER_gate_details.npz',allow_pickle=False)
old=list(csv.DictReader(Path('outputs/best20_best36_FIT_INNER_review_20261008T012035Z/best36_inner_y_b_p_video.csv').open(encoding='utf8')));assert [r['row_id'] for r in old]==q['row_ids'].astype(str).tolist()
with (out/'INNER五方法逐样本.csv').open('w',encoding='utf8',newline='') as file:
    fields=['row_id','video','y']+list(labels)+['flow_delta','flow_rhat','flow_gate','regression_delta','regression_rhat','regression_gate'];w=csv.DictWriter(file,fieldnames=fields);w.writeheader()
    for i,r in enumerate(old):w.writerow(dict(row_id=r['row_id'],video=r['video'],y=r['y'],**{k:float(q[k][i]) for k in labels},**{k:float(g[k][i]) for k in fields if k in g}))
shutil.copy2(Path(c['root'])/'frozen_source_bundle.zip',out/'冻结原型源码.zip');shutil.copy2(Path(c['execution_plan']),out/'固定训练协议.json')
manifest={f.name:sha(f) for f in out.iterdir()};write(out/'成员SHA.json',manifest);zp=Path(str(out)+'.zip')
with zipfile.ZipFile(zp,'x',zipfile.ZIP_DEFLATED,6) as z:
    for f in out.iterdir():z.write(f,f.name)
with zipfile.ZipFile(zp) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,h in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==h
delivery=D/('delivery_'+stamp);delivery.mkdir();shutil.copy2(zp,delivery/zp.name);shutil.copy2(Path('work/deliver_innovation_pilot.py'),delivery/'deliver_source.py');assert sha(zp)==sha(delivery/zp.name)
status=dict(status='ACTUAL_SINGLE_FROZEN_FEATURE_MECHANISM_PILOT_COMPLETE_NEGATIVE_NOT_FULL_FIVEFOLD',actualclock_delivery_UTC=a.clock,actual_delivery_UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),report=str((out/'新息流与折外残差门实际报告.md').resolve()),share_ZIP=str(zp.resolve()),share_ZIP_SHA=sha(zp),D=str(D),source_plan_SHA=c['execution_plan_SHA'],joint_original_archives={n:read(D/n/'actual_D_verification.json')['archive_SHA'] for n in ('A_native','B_native','A_cache','A_train','B_train_audit','B_TRAIN_CAL_diagnostic')},score_SHA=sha(Path(sp['D'])/'out/actual_scores.json'),delivery_CRC_SHA_unique_passed=True,all_four_connections_closed_natural_0=[68578,77889,77440,91056],C_not_fresh_this_batch=True,D_remaining_bytes=shutil.disk_usage('D:/').free,no_additional_deletion=True,next='Repair baseline generalization on TRAIN-only video validation and validate smaller residual estimator; no current matrix expansion or saved-model rerun',whole_goal_complete=False)
write('outputs/新息流与折外残差门单折机制实验实际接续.json',status);write(D/'actual_delivery_receipt.json',status);c.update(status=status['status'],latest_report=status['report'],latest_share_ZIP=status['share_ZIP'],overall_goal_complete=False);write('work/innovation_v1_current.json',c)
head=f"最新实际 {a.clock}：先读《新息流与折外残差门单折机制实验实际接续.json》及其report。冻结公共特征单fold0原型真实A训练1583自然0/72秒/完整53MB，69头model Adam RNG历史与五INNER264预测真正D+B原TorchCPU49545自然0/误差4.78e-7/零消息严格0，六完整原capture SHA CRC成员唯一 source argv PID全D通过。全部五项一次评分本地自然0：基线MAE.865328/完整消息.894944/流普通门.868055/流残差门.865791/回归残差门.865576，无改善/无五项全胜；新Δ基线解释仅3.32%但残差偏相关−.12191。TRAIN/CAL原记录B50092自然0：基线后40训练MSE.111 vs视频OOF1.388；残差元训练corr.495→CAL.0043，斜率.018近回退。停止当前矩阵扩容，下一先TRAIN-only修低容量正则基线和残差泛化；尚未冻结运行下一版。公共冻结/160步不能与旧全微调40轮公平横比；INNER已探索/合并TEST非保留/无新官方TEST或CaReFlow/非完整五折/全目标未完成。原件与报告全D保存，未删旧件，C未fresh，4新SSH/SFTP68578/77889/77440/91056均exit/bye0禁复用。以下历史。\n\n"
state=Path('outputs/研究接续状态.md');state.write_text(head+state.read_text(encoding='utf8'),encoding='utf8');print(status)
