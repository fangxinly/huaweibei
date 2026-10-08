"""Read original arrays/audits, verify independent CPU receipt, write findings."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
work=Path(__file__).resolve().parent; outputs=work.parent/'outputs'
done=Path('D:/CodexBackups/selective_flow_20261003_1105/train_oracle_completed_20261006T0158Z/c')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
local=json.loads((outputs/'TRAIN_Oracle原始数组独立核验.json').read_text(encoding='utf-8'))
cpu=json.loads((done/'cpu_preservation_receipt.json').read_text(encoding='utf-8'))
manifest=json.loads((done/'preservation_manifest.json').read_text(encoding='utf-8'))
assert cpu['status']=='ORIGINAL_TRAIN_ORACLE_CPU_FILES_AND_ARRAYS_VERIFIED'
assert cpu['diagnostic_node']=='c' and cpu['target_cpu_node']=='b' and not cpu['torch_imported'] and not cpu['cuda_training_executed']
assert cpu['target_uuid']=='GPU-a259ab5e-884a-ec9f-4208-1bc34a6e5067'
assert cpu['files']==manifest['files'] and cpu['manifest_sha256']==sha(done/'preservation_manifest.json')
assert cpu['cpu_source_sha256']==sha(work/'verify_train_oracle_cpu_v1.py')
for name,m in cpu['files'].items():assert sha(done/name)==m['sha256'] and (done/name).stat().st_size==m['bytes']
assert cpu['array_audit_sha256']==sha(done/'independent_cpu_array_audit.json')
independent=json.loads((done/'independent_cpu_array_audit.json').read_text(encoding='utf-8'))
assert independent['receipt']==local['receipt']
for condition,values in local['results'].items():
 for key,value in values.items():
  other=independent['results'][condition][key]
  if value is None:assert other is None
  else:assert abs(value-other)<1e-12
utc=datetime.datetime.now(datetime.timezone.utc)
proof=dict(utc=utc.isoformat(),status='C_ORIGINAL_ORACLE_CPU_B_RECEIPT_DOWNLOADED_AND_INDEPENDENTLY_AUDITED',
           cpu_receipt=cpu,local_and_cpu_metrics_max_tolerance=1e-12,new_fullweight_created=False)
(outputs/'TRAIN_Oracle异节点CPU原回执核验.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
rows=local['results'];raw=rows['raw_fixed']['mse'];learn=rows['learned_last']['mse'];oracle=rows['oracle_last']['mse']
gain_fraction=(raw-learn)/(raw-oracle)
r=local['receipt']
table=[]
for key,label in [('raw_fixed','原固定消息'),('learned_last','学习残差三步'),('oracle_last','真实标签残差Oracle三步'),('oracle_best_including_F','Oracle含原F最佳候选'),('zero_last','零残差三步'),('mismatched_last','循环错配标签三步')]:
 x=rows[key];table.append(f"|{label}|{x['mae']:.8f}|{x['mse']:.8f}|{x['video_equal_mse']:.8f}|{x['improvement_fraction']*100:.2f}%|")
report=f'''# 固定C2的TRAIN Oracle可达性诊断

真实GPU执行UTC {r['started_utc']}至{r['completed_utc']}，共{r['elapsed_seconds']:.2f}秒，C节点PID{r['pid']}正常exit0并退休。官方TRAIN1281条/52视频、41个32行批次含末批1条，六条件均完成。原始数组本地独立审核通过，已永久D保存；原数组/源/计划/日志/exit0九文件在B作NumPy CPU全SHA及数组审核，原回执已下载并独立核验，不是重算C模型或新训练。

|条件|TRAIN MAE|TRAIN样本加权MSE|视频等权MSE|相对原F观测损失改善比例|
|---|---:|---:|---:|---:|
{chr(10).join(table)}

在同一固定C2权重、相同三步预算及信赖域下，学习残差相对原F的TRAIN MSE下降{(raw-learn)/raw*100:.2f}%，真实标签Oracle下降{(raw-oracle)/raw*100:.2f}%。学习残差的净改善约为这次有限步Oracle所观察净改善的{gain_fraction*100:.2f}%；这一比值不把Oracle称严格上界、也不代表总体可达改善比例。

当前可行消息空间确实找到了标签已知的逐例改善，因而不能把此前DEV无提升简单归因于消息完全无法改变有效任务输出。Oracle末步1281例全部改善；加入原F的最佳候选在审核容差内不劣，最优步与末步重算预测有float32微小差异。这仍不是求得全局最优，不代表未知条件均值、无标签部署或DEV/TEST收益。

负对照给出方向性证据：零残差令风险目标偏向p0，不等于无控制，TRAIN MSE更差；循环错配标签也更差。残差内容对当前求解器结果有实际影响。不能由这两个对照唯一诊断所有误差源，不能把观测标签残差与真实条件残差混称。学习残差代理/观测风险Pearson {rows['learned_last']['proxy_observed_pearson']:.4f}、Spearman {rows['learned_last']['proxy_observed_spearman']:.4f}，只对应当前已拟合TRAIN诊断。

原控制重放最大误差{r['cached_learned_replay_error']}、替换学习分支输入标签误差{r['label_replacement_error']}；参数SHA前后均{r['parameter_sha_before']}且所有参数梯度None。实际p0重放误差{r['p0_replay_error']:.3g}。同预测double直接/展开风险目标相对误差{r['equivalence']['float64_objective_relative_error']:.3g}、输入梯度和HVP相对误差均0。两种Oracle目标数学等价，仅用于核验，未当两个不同能力层级。峰值已分配显存{r['peak_allocated_bytes']}字节，没有优化器或更新模型参数。

保持原β65294.20088076077、TRAIN尺度、步长.25、三步及相对信赖域.25，未扫参。Oracle末步边界命中{rows['oracle_last']['boundary_fraction']*100:.2f}%，学习残差0%，错配标签{rows['mismatched_last']['boundary_fraction']*100:.2f}%；只能说明本次路径的边界状况。原参考和C2已拟合整个TRAIN、并在DEV选模，TRAIN低误差和Oracle收益均有拟合/标签已知边界，不能说方法已经获得泛化优势。这里没有读取DEV/TEST缓存，原九个100轮及诊断不重跑。

下一有依据的工作是三折按视频隔离标量教师的真实机制预检：从公共预训练和随机任务模块初始化，fit-only统计、内层选模、outer标签隔离，完整梯度/初始化/显存/实测预算通过后才正式训练。现有A特征仍全TRAINfit/DEVselected，即使教师OOF也不称整个流程crossfit。仍需判断该教师标量能否改善可部署残差与最终DEV行为，不能由Oracle成功跳过教师误差与匹配控制。

工具情况：原SSH38957已不存在，新正常SSH会话输入与nvidia-smi实际成功，三预期UUID核验通过。旧原因仍未定位，不声称底层工具代码已修复。C最初缺少映射文件，已由永久A快照提取完全相同SHA的1281行视频映射后通过21pins并运行；原源/协议/参数未改。

当前完整模型研究和租期保存仍未整体结束；旧C最终/旧14:30捕获缺口不回填。完成后扩展capture16纳入新Oracle和CPU所有日志/exit文本，三节点原子快照及原回执另行审核后才称联合动态保存通过。
'''
(outputs/'TRAIN_Oracle可达性实验分析.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(status=proof['status'],learned_mse_reduction=(raw-learn)/raw,oracle_mse_reduction=(raw-oracle)/raw,ratio=gain_fraction)))
