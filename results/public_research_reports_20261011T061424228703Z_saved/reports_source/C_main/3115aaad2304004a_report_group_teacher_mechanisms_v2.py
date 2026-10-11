from pathlib import Path
import json,datetime,hashlib,shutil
w=Path(__file__).resolve().parent;o=w.parent/'outputs';d=Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_precheck_20261006T0210Z')
audit=json.loads((o/'视频隔离教师v2三GPU机制回执独立核验.json').read_text(encoding='utf-8'));cpu=json.loads((o/'视频隔离教师v2异节点CPU原回执核验.json').read_text());formal=json.loads((o/'视频隔离教师正式100冻结核验.json').read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert audit['status'].endswith('AUDIT_PASSED') and cpu['status'].endswith('AUDITED') and formal['status'].startswith('FORMAL_V4_SOURCE_AND_BUDGET_FROZEN')
lines=['# 按视频隔离教师的实际机制核验与正式预算','',f'报告实际UTC {datetime.datetime.now(datetime.timezone.utc).isoformat()}。本地静态设计已推进到三节点真实GPU预检。正式训练是否健康及轮数，以另附实际live快照与原协议为准，不由本文件证明100完成。','',
'第一版预检真实四次Adam更新、七模块梯度、标签替换、fit-only统计、初始化/100订单及磁盘重载均通过。但进一步逐个检查全部参数时发现两个作者遗留pooler参数没有梯度。三个原检查均自然exit1，失败日志和原回执已永久D保留。这不是原C单token数值失败，也没有重启旧实验。原七模块检查不能被表述为全部参数梯度检查。','',
'新候选只删除标量前向完全不用的pooler.dense.weight/bias，在优化器构造前移除，另存runtime_v2与plan_v3，未修改旧冻结源。三个真实GPU修订预检证明：其他全部初始张量逐一相等，原32条输入预测误差0，291个保留参数张量全部有有限梯度；四次更新成功。原先185339595个标记可训练参数减去590592个无用pooler参数，现在184749003个参数实际参与标量教师训练。','',
'三折fit/inner/outer分别695/153/433、715/142/424、714/143/424；只官方TRAIN52视频1281条。归一化只用各自fit内容token，NumPy从原witness重新计算mean/std/active通过。outer标签访问guard实际拒绝；fit/inner标签替换不改变前向预测；原输入严格初始整模型磁盘重载误差0。跨折非统计初始SHA完全相同，归一化缓冲区因fit视频不同允许不同。','',
'这次教师与此前流控制实验不同：从公共DeBERTa加随机任务初始化开始，删除流和无用pooler，全量微调保留的文本/音频/视觉编码器、fusion/predictor。不能把既有冻结骨干的流实验说成全模型端到端训练；也不能把新教师微调当成已经训练新的方向控制策略。','',
'|折/节点|四次GPU更新最慢秒|预检峰值已分配GiB|保守100轮估计分钟|','|---|---:|---:|---:|']
for fold,node in enumerate(['a','b','c']):
 r=json.loads((d/node/'precheck_v2/receipt.json').read_text());e=audit['folds'][fold]
 lines.append(f'|{fold}/{node.upper()}|{max(r["update_seconds"]):.3f}|{r["peak_allocated_bytes"]/1024**3:.3f}|{e["seconds_conservative_pilot_estimate"]/60:.2f}|')
lines+=['','预算将最慢实测更新和内评估翻倍，加上初始化/重载成本；仅估计不是保证。相对于人类24h估计Oct6UTC12:08:17，保留至少2小时保存余量，平台截止仍未核实。正式冻结时D实际约21GB，为三份新选中完整教师及资料预留8GiB；各远端容量独立fresh核实且无需删除。','',
'正式plan_v4和训练源已在实际三GPU修订预检、独立数组审核与B/C原CPU回执下载后冻结。每折完整100轮，只按INNER样本加权MSE的最早严格最小选模型；100结束后整模型严格磁盘重载，再一次零标签OUTER输入预测任务标量μ。outer真实标签不进入拟合、归一化、选模或步数选择，DEV/TEST不参与教师设计训练。','',
'B与C分别保留41份原始小文件/数组/源/失败证据并作NumPy CPU全SHA审核，两个原CPU回执已下载独立核对；A/C原证据在B有异源节点副本，B原证据在C有异源节点副本。CPU同节点部分明确记录，不能把B自检当作B的异节点保存。初始整权重未下载或在这些CPU副本整模型重载；预检/动态捕获只提供其fresh整体SHA和原GPU严格重载证据。100选中完整模型仍需完成后另取D与异节点CPU。','',
'完成以后才可计算视频隔离OOF质量及其残差目标偏差；当前没有OOF结果或泛化收益结论。跨老师只转同一任务尺度的μ标量；现有A参考仍全TRAIN拟合/DEV选模，因此不是全流程crossfit。既有C2未优于对照、Oracle仅TRAIN标签已知可达性及旧C最终/14:30缺口仍保持。']
t=o/'视频隔离教师机制核验与正式预算.md';t.write_text('\n'.join(lines)+'\n',encoding='utf-8')
for f in [t,o/'视频隔离教师v2三GPU机制回执独立核验.json',o/'视频隔离教师v2异节点CPU原回执核验.json',o/'视频隔离教师预检三节点动态保存联合核验.json',o/'视频隔离教师正式100冻结核验.json',w/'audit_group_teacher_formal_snapshot_v1.py',w/'audit_group_teacher_snapshots_v2.py',w/'audit_group_teacher_downloaded_cpu_v2.py',w/'capture_soft_vector_v19.py',w/'audit_soft_snapshot_v19.py']:
 dest=d/f.name;assert not dest.exists();shutil.copy2(f,dest);assert sha(dest)==sha(f)
print('TEACHER_MECHANISM_AND_FORMAL_BUDGET_REPORT_PERMANENT_D_SHA_PASSED')
