"""Audit local plan/source consistency and preserve preparation, never GPU claims."""
from pathlib import Path
import ast, datetime, hashlib, json, shutil
work=Path(__file__).resolve().parent; outputs=work.parent/'outputs'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
plan=json.loads((work/'train_oracle_plan_v2.json').read_text(encoding='utf-8'))
assert plan['diagnostic_source_sha256']==sha(work/'diagnose_train_oracle_v2.py')
source=(work/'diagnose_train_oracle_v2.py').read_text(encoding='utf-8')
tree=ast.parse(source)
assert 'dev_cache.npz' not in source and 'test_cache.npz' not in source
assert 'torch.optim' not in source and '.backward(' not in source
assert "y if name == 'oracle' else None" in source
assert plan['rows']==1281 and plan['batch_size']==32 and plan['optimizer_steps']==0
assert plan['controller']['inner_steps']==3 and plan['controller']['step_size']==.25 and plan['controller']['trust_fraction']==.25
assert plan['maximum_seconds']==2700 and plan['preservation_reserve_seconds']==7200
for filename in ['audit_train_oracle_v1.py','freeze_train_oracle_plan_v2.py']:
    ast.parse((work/filename).read_text(encoding='utf-8'))
utc=datetime.datetime.now(datetime.timezone.utc); stamp=utc.strftime('%Y%m%dT%H%M%SZ')
failure=dict(utc=utc.isoformat(),action='Same original SSH38957 read-only GPU and disk query',
    error='write_stdin rejected: approval required by policy, but AskForApproval::Granular.sandbox_approval is false',
    remote_command_executed=False,remote_status_unknown=True,alternative_connection_attempted=False,
    conclusion='Current tool approval denial; not evidence of server or credential failure.')
(outputs/('远端实验启动工具拒绝_'+stamp+'.json')).write_text(json.dumps(failure,ensure_ascii=False,indent=2),encoding='utf-8')
report=f'''# TRAIN Oracle 短实验准备与启动状态

真实本地UTC {utc.isoformat()}。用户已授权开展实验。本轮同一原SSH38957只读查询被工具以Granular.sandbox_approval=false拒绝，命令没有执行；没有换连接、命令或客户端绕过。服务器实时状态未知，GPU实验尚未启动。

已准备新诊断源work/diagnose_train_oracle_v2.py、固定计划work/train_oracle_plan_v2.json及独立NumPy审核器work/audit_train_oracle_v1.py。v1本地准备保留；v2在任何GPU执行前加强标签隔离，仅oracle候选选择获得真实y，学习残差和零残差控制不接收y。语法/AST和计划一致性已检查，不能冒称真实GPU标签/梯度/HVP通过。

估计对象：固定已完成C2第37轮供体/任务头及既有全TRAINfit、DEVselected参考，以官方TRAIN1281/52视频诊断当前消息空间及三步求解器。21项原源码/权重/缓存/映射/选模文件SHA已从永久材料锁定。不开DEV/TEST缓存；没有优化器或模型参数更新，不重训或重传九个完成模型。

条件：原固定F；已学习残差三步最终输出；真实标签残差三步最终输出与包括原F的最佳可行候选；零残差；官方行序循环错配标签负控制。保留原beta、TRAIN尺度、步长.25、三步和信赖域.25，不扫参。直接风险/展开式仅核验恒等式及输入梯度/HVP，不能命名为更强oracle。

预算：单C节点，32行诊断批次且保留末批1行，41批；最多2700秒/峰值已分配1GiB，启动需空compute、实际UUID、1GiB远端空闲和距人类估计租期结束至少实验预算加7200秒。过预算在本诊断批次边界拒绝标记完成，不停止其他健康训练。程序记录41次进度、完整argv、真实开始/结束、参数SHA/无参数梯度、标签替换、原控制重放、1281数组原回执。

数组记录四候选预测/近端项/目标、相对变化、六方向变化范数及余弦、残差、行与视频。独立审核器核对全SHA、原F候选、原控制重放、目标重建和oracle不劣容差，报告样本加权/视频等权风险及负控制。有限步oracle无提升不能证明整个机制无效，有提升不能保证无标签泛化。训练样本和DEV选模参考的偏差边界保留。

保存依赖：新root为 {plan['new_root']}；尚不存在部署证明。实际上传/执行前必须扩展原子capture覆盖此根及审核源/计划，然后真实CAPTURE_COMPLETE、receipt/ZIP全SHA/CRC/member及本地D/异节点CPU审核。现有capture14不覆盖新根，不能当新实验保存证据。当前仅本地D准备保存，不冒新远端捕获、诊断完成或整体完成。

许可恢复后先重复同一受阻查询验证正式工具路径；通过后才核预算并部署。隔离教师仍未启动，根据此诊断与教师独立误差证据决定后续正式训练。旧C最终和旧14:30缺口仍未解决。
'''
(outputs/'TRAIN_Oracle短实验执行状态.md').write_text(report,encoding='utf-8')
proof=dict(utc=utc.isoformat(),status='LOCAL_ORACLE_V2_PLAN_SOURCE_AST_CONSISTENCY_CHECKED_NOT_GPU',
           source_sha256=sha(work/'diagnose_train_oracle_v2.py'),plan_sha256=sha(work/'train_oracle_plan_v2.json'),
           auditor_sha256=sha(work/'audit_train_oracle_v1.py'),conditions=plan['conditions'],
           gpu_executed=False,formal_training_started=False,remote_command_executed=False)
(outputs/'TRAIN_Oracle实验本地独立准备核验.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2),encoding='utf-8')
state=outputs/'研究接续状态.md'
with state.open('a',encoding='utf-8') as f:
    f.write(f"\nUTC{utc.isoformat()}：用户授权实际实验，原SSH38957同查询再次被当前Granular.sandbox_approval=false拒绝，未发到远端/未换连接。已冻结TRAIN Oracle短诊断v2/21pins/原C2best37/六条件/2700秒+7200保存余量，并准备独立数组审核；仅AST/本地一致性，未GPU预检/部署/执行/训练。现有capture14不含新root，合法恢复后必须先扩展保存。证据：TRAIN_Oracle短实验执行状态.md；TRAIN_Oracle实验本地独立准备核验.json。\n")
base=Path('D:/CodexBackups/selective_flow_20261003_1105')
assert shutil.disk_usage(base).free>100*1024**2
dest=base/('train_oracle_preparation_'+stamp);dest.mkdir()
files=[work/n for n in ['diagnose_train_oracle_v1.py','diagnose_train_oracle_v2.py','freeze_train_oracle_plan_v1.py','freeze_train_oracle_plan_v2.py','train_oracle_plan_v1.json','train_oracle_plan_v2.json','audit_train_oracle_v1.py','prepare_train_oracle_v2.py','preserve_train_oracle_preparation_v1.py']]
files += [outputs/n for n in ['TRAIN_Oracle实验本地冻结核验.json','TRAIN_Oracle实验本地独立准备核验.json','TRAIN_Oracle短实验执行状态.md','研究接续状态.md','远端实验启动工具拒绝_'+stamp+'.json']]
manifest={}
for source_path in files:
    target=dest/source_path.name;shutil.copy2(source_path,target);assert sha(source_path)==sha(target)
    manifest[source_path.name]=dict(bytes=target.stat().st_size,sha256=sha(target),source=str(source_path))
(dest/'preservation_manifest.json').write_text(json.dumps(dict(utc=utc.isoformat(),scope='Local preparation only; no new remote capture',files=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(status='LOCAL_PREPARATION_D_SHA_VERIFIED',destination=str(dest),files=len(files),gpu_executed=False),ensure_ascii=True))
