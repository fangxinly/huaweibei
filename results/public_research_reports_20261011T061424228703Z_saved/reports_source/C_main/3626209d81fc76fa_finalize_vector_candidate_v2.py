from pathlib import Path
import datetime,hashlib,json,shutil
root=Path.cwd(); out=root/'outputs'
now=datetime.datetime.now(datetime.timezone.utc)
proof=json.loads((out/'向量反馈候选几何与静态接口核验_v2.json').read_text(encoding='utf-8'))
paths=[root/'work/task_gradient_vector_candidate_v2.py',out/'task_gradient_vector_candidate_v2.py']
hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
assert hashes[0]==hashes[1]==proof['source_sha256']
check=root/'work/check_vector_candidate_geometry_v2.py'
assert hashlib.sha256(check.read_bytes()).hexdigest()==proof['checker_sha256']
disk={drive:shutil.disk_usage(drive+':/').free for drive in ('C','D')}
assert disk['D']>3_000_000_000
state=out/'研究接续状态.md'
backup=root/'work'/('研究接续状态_向量候选前_'+now.strftime('%Y%m%dT%H%M%SZ')+'.md')
backup.write_bytes(state.read_bytes())
s=state.read_text(encoding='utf-8')
first=s.splitlines()[2]
s=s.replace(first,'最新实际UTC'+now.strftime('%Y-%m-%d %H:%M')+'/BJ'+(now+datetime.timedelta(hours=8)).strftime('%H:%M')+'。本地向量候选已实现及参考检查，远端恢复依赖未解决。旧heartbeat运行轮数/PID不是最新状态；禁重注入旧长提示。',1)
s+='\nUTC'+now.strftime('%H:%M')+'新增未接入的任务梯度向量反馈候选v2：task_gradient_vector_candidate_v2.py、任务梯度向量反馈候选实现说明.md、向量反馈候选几何与静态接口核验_v2.json。预测无标签参考处任务梯度，移除供体消息估计一阶有害分量，fixed/scalar/projected匹配模块；float32接口避免阈值float16下溢。源码/交付/核验SHA一致。NumPy合成几何与AST接口通过，二阶风险增加反例成立；无PyTorch前反向/真实梯度隔离/完整模型接入/教师采集/GPU训练。既有A教师只能机制拟合，不称交叉拟合或独立泛化；先真实GPU机制、20更新预算及统一初始化/订单再冻结新实验。v1保留，现有v5源冻结不改。C100恢复与14:30缺口仍未解决，不连接旧地址；新增资料另永久D保存，不在先前239包内。D本轮实查剩余'+str(disk['D'])+'bytes，无删除。\n'
state.write_text(s,encoding='utf-8')
record={'status':'DELIVERED_VECTOR_SOURCE_AND_REFERENCE_PROOF_SHA_MATCH','checked_at':now.isoformat(),'source_sha256':hashes[0],'checker_sha256':proof['checker_sha256'],'free_bytes':disk,'torch_executed':False,'gpu_verified':False,'model_integrated':False,'training_started':False,'C_final_recovered':False,'lease_1430_capture_completed':False}
(out/'向量反馈候选交付一致性核验.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(record,ensure_ascii=False))
