from pathlib import Path
import datetime,hashlib,json,shutil
r=Path.cwd();o=r/'outputs';now=datetime.datetime.now(datetime.timezone.utc)
proof=json.loads((o/'向量反馈阈值连续性与消息梯度边界核验.json').read_text(encoding='utf-8'))
assert hashlib.sha256((r/'work/task_gradient_vector_candidate_v2.py').read_bytes()).hexdigest()==proof['source_sha256']
assert hashlib.sha256((o/'task_gradient_vector_candidate_v2.py').read_bytes()).hexdigest()==proof['source_sha256']
free={d:shutil.disk_usage(d+':/').free for d in ('C','D')};assert free['D']>3_000_000_000
p=o/'任务梯度向量反馈候选实现说明.md';s=p.read_text(encoding='utf-8')
s+='\n新增启动依赖（UTC'+now.strftime('%H:%M')+'）：硬阈值连续性反例及零处连续/精确半空间/保持消息三要求不能兼容的证明，见《向量反馈阈值连续性与梯度边界.md》。合法投影可能消除主任务消息梯度，GPU检查须对照解析Jacobian而非要求每样本非零。候选v2源保持不变，正式实验前须明确上述取舍并完成真实机制检查。\n'
p.write_text(s,encoding='utf-8')
p=o/'研究接续状态.md';s=p.read_text(encoding='utf-8');lines=s.splitlines()
lines[2]='最新实际UTC'+now.strftime('%Y-%m-%d %H:%M')+'/BJ'+(now+datetime.timedelta(hours=8)).strftime('%H:%M')+'。本地向量候选新增阈值连续性依赖，远端恢复依赖未解决。旧heartbeat轮数/PID不是最新状态；禁重注入旧长提示。'
s='\n'.join(lines)+'\n\nUTC'+now.strftime('%H:%M')+'数学边界新增：向量反馈阈值连续性与梯度边界.md/消息梯度边界核验.json。float32合成梯度变化约2e-10可使单位消息变化约1；零处保持消息、精确半空间约束与连续性不兼容。仅分母平滑会放松半空间保证，未实施/未选定。固定估计梯度时实际单通道Jacobian .125(I−nnᵀ)，法向上游梯度为零可属预期，正交探针才可核对传播；NumPy有限差分通过，无Torch/GPU。正式训练须先解决上述取舍、真实尺度/阈值切换及解析梯度验证，v2源码不改。新记录与修订说明另永久D保存；C恢复/14:30缺口不变，无新连接、数据访问或训练。D本轮实际剩余'+str(free['D'])+'bytes。\n'
p.write_text(s,encoding='utf-8')
print(json.dumps({'time':now.isoformat(),'free_bytes':free,'candidate_source_unchanged':True},ensure_ascii=False))
