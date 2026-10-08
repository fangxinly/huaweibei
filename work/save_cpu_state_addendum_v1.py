from pathlib import Path
import datetime, hashlib, json, re, shutil
r=Path(__file__).resolve().parents[1];p=r/'outputs/研究接续状态.md';now=datetime.datetime.now(datetime.timezone.utc)
dest=sorted(Path('D:/CodexBackups/selective_flow_20261003_1105').glob('jacobian_cpu_records_*'))[-1]
free={n:shutil.disk_usage(n+':/').free for n in ['C','D']}
s=p.read_text(encoding='utf-8')
s=re.sub(r'当前实查C \d+bytes/D \d+bytes',f"最新UTC{now.isoformat()}实查C {free['C']}bytes/D {free['D']}bytes",s)
s+='\n本轮13份报告/源码/接续/六关闭证据已永久D逐文件SHA核验，目录'+dest.name+'，输出Jacobian独立CPU保存接续资料永久核验_20261005T141703Z.json。automation工具已确认更新为ACTIVE十分钟，下一有限风险模块机制预检；不重复已完成CPU保存。\n'
p.write_text(s,encoding='utf-8');target=dest/('研究接续状态_最终_'+now.strftime('%Y%m%dT%H%M%SZ')+'.md');target.write_bytes(p.read_bytes())
h=hashlib.sha256(p.read_bytes()).hexdigest();assert hashlib.sha256(target.read_bytes()).hexdigest()==h
proof=dict(status='FINAL_SHORT_STATE_PERMANENT_SHA_VERIFIED',utc=now.isoformat(),source=str(p),destination=str(target),sha256=h,free=free)
with (dest/('state_addendum_'+now.strftime('%Y%m%dT%H%M%SZ')+'.json')).open('x',encoding='utf-8') as f:json.dump(proof,f,ensure_ascii=False,indent=2)
print(proof['status'])
