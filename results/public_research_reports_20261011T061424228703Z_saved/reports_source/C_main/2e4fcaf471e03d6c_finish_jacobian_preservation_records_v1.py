from pathlib import Path
import datetime, hashlib, json
r=Path(__file__).resolve().parents[1];o=r/'outputs';w=r/'work'
now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
proof=json.loads((o/'Jacobian三臂七文件异训练与组装节点CPU保存核验.json').read_text(encoding='utf-8'))
capture=json.loads((o/'Jacobian独立CPU保存后联合快照核验.json').read_text(encoding='utf-8'))
closed=json.loads((w/'new_p4_checks/session_closures_20261005T141523Z.json').read_text(encoding='utf-8'))
assert len(proof['receipts'])==3 and all(x['exit_code']==0 for x in closed['rows'])
assert capture['status']=='THREE_FRESH_SNAPSHOTS_AND_NEW_CPU_PRESERVATION_INDEPENDENTLY_AUDITED'
state=o/'研究接续状态.md';old=state.read_text(encoding='utf-8')
(w/('研究接续状态_CPU保存前_'+stamp+'.md')).write_text(old,encoding='utf-8')
text=old
first=text.splitlines()[2]
text=text.replace(first,'最新真实UTC'+now.isoformat()+'。新P4 seed91815及Jacobian91816两组三臂各100轮/20冻结诊断/六新整weights永久D均完成。91816新七文件加5extra异训练与组装节点CPU三原回执已下载独立审核，保存依赖已收尾。下一有限任务风险候选仅设计，尚未实施或启动；整体研究及24h租期保存继续。',1)
start=text.index('UTC13:50:54 SSH')
end=text.index('\n\n',start)
text=text[:start]+'UTC14:15:23 SSH95207/42950/67060和SFTP33768/1710/55000均明确exit/bye实际exit0，原关闭证明work/new_p4_checks/session_closures_20261005T141523Z.json。旧13:50六会话也已关闭，全部旧ID不可write_stdin。下一fresh SSH/SFTP只在实际password提示后用本聊天三新P4人类消息原凭据，禁密码文件/命令/自动提示/猜测。训练、组装均自然退休，无健康任务被停。'+text[end:]
start=text.index('下一先按preservation_manifest.json')
end=text.index('\n\n',start)
text=text[:start]+'91816三新CPU保存已完成：A到B/B到C/C到B，全部目标同时不同于原训练节点和新A组装节点；这不是三不同目标的循环轮转。根/data/coding/jacobian_preservation_20261005T1406Z，七文件加5extras完整SHA/ZIPCRC/CPU457有限张量/供体头与addon完全相等/100最小DEV选模/229形状通过，CUDA未初始化。三实际原回执均下载到D每node目录，独立核验Jacobian三臂七文件异训练与组装节点CPU保存核验.json；不混用91815回执。'+text[end:]
start=text.index('当前已三节点部署原子capture_soft_vector_v3.py')
end=text.index('\n\n',start)
text=text[:start]+'当前三节点新部署原子capture_soft_vector_v4.py SHA43dd5cd61f5f56c6654a2c7d3e9c631b3f9fda4e7cb4bfa7ade7bdb440b308f8，新增91816 CPU保存根及原验证源码，旧v3保留。参数--root公共源根 --deployment当前91816根 --stamp真实clock新唯一ID；等实际CAPTURE_COMPLETE/exit0再先receipt后ZIP，work/audit_soft_snapshot_v4.py独立审核。实际A/C UTC14:11:26–27、B14:14:09三快照已永久D jacobian_preservation_snapshots_20261005T1415Z，全包SHA/ZIP/memberSHA/源协议/完整订单/100选模/新CPU七加5成员原回执均审核通过，Jacobian独立CPU保存后联合快照核验.json。GPU UUID符合三新节点、compute均空、实际训练argv为null且100完成，不能套live。full大文件仅freshSHA引用，不冒充再次下载。未来新增保存/实验根需新capture版本扩展。早1328不完整下载拒收保留；后verified及13:33原子快照保留，目录stamp仅ID不冒充时刻。'+text[end:]
text=text.replace('仅设计，先收尾独立CPU保存，再实现/冻结','仅设计，独立CPU保存已收尾；下一实现/冻结')
text=text.replace('91816独立CPU七文件原回执仍待执行，保存依赖未全部结束。','此项后续已完成，以上14:15状态及新CPU回执为准。')
text+='\n本轮无新训练启动。下一先实现有限风险模块并真实机制预检，不重跑已经完成六100轮。C/D最新空间见真实快照及逐次实查，旧C最终恢复缺口/旧14:30缺口均未解决。\n'
state.write_text(text,encoding='utf-8')
report=o/'Jacobian方向约束三臂实验分析.md';s=report.read_text(encoding='utf-8')
s=s.replace('第二组独立CPU副本与原回执尚待执行。','第二组七文件加5extras已按A到B/B到C/C到B保存，CPU全SHA/ZIPCRC/有限完整张量及addon对应、100选模和229预测检查通过，三原回执已下载独立审核。全部目标同时不同于原训练节点及新A组装节点；这不是三个不同目标的循环轮转。')
s+='\n本轮保存证据：Jacobian三臂七文件异训练与组装节点CPU保存核验.json、Jacobian独立CPU保存后联合快照核验.json。三快照实际UTC14:11:26–27/14:14:09，目录ID不冒充同时采集。所有六传输明确exit/bye且exit0；未重训旧实验。\n'
report.write_text(s,encoding='utf-8')
design=o/'有限任务风险反馈下一步设计.md';s=design.read_text(encoding='utf-8')
s=s.replace('依赖：第二组整weights/独立CPU保存先收尾；','依赖进展：第二组整weights/独立CPU保存及三原回执已经收尾。下一先实现真实有限风险模块及机制预检；')
design.write_text(s,encoding='utf-8')
dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('jacobian_cpu_records_'+stamp);dest.mkdir(exist_ok=False)
files=[state,report,design,o/'Jacobian三臂七文件异训练与组装节点CPU保存核验.json',o/'Jacobian独立CPU保存后联合快照核验.json']
files += [w/n for n in ['capture_soft_vector_v4.py','audit_soft_snapshot_v4.py','verify_soft_preservation_cpu_v1.py','audit_jacobian_cpu_receipts_v1.py','prepare_jacobian_preservation_capture_v4.py','run_jacobian_preservation_audits_v1.py','finish_jacobian_preservation_records_v1.py']]
files += [w/'new_p4_checks/session_closures_20261005T141523Z.json']
rows=[]
for p in files:
    target=dest/p.name;b=p.read_bytes();target.write_bytes(b);h=hashlib.sha256(b).hexdigest()
    assert hashlib.sha256(target.read_bytes()).hexdigest()==h
    rows.append(dict(source=str(p),destination=str(target),bytes=len(b),sha256=h))
saved=dict(status='NEW_CPU_PRESERVATION_REPORTS_SOURCES_AND_SHORT_STATE_PERMANENT_ALL_SHA_VERIFIED',utc=now.isoformat(),rows=rows)
for p in [dest/'late_manifest.json',o/('Jacobian独立CPU保存接续资料永久核验_'+stamp+'.json')]:
    with p.open('x',encoding='utf-8') as f:json.dump(saved,f,ensure_ascii=False,indent=2)
print(json.dumps(dict(status=saved['status'],files=len(rows),directory=str(dest))))
