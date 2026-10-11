from pathlib import Path
import datetime,hashlib,json,shutil,subprocess,sys
p=Path('work/save_finite_short_state_v1.py');s=p.read_text(encoding='utf-8').replace('capture_soft_vector_v10.py','capture_soft_vector_v11.py').replace('0d1c6c4af5b8d7d7bb0a12236ba5d3895cf3d81a66cfb502dedfd6baf2ba0149','8281281e450689274c2558cbb25e517434c3b5438384b94fb7194159297df158').replace('audit_soft_snapshot_v10.py','audit_soft_snapshot_v11.py').replace('audit_finite_snapshot_full_extras_v2.py','audit_finite_snapshot_full_extras_v3.py').replace('finite_risk_snapshots_20261005T1538Z','finite_risk_snapshots_20261005T1541Z').replace('目录1538只ID','目录1541只ID').replace('v10增加.py成员后新捕获，不删或冒首版全通过。','v10增加.py后出现重复源码成员，仍保留而不当最终规范包；v11按同名同字节去重、独立审核强制成员唯一后新捕获，不删或冒首版全通过。')
fixed=Path('work/save_finite_short_state_v2.py');fixed.write_text(s,encoding='utf-8');subprocess.run([sys.executable,str(fixed)],check=True)
state=Path('outputs/研究接续状态.md');closure=Path('work/new_p4_checks/finite_session_closures_20261005T1541Z.json');c=json.loads(closure.read_text());assert len(c['closures'])==7 and all(v['exit_code']==0 for v in c['closures'].values())
with state.open('a',encoding='utf-8') as f:f.write('\n七本轮实际会话已关闭，SSH96572/12699/1803/额外只读93354、SFTP78308/13449/47616均明确exit/bye真实exit0。关闭证明work/new_p4_checks/finite_session_closures_20261005T1541Z.json。所有这些ID禁止write_stdin，下一需fresh真实password。capture11三真实UTC15:40内快照（见回执）已完整D/ZIP/成员唯一/SHA/原回执独立审核，A100/B100完成、C10失败分别审核，不套live。\n')
report=Path('outputs/有限任务风险实验分析.md')
with report.open('a',encoding='utf-8') as f:f.write('\n补充：原输入含行620 TRAIN32三模式重放及换标签0已通过，初始化/20更新/完整订单/400fixed共享期全部张量误差0已独立审核。修正版尚未正式启动，剩余单token专门double有限差/主辅隔离、新协议/预算冻结。capture9缺CPU.extra源码未作为完整联合证明，capture10重复同源码未当规范最终包；capture11全SHA/ZIP/成员唯一/旧新根与原CPU回执审核已通过，原包保留。七会话真实exit0关闭，未停止健康训练。\n')
now=datetime.datetime.now(datetime.timezone.utc);dest=Path('D:/CodexBackups/selective_flow_20261003_1105')/('finite_risk_records_'+now.strftime('%Y%m%dT%H%M%SZ'));dest.mkdir(exist_ok=False);names=set()
for pattern in ['*finite*','*single_token*']:
 for path in Path('work').glob(pattern):
  if path.is_file() and path.suffix in ['.py','.json','.npy']:names.add(path)
for path in Path('outputs').iterdir():
 if path.is_file() and (any(k in path.name for k in ['有限任务风险','单token解析分支']) or path.name=='研究接续状态.md'):names.add(path)
names.add(closure)
for path in Path('work/new_p4_checks').glob('finite_*audit.json'):names.add(path)
for path in sorted(names):
 d=dest/path;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,d)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();rows={}
for path in sorted(names):
 assert sha(path)==sha(dest/path);rows[str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
r=dict(status='FINITE_CURRENT_SOURCE_REPORTS_SHORT_STATE_AUDITS_AND_CLOSED_SESSIONS_PERMANENT_D_SHA_VERIFIED',utc=now.isoformat(),directory=str(dest),files=rows,scope='New records only; no repeated oldweights, no deletion. Fullmodels andCPU already separately verified. C v1 failed, C2notstarted, interventions pending. Research and rental saves continue.')
receipt=Path('outputs/有限任务风险研究资料与接续永久保存核验.json');receipt.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(receipt,dest/'backup_receipt.json');print(str(dest),len(rows))
