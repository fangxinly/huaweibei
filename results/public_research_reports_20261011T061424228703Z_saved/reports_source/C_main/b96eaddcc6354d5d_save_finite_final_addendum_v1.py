from pathlib import Path
import json,datetime,hashlib,shutil
rpath=Path('outputs/有限任务风险研究资料与接续永久保存核验.json');r=json.loads(rpath.read_text(encoding='utf-8'));dest=Path(r['directory']);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
for name,x in r['files'].items():assert sha(Path(name))==x['sha256']==sha(dest/name)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();state=Path('outputs/研究接续状态.md')
with state.open('a',encoding='utf-8') as f:f.write('\n最新UTC'+now+'：automation工具再次确认ACTIVE十分钟，提示已更新为CPU完成/capture11/下一C2二阶与新协议冻结，避免旧提示重复保存。最新实查C9869807616bytes、D23828008960bytes；无删除。61份新source/报告/短状态/原审核/关闭证据已永久D逐SHA，目录finite_risk_records_20261005T154319Z；最新附记同步SHA见有限任务风险研究资料与接续永久保存核验.json。\n')
extra=Path('work/new_p4_checks/finite_final_addendum_20261005T1544Z.json');extra.write_text(json.dumps(dict(utc=now,automation_tool_confirmed='ACTIVE ten-minute prompt updated',disk_free={'C':9869807616,'D':23828008960},all_sessions_closed_exit0=True,candidate_c2_formal_started=False,current_pending=['single-token-specific double finite difference/main-aux isolation','amended C2 source/protocol/budget freeze and100 formal','new20 frozen interventions','24hour future dynamic saves'],scope='No overall completion claim, no deletion, no oldweight retransmission.'),indent=2),encoding='utf-8')
for path in [state,extra,Path(__file__)]:
 d=dest/path;d.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,d);assert sha(path)==sha(d);r['files'][str(path)]=dict(bytes=path.stat().st_size,sha256=sha(path))
r['last_addendum_utc']=now;rpath.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(rpath,dest/'backup_receipt.json')
for name,x in r['files'].items():assert sha(Path(name))==x['sha256']==sha(dest/name)
print('FINAL_ADDENDUM_AND_ALL_RECORDS_SHA_VERIFIED',len(r['files']))
