import hashlib,json,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
D=Path(json.loads((ROOT/'work/squared_risk_residual_math_20261006T1259Z/permanent_D_location.json').read_text())['directory'])
assert D.resolve().is_relative_to(Path('D:/CodexBackups/selective_flow_20261003_1105').resolve())
utc=datetime.now(timezone.utc).isoformat()
space={p:shutil.disk_usage(p).free for p in ('C:/','D:/')}
assert space['D:/']>100*1024**2
automation=json.loads((D/'official_automation_tool_result_and_review_wait_observation.json').read_text(encoding='utf-8'))
assert automation['automation_result'].get('isError') is False
statuses=[json.loads(c['text']) for c in automation['automation_result']['content'] if c.get('type')=='text' and c.get('text','').startswith('{')]
assert any(x.get('automationId')=='automation' and x.get('status')=='ACTIVE' for x in statuses)
ledger_path=ROOT/'outputs/研究建议交流接续.json';ledger=json.loads(ledger_path.read_text(encoding='utf-8'))
batch=ledger['squared_risk_residual_math_decision']
assert batch['sent_to_review'] is True and Path(batch['permanent_D'])==D
obs=automation['review_wait_observation']
ledger['review_thread']['current_turn']=obs['turnId']
ledger['review_thread']['last_wait_cursor']=obs['cursor']
ledger['review_thread']['status']='eighth_math_decision_in_progress_after_one_wait'
batch['review_status']='IN_PROGRESS_COMPLETE_REPORT_NOT_READ'
batch['official_automation_receipt']=str(D/'official_automation_tool_result_and_review_wait_observation.json')
ledger['automation']['updated_at_utc']=utc
ledger['automation']['prompt_sha256']=hashlib.sha256(automation['automation_prompt'].encode('utf-8')).hexdigest()
ledger['automation']['original_tool_receipt']=batch['official_automation_receipt']
ledger['updated_at_utc']=utc;ledger['updated_utc']=utc
ledger_path.write_text(json.dumps(ledger,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
state=ROOT/'outputs/研究接续状态.md';t=state.read_text(encoding='utf-8')
t=t.replace(t.splitlines()[0],f'更新UTC {utc}。只读短接续/按需原证据，不重注入历史。研究、后续实验和租期保存未整体完成。',1)
t=t.replace('第八数学决策批已发送，回复尚未审阅。','第八数学决策批已发且一次wait仍inProgress，完整报告未审阅；turn01a11156-3c19-74a3-bec8-3a1388ee0db5/cursor29按交流JSON下轮一次读，不冒已返回。')
state.write_text(t,encoding='utf-8')
out=D/'final_records';out.mkdir(exist_ok=False)
paths=[state,ledger_path,ROOT/'outputs/平方误差收益与残差估计的条件信息及匹配对照决策.md',D/'original_preservation_and_review_send_tool_receipts.json',D/'official_automation_tool_result_and_review_wait_observation.json',Path(__file__)]
members={}
for p in paths:
 assert p.name not in members
 q=out/p.name;shutil.copyfile(p,q);assert sha(q)==sha(p)
 members[p.name]={'bytes':q.stat().st_size,'sha256':sha(q)}
pkg=D/'final_records.zip'
with zipfile.ZipFile(pkg,'x',zipfile.ZIP_DEFLATED) as z:
 for name in sorted(members):z.write(out/name,name)
with zipfile.ZipFile(pkg) as z:
 assert z.testzip() is None and set(z.namelist())==set(members) and len(z.namelist())==len(set(z.namelist()))
 for name,v in members.items():assert hashlib.sha256(z.read(name)).hexdigest()==v['sha256']
receipt={'status':'MATH_DECISION_FINAL_SHORT_STATE_LEDGER_TOOL_RECORDS_D_SHA_ZIP_CRC_PASSED','actual_utc':utc,'free_bytes_before':space,'members':members,'zip_sha256':sha(pkg),'review_complete_report_read':False,'actual_GPU_or_new_task_scores':False,'new_remote_capture':False,'whole_research_or_lease_complete':False}
(D/'final_records_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':receipt['status'],'directory':str(D),'members':len(members),'actual_utc':utc,'review_status':'inProgress','automation_status':'ACTIVE','new_task_scores':False}))
