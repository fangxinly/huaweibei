from pathlib import Path
import datetime,hashlib,json,shutil,tomllib,zipfile
base=Path(__file__).parent.parent;out=base/'outputs';dest=Path('D:/CodexBackups/selective_flow_20261003_1105/scalar_geometry_actual_20261006T060719Z');records=dest/'research_records'
sha=lambda b:hashlib.sha256(b).hexdigest();now=datetime.datetime.now(datetime.timezone.utc).isoformat()
assert shutil.disk_usage(dest).free>1073741824
prompts=json.loads((records/'communication_prompts.json').read_text(encoding='utf-8'))
automation=tomllib.loads(Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'))
assert automation['id']=='automation' and automation['kind']=='heartbeat' and automation['status']=='ACTIVE'
assert automation['rrule']=='FREQ=MINUTELY;INTERVAL=10' and automation['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58'
assert automation['prompt']==prompts['automation_prompt']
coord=json.loads((out/'研究建议交流接续.json').read_text(encoding='utf-8'))
batches=coord['evidence_batches'];assert len(batches)==1 and batches[0]['status']=='ready_to_send_after_preservation'
assert batches[0]['report_sha256']==sha(Path(batches[0]['report']).read_bytes())
batches[0].update(status='sent',sent_utc=prompts['sent_utc'],send_prompt_sha256=sha(prompts['second_review_prompt'].encode()),send_tool_result=prompts['send_tool_result'],prompt_preserved=str(records/'communication_prompts.json'),repeat_send_allowed=False)
coord['updated_at_utc']=now
coord['review_thread'].update(status='second_review_in_progress',last_wait_cursor=prompts['review_snapshot']['cursor'],current_turn=prompts['review_snapshot']['turn_id'])
coord['first_sent_prompt_path']=str(records/'preceding_state/研究建议交流接续.json')
coord.pop('first_sent_prompt',None)
coord['automation'].update(result='Official tool updated existing ACTIVE heartbeat; local saved fields independently matched.',updated_at_utc=now,prompt_sha256=sha(automation['prompt'].encode()),rrule=automation['rrule'],duplicate_automation_created=False)
(out/'研究建议交流接续.json').write_text(json.dumps(coord,ensure_ascii=False,indent=2),encoding='utf-8')
state=out/'研究接续状态.md';text=state.read_text(encoding='utf-8')
text=text.replace('新实质结果D/CPU保存后只向同聊天发送一次','本轮实质结果D/CPU保存后UTC06:29:39已向同聊天发送第二批一次')
text=text.replace('未完成第二次建议不冒已返回。','第二次建议实际inProgress，cursor :5，未返回不冒已返回；十分钟原自动任务已官方更新并独立核保存字段，未新建自动任务。')
state.write_text(text,encoding='utf-8')
snapshot=records/'coordination_after_dispatch';snapshot.mkdir(exist_ok=False)
members={}
sources=[('研究建议交流接续.json',out/'研究建议交流接续.json'),('研究接续状态.md',state),('communication_prompts.json',records/'communication_prompts.json'),('record_scalar_geometry_review_dispatch_v1.py',Path(__file__))]
with zipfile.ZipFile(snapshot/'snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
    for name,path in sources:
        data=path.read_bytes();(snapshot/name).write_bytes(data);z.writestr(name,data);members[name]=dict(bytes=len(data),sha256=sha(data))
    z.writestr('member_manifest.json',json.dumps(members,ensure_ascii=False,indent=2))
with zipfile.ZipFile(snapshot/'snapshot.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,item in members.items():assert sha(z.read(name))==item['sha256']==sha((snapshot/name).read_bytes())
receipt=dict(status='REVIEW_SECOND_BATCH_SENT_DEDUP_AND_EXISTING_AUTOMATION_SAVED_FIELDS_VERIFIED',actual_utc=now,batch=batches[0],second_review_status='inProgress_not_completed',automation_id='automation',automation_prompt_sha256=sha(automation['prompt'].encode()),rrule=automation['rrule'],target_thread_id=automation['target_thread_id'],new_automation_or_chat_created=False,zip_sha256=sha((snapshot/'snapshot.zip').read_bytes()),members=members,free_C_bytes=shutil.disk_usage('C:/').free,free_D_bytes=shutil.disk_usage(dest).free)
(snapshot/'receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'首步几何实证独立建议发送与自动任务接续核验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:receipt[k] for k in ['status','actual_utc','second_review_status','new_automation_or_chat_created','zip_sha256','free_C_bytes','free_D_bytes']}))
