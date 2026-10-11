import datetime,hashlib,json,shutil,zipfile,tomllib
from pathlib import Path
b=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');o=b/'outputs';d=Path('D:/CodexBackups/selective_flow_20261003_1105/matched_head_score201_actual_20261006T200020Z');c=d/'control'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
now=datetime.datetime.now(datetime.timezone.utc).isoformat();ledger=read(o/'研究建议交流接续.json')
batch={'batch_id':'matched_head201_actual_20261006T200020Z','actual_record_utc':now,'joint_sha256':sha(d/'complete_matched_head201_scores_D_B_CPU_joint.json'),'sent_once_after_D_original_other_CPU_joint':True,'thread_id':'01a10fcb-6663-70a2-9a76-60e5634d0c03','turn_id':'01a112d3-21aa-7b72-a774-00c8b2a999db','status':'IN_PROGRESS_NO_COMPLETE_REPORT_YET','wait_once':True,'cursor':'c9fdcd3f-7d4b-4678-809d-0742a5ab961e:7','new_advice_read_or_adopted':False,'scope':'new actual once201 all nine scores; development only; no credentials'}
ledger['sixteenth_actual_batch']=batch;write(o/'研究建议交流接续.json',ledger)
state=read(o/'完整流匹配U_W头实际接续.json');state['prediction_joint_D']='D:/CodexBackups/selective_flow_20261003_1105/matched_head_predict201_actual_20261006T195233Z';state['prediction_joint_sha256']='dc51168d9df07cf563d38f13c4216a6d4054f2956f42cadc5cfd3f678e08b2d4';state['advisor_batch']=batch;state['automation_verified']=True;state['no_open_remote_session']=True;write(o/'完整流匹配U_W头实际接续.json',state)
write(c/'actual_session_exit_receipt.json',{'actual_record_utc':now,'A_SSH':{'id':27749,'operation':'exit','actual_exit_code':0},'A_SFTP':{'id':52161,'operation':'bye','actual_exit_code':0},'B_SSH':{'id':28761,'operation':'exit','actual_exit_code':0},'B_SFTP':{'id':92646,'operation':'bye','actual_exit_code':0},'reuse_forbidden':True})
a=read(b/'work/matched_head201_actual_automation_update_args.json');t=tomllib.loads(Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'));assert t['prompt']==a['prompt'] and t['rrule']=='FREQ=MINUTELY;INTERVAL=10' and t['status']=='ACTIVE' and t['target_thread_id']==a['targetThreadId']
write(c/'actual_existing_automation_verified.json',{'actual_utc':now,'id':t['id'],'kind':t['kind'],'status':t['status'],'rrule':t['rrule'],'target_thread_id':t['target_thread_id'],'created_at':t['created_at'],'exact_saved_prompt_equals_tool_argument':True,'new_automation_created':False})
for n in ['完整流匹配U_W头201一次开发五指标实际结果.md','完整流匹配U_W头201一次开发五指标实际结果.json','完整流匹配U_W头实际接续.json','研究建议交流接续.json','研究接续状态.md']:shutil.copy2(o/n,c/n)
for n in ['matched_head201_actual_automation_update_args.json','matched_head201_actual_automation_update_result.json','record_matched_head201_actual_v1.py','finalize_matched_head201_control_v1.py']:shutil.copy2(b/'work'/n,c/n)
members={p.name:sha(p) for p in c.iterdir() if p.is_file() and p.name not in ['control_original_SHA.json','control_snapshot.zip','control_actual_archive_audit.json']};write(c/'control_original_SHA.json',members)
zpath=c/'control_snapshot.zip'
assert not zpath.exists()
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for name in [*members,'control_original_SHA.json']:z.write(c/name,name)
with zipfile.ZipFile(zpath) as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==len(members)+1
 for n,h in members.items():assert hashlib.sha256(z.read(n)).hexdigest()==h==sha(c/n)
write(c/'control_actual_archive_audit.json',{'status':'ACTUAL_COMPLETE_LOCAL_CONTROL_ORIGINAL_SHA_ZIP_CRC_UNIQUE_PASSED','actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'zip_sha256':sha(zpath),'members':len(members)+1,'remote_capture':False,'new_task_score':False,'remote_originals_preserved_in_separate_A_B_capsules':True})
print(json.dumps({'control_D':str(c),'members':len(members)+1,'advisor_status':batch['status'],'sessions_closed':True}))
