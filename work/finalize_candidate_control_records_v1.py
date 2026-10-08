import datetime,json,hashlib,shutil,tomllib,zipfile
from pathlib import Path
R=Path(__file__).parent.parent;O=R/'outputs';D=Path('D:/CodexBackups/selective_flow_20261003_1105/fixed_head_candidate_actual_20261006T191103Z')
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'));sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
saved=tomllib.loads(Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'));args=read(R/'work/candidate_existing_automation_update_args.json')
assert saved['prompt']==args['prompt'] and saved['id']=='automation' and saved['kind']=='heartbeat' and saved['status']=='ACTIVE' and saved['rrule']=='FREQ=MINUTELY;INTERVAL=10' and saved['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58' and saved['created_at']==1790963097773
now=datetime.datetime.now(datetime.timezone.utc).isoformat();proof={'status':'EXISTING_HEARTBEAT_OFFICIAL_UPDATE_AND_SAVED_FIELDS_VERIFIED','actual_verified_utc':now,'id':'automation','prompt_sha256':hashlib.sha256(saved['prompt'].encode()).hexdigest(),'rrule':saved['rrule'],'created_at_preserved':True,'new_automation_created':False,'official_source_checked':'https://developers.openai.com/cookbook/examples/agents_sdk/agent_improvement_loop','schema_source':'Exposed official app tool instructions, not a claimed web REST schema'}
prep=read(O/'完整流匹配U_W头协议本地准备接续.json');l=read(O/'研究建议交流接续.json');l['updated_at_utc']=now;l['automation'].update(updated_at_utc=now,prompt_sha256=proof['prompt_sha256'],result=proof['status'],original_tool_receipt=str(R/'work/candidate_existing_automation_actual_update.json'));l['next_matched_heads_local_preparation']=prep;write(O/'研究建议交流接续.json',l)
p=O/'研究接续状态.md';p.write_text('本地准备UTC '+prep['actual_utc']+'：匹配U/W共同FIT统计与同ridge闭式核心、9固定读出和物理文件SHA评价标签门已写，合成独立增广最小二乘差3.33e-16/零常数feature/毒化门过；D '+prep['D']+'保存，仅准备非真实拟合/201推理/新成绩。正式runner、原标签提取与chronology、201原输入collector/CPU/capture尚待全冻后执行。第15实际候选批已一次send/wait，当前inProgress/cursor:5未完整审阅；第14已读，不因等待停研究/保存。接续完整流匹配U_W头协议本地准备接续.md/json。原10分钟ACTIVE官方更新已实核，未创建新任务或改设置。\n\n'+p.read_text(encoding='utf-8'),encoding='utf-8')
dest=D/('final_control_records_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));dest.mkdir(exist_ok=False)
names=[O/'研究接续状态.md',O/'研究建议交流接续.json',O/'完整流单候选232零标签实际结果.md',O/'完整流单候选232零标签实际结果.json',O/'完整流单候选232零标签采集实际接续.json',O/'完整流匹配U_W头协议本地准备接续.md',O/'完整流匹配U_W头协议本地准备接续.json',R/'work/candidate_existing_automation_actual_update.json',R/'work/candidate_existing_automation_update_args.json',R/'work/fixed_head_candidate_batch15_prompt.txt',R/'work/fixed_head_candidate_batch15_actual_send_and_wait.json',Path(__file__)]
for f in names:shutil.copyfile(f,dest/f.name)
write(dest/'automation_actual_saved_verification.json',proof)
mf={p.name:sha(p) for p in dest.iterdir()};write(dest/'manifest.json',{'actual_utc':now,'files':mf,'fresh_C_D':{x:shutil.disk_usage(x+':/').free for x in ['C','D']},'new_remote_capture_or_weights_CPU_audit_for_these_local_control_records':False})
with zipfile.ZipFile(dest/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in mf:z.write(dest/n,n)
 z.write(dest/'manifest.json','manifest.json')
with zipfile.ZipFile(dest/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in mf.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
print(json.dumps({'status':'CANDIDATE_REAL_JOINT_PLUS_LOCAL_CONTROL_PREP_ADVISOR_AUTOMATION_D_RECORDS_VERIFIED','D':str(dest),'zip_SHA':sha(dest/'records.zip'),'files':len(mf),'overall_research_or_CaReFlow_target_complete':False,'next_actual_fit_or201_not_started':True},ensure_ascii=False))
