import datetime,hashlib,json,shutil,sys,tomllib,zipfile
from pathlib import Path
R=Path(__file__).parent.parent;O=R/'outputs';W=R/'work'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
actual=datetime.datetime.strptime(sys.argv[1],'%Y%m%dT%H%M%SZ').replace(tzinfo=datetime.timezone.utc).isoformat()
saved=tomllib.loads(Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'));args=read(W/'matched_head_preparation_existing_automation_update_args.json')
assert saved['prompt']==args['prompt'] and saved['rrule']=='FREQ=MINUTELY;INTERVAL=10' and saved['status']=='ACTIVE' and saved['id']=='automation' and saved['kind']=='heartbeat' and saved['created_at']==1790963097773 and saved['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58'
proof={'status':'EXISTING_HEARTBEAT_OFFICIAL_UPDATE_AND_SAVED_FIELDS_VERIFIED','actual_utc':actual,'prompt_sha256':hashlib.sha256(saved['prompt'].encode()).hexdigest(),'frequency_minutes':10,'original_creation_and_thread_preserved':True,'new_automation_or_settings_change':False,'original_tool_receipt':str(W/'matched_head_preparation_automation_actual_update.json')}
ledger=read(O/'研究建议交流接续.json');ledger['updated_at_utc']=actual;ledger['automation'].update(updated_at_utc=actual,prompt_sha256=proof['prompt_sha256'],result=proof['status'],original_tool_receipt=proof['original_tool_receipt']);write(O/'研究建议交流接续.json',ledger)
prep=read(O/'完整流匹配U_W头协议本地准备接续.json');dest=Path(prep['D'])/('control_records_'+sys.argv[1]);dest.mkdir(exist_ok=False)
files=[O/'研究接续状态.md',O/'研究建议交流接续.json',O/'完整流匹配U_W头协议本地准备接续.md',O/'完整流匹配U_W头协议本地准备接续.json',O/'第十五次建议独立决策与匹配头准备.md',W/'matched_head_preparation_automation_actual_update.json',W/'matched_head_preparation_existing_automation_update_args.json',Path(__file__)]
for p in files:shutil.copyfile(p,dest/p.name)
write(dest/'automation_verification.json',proof)
manifest={p.name:sha(p) for p in dest.iterdir()};write(dest/'manifest.json',{'actual_utc':actual,'files':manifest,'fresh_C_D':{x:shutil.disk_usage(x+':/').free for x in ('C','D')},'new_remote_capture_or_othernode_CPU':False})
with zipfile.ZipFile(dest/'records.zip','x',zipfile.ZIP_DEFLATED) as z:
 for n in manifest:z.write(dest/n,n)
 z.write(dest/'manifest.json','manifest.json')
with zipfile.ZipFile(dest/'records.zip') as z:
 assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
 for n,h in manifest.items():assert sha(dest/n)==h and hashlib.sha256(z.read(n)).hexdigest()==h
print(json.dumps({'status':'LOCAL_PREPARATION_REVIEW15_AND_EXISTING_HEARTBEAT_CONTROL_RECORDS_D_VERIFIED','D':str(dest),'zip_sha256':sha(dest/'records.zip'),'new_task_experiment_or_performance':False,'overall_research_or_preservation_complete':False}))
