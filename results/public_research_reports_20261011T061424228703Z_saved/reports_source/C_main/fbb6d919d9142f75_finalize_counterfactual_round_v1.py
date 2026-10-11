from pathlib import Path
import datetime,hashlib,json,shutil,tomllib
r=Path(__file__).resolve().parents[1];o=r/'outputs';state=o/'研究接续状态.md';s=state.read_text(encoding='utf-8').replace('`n','\n');state.write_text(s,encoding='utf-8')
auto=tomllib.loads(Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'));assert auto['status']=='ACTIVE' and auto['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58' and auto['rrule']=='RRULE:FREQ=MINUTELY;INTERVAL=10'
assert 'capture_counterfactual_followup_v4_20261005T0545Z.py' in auto['prompt'] and '租期1340动态保存完成记录.json' in auto['prompt'] and '14:10/14:30' in auto['prompt']
now=datetime.datetime.now(datetime.timezone.utc).isoformat();ap={k:auto[k] for k in ['id','name','status','rrule','target_thread_id','updated_at']};ap.update(verified_at=now,prompt_chars=len(auto['prompt']),prompt_sha256=hashlib.sha256(auto['prompt'].encode()).hexdigest(),phase='v5 live64/68/46;13:40 done;14:10/14:30 pending; no overall completion')
with (o/'主任务反事实效用本轮监管核验.json').open('x',encoding='utf-8') as f:json.dump(ap,f,ensure_ascii=False,indent=2)
close=json.loads((o/'主任务反事实效用本轮连接关闭核验.json').read_text(encoding='utf-8'));assert len(close['rows'])==6 and all(x['exit_code']==0 for x in close['rows'])
d=Path('D:/CodexBackups/selective_flow_20261003_1105/metadata_202610050552Z/late_records');assert shutil.disk_usage(d.parent).free>20_000_000;d.mkdir(exist_ok=False)
names=['研究接续状态.md','主任务反事实效用本轮连接关闭核验.json','主任务反事实效用本轮监管核验.json','主任务反事实效用本轮最终资料本地核验.json','主任务反事实效用本轮最终资料独立核验.json','租期1340动态保存完成记录.json','租期1340永久本地资料保存核验.json','租期1340独立节点资料保存核验.json','C临时完整权重本地保存核验_1342.json','C临时完整权重独立保存核验_1342.json']
files=[]
for name in names:
 p=o/name;q=d/name;shutil.copyfile(p,q);b=p.read_bytes();digest=hashlib.sha256(b).hexdigest();assert digest==hashlib.sha256(q.read_bytes()).hexdigest();files.append({'name':name,'bytes':len(b),'sha256':digest,'destination':str(q)})
report={'verified_at':now,'status':'LATE_CLOSURE_AUTOMATION_RECEIPTS_AND_LATEST_STATE_PERMANENT_D_ALL_SHA_VERIFIED','destination':str(d),'files':files,'all_six_connections_exit0':True,'lease1340_done':True,'pending_deadlines_beijing':['14:10','14:30'],'limits':'Late files are outside immutable194-member ZIP; C provisional weight has separate permanent and independent proof. Training and overall research still incomplete.'}
p=o/'主任务反事实效用本轮最终记录保存核验.json'
with p.open('x',encoding='utf-8') as f:json.dump(report,f,ensure_ascii=False,indent=2)
shutil.copyfile(p,d/p.name);assert hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256((d/p.name).read_bytes()).hexdigest()
print(json.dumps({'status':report['status'],'files':len(files),'state_chars':len(s),'drive_D_free_bytes':shutil.disk_usage(d).free}))
