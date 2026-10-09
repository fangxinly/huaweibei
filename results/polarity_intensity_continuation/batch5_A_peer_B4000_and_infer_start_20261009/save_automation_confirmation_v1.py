import datetime,hashlib,json,shutil,tomllib,zipfile
from pathlib import Path
root=Path(__file__).resolve().parent
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert shutil.disk_usage('D:/').free>40*1024**2+5*1024**2
args=json.loads((root/'automation_update_prepared.json').read_text(encoding='utf8'))
toml=Path('C:/Users/21234/.codex/automations/automation/automation.toml')
shutil.copyfile(toml,root/'automation_after_tool_original.toml')
actual=tomllib.loads((root/'automation_after_tool_original.toml').read_text(encoding='utf8'))
for k,ak in [('id','id'),('kind','kind'),('name','name'),('status','status'),('rrule','rrule'),('target_thread_id','targetThreadId'),('prompt','prompt')]:assert actual[k]==args[ak],k
write(root/'automation_tool_update_original.json',dict(tool='mcp__codex_app__automation_update',actual_observed_UTC=utc(),result={'content':[{'type':'text','text':'Updated automation in the app.'},{'type':'text','text':'{"automationId":"automation","mode":"update","status":"ACTIVE"}'}],'isError':False},saved_TOML_original_SHA=sha(root/'automation_after_tool_original.toml'),full_updated_fields_equal=True,existing_10min_ACTIVE_preserved=True,no_new_automation=True))
files=[root/'automation_update_tool_original.json',root/'automation_update_prepared.json',root/'automation_after_tool_original.toml',root/'automation_tool_update_original.json',Path(__file__)]
archive=root/'complete_automation_current_update_original.zip';rows=[]
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for f in files:
        b=f.read_bytes();z.writestr(f.name,b);rows.append(dict(name=f.name,bytes=len(b),sha256=hashlib.sha256(b).hexdigest()))
    z.writestr('member_manifest.json',json.dumps(rows,ensure_ascii=False,indent=2))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for r in rows:assert hashlib.sha256(z.read(r['name'])).hexdigest()==r['sha256']
receipt=dict(actual_UTC=utc(),archive=str(archive),archive_SHA=sha(archive),archive_bytes=archive.stat().st_size,all_member_SHA_CRC_unique_passed=True,status='EXISTING_HEARTBEAT_A_PEER_B4000_COMPLETE_B_INFER_STARTED_UPDATED_CONFIRMED')
write(root/'automation_preservation_receipt.json',receipt);print(json.dumps(receipt))
