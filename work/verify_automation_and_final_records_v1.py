from pathlib import Path
import datetime,hashlib,json,shutil,tomllib,zipfile
cwd=Path(__file__).parent.parent;out=cwd/'outputs';d=Path('D:/CodexBackups/selective_flow_20261003_1105/native_radius_cal_mechanism_actual_20261006T072031Z');rec=d/'research_records'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();tool=json.loads((rec/'official_automation_update.json').read_text(encoding='utf-8'))
saved=tomllib.loads(Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'))
assert saved['id']=='automation' and saved['kind']=='heartbeat' and saved['name']=='多模态流研究与P4租期保存' and saved['status']=='ACTIVE'
assert saved['rrule']=='FREQ=MINUTELY;INTERVAL=10' and saved['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58' and saved['created_at']==1790963097773
assert saved['prompt']==tool['prompt']
proof={'status':'EXISTING_HEARTBEAT_OFFICIAL_UPDATE_AND_SAVED_FIELDS_VERIFIED','id':'automation','updated_utc':datetime.datetime.fromtimestamp(saved['updated_at']/1000,datetime.timezone.utc).isoformat(),
    'prompt_sha256':hashlib.sha256(saved['prompt'].encode()).hexdigest(),'rrule':saved['rrule'],'target_thread_id':saved['target_thread_id'],'created_at_preserved':True,'status_value':saved['status'],'new_automation_created':False}
(rec/'automation_update_verified.json').write_text(json.dumps(proof,indent=2))
coord=json.loads((out/'研究建议交流接续.json').read_text(encoding='utf-8'));coord['automation'].update(updated_at_utc=proof['updated_utc'],prompt_sha256=proof['prompt_sha256'],result=proof['status'])
coord['review_thread'].update(last_wait_cursor=tool['current_review_cursor'],current_turn=tool['current_review_turn'],status='fifth_review_inProgress_not_completed')
(out/'研究建议交流接续.json').write_text(json.dumps(coord,ensure_ascii=False,indent=2),encoding='utf-8');shutil.copy2(out/'研究建议交流接续.json',rec/'研究建议交流接续.json')
for name in ['session_closure.json','nonF_CAL_coverage_analysis.json']:assert (d/name).is_file()
assert sha(d/'diagnose_native_radius_cal_v1.py')=='797f3a7fb661cbcdadd86126b017d8d336802ae185e51d1ef5c1dcb09e68355e'
shutil.copy2(Path(__file__),rec/Path(__file__).name)
manifest={p.relative_to(rec).as_posix():{'bytes':p.stat().st_size,'sha256':sha(p)} for p in rec.rglob('*') if p.is_file() and p.suffix!='.zip' and p.name!='records_manifest.json'}
doc={'status':'LOCAL_RESEARCH_FINAL_RECORDS_ALL_SHA_AND_ZIP_VERIFIED','files':manifest,'no_remote_capture_or_CPU_rerun_claim_for_new_local_reports':True,'fresh_free_D':shutil.disk_usage(d).free,'fresh_free_C':shutil.disk_usage(cwd).free}
(rec/'records_manifest.json').write_text(json.dumps(doc,ensure_ascii=False,indent=2),encoding='utf-8')
with zipfile.ZipFile(rec/'research_records.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for name in manifest:z.write(rec/name,name)
    z.write(rec/'records_manifest.json','records_manifest.json')
with zipfile.ZipFile(rec/'research_records.zip') as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
    for name,it in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==it['sha256']
print(json.dumps({'automation':proof['status'],'local_record_files':len(manifest),'zip_sha256':sha(rec/'research_records.zip'),'fresh_free_D':doc['fresh_free_D'],'whole_research_complete':False}))
