from pathlib import Path
import datetime,hashlib,json,shutil,tomllib
b=Path(__file__).resolve().parents[1];a=tomllib.loads(Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf-8'))
assert a['id']=='automation' and a['kind']=='heartbeat' and a['status']=='ACTIVE' and a['rrule']=='FREQ=MINUTELY;INTERVAL=10'
assert a['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58'
assert '新OOFμ驱动短实验实际C PID12392' in a['prompt'] and '新校准/折匹配fit-INNER标量采集/新学生/GPU预检/正式100都未实施' in a['prompt']
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),id=a['id'],kind=a['kind'],name=a['name'],status=a['status'],rrule=a['rrule'],target_thread_id=a['target_thread_id'],prompt_sha256=hashlib.sha256(a['prompt'].encode()).hexdigest(),quiet_unchanged_intent_retained=True,new_diagnostic_complete_not_stale_pending=True,next_calibration_not_claimed_executed=True)
out=b/'outputs/OOF效用完成后监管接续更新.json';out.write_text(json.dumps(record,indent=2),encoding='utf-8')
r=json.loads((b/'outputs/OOF教师效用研究报告与接续永久保存.json').read_text(encoding='utf-8'))
d=Path(r['backup']);shutil.copy2(out,d/'monitor_update_verification.json')
(d/'applied_heartbeat_prompt.json').write_text(json.dumps(dict(prompt=a['prompt'],sha256=record['prompt_sha256']),ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(Path(__file__),d/Path(__file__).name)
assert hashlib.sha256(out.read_bytes()).hexdigest()==hashlib.sha256((d/'monitor_update_verification.json').read_bytes()).hexdigest()
print(json.dumps(record))
