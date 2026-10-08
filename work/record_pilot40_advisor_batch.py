import json,hashlib,zipfile,shutil
from pathlib import Path
root=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni');out=root/'outputs'
base=Path('D:/CodexBackups/selective_flow_20261003_1105/pilot40_actual_advisor_batch_20261007T105144Z');base.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
path=out/'研究建议交流接续.json';shutil.copy2(path,base/'prior_exchange_original.json')
j=json.loads(path.read_text(encoding='utf-8'))
assert 'pilot40_actual_batch' not in j
j['pilot40_actual_batch']={'batch_id':'anchored_flow_pilot40_complete_actual_20261007T103742Z',
 'actualclock_record_UTC':'2026-10-07 10:51:44 UTC','joint_sha256':'45a68a9aaed2136ed5849d43a46bce1cbcfe876b646fa96aec6e8722b05bae0c',
 'sent_once_after_D_original_other_CPU_joint':True,'prompt_sha256':sha(root/'work/pilot40_actual_advisor_batch.txt'),
 'thread_id':'01a10fcb-6663-70a2-9a76-60e5634d0c03','turn_id':'01a115fd-7eb1-7001-8055-99b1de6ccd1f',
 'status':'ACTUALLY_IN_PROGRESS_NO_COMPLETE_REPORT','wait_once':True,'cursor':'a752fe6f-3a58-446f-9d08-6606385560e1:1',
 'complete_report_read_or_adopted':False,'sixteenth_failed_usage_limit_history_preserved':True,
 'same_batch_retry_forbidden':True,'scope':'Analysis-only new pilot40 complete and historical0.59 alignment; no GPU/credentials/new chats/subagents',
 'independent_main_decision':'Current pilot not proved improvement. Retain old positive DEV signal; controlled two-stage training proposal pending exact source/storage/whole queue budget, no new training launch.',
 'later_once_read_when_new_reply_available':True,'preservation_not_delayed':True}
path.write_text(json.dumps(j,ensure_ascii=False,indent=2),encoding='utf-8')
payload=base/'payload';payload.mkdir()
for p in [path,out/'新方案40轮实际完成与完整保存接续.json',out/'TEST与既有0.59结果对齐分析.json',out/'保留旧0.598信号的下一项受控训练方向.md',root/'work/pilot40_actual_advisor_batch.txt',Path(__file__)]:shutil.copy2(p,payload/p.name)
manifest={p.name:sha(p) for p in payload.iterdir()}
(payload/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
archive=base/'actual_analysis_batch_local_seal.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for p in sorted(payload.iterdir()):z.write(p,p.name)
with zipfile.ZipFile(archive) as z:
 assert len(z.namelist())==len(set(z.namelist())) and z.testzip() is None
 for n,h in manifest.items():assert hashlib.sha256(z.read(n)).hexdigest()==h
print(json.dumps({'status':'NEW_ACTUAL_BATCH_SENT_ONCE_REPLY_PENDING_D_LOCAL_SEAL_COMPLETE','zip_sha256':sha(archive)},ensure_ascii=False))
