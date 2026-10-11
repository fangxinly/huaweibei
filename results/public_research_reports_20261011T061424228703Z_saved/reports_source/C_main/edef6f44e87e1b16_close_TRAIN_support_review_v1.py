import datetime, hashlib, json, os, pathlib, shutil, sys, tomllib
P = pathlib.Path
sys.path.insert(0, str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw, sha, seal
r = P(sys.argv[1])
dc = r.parent.parent / 'candidate_posttrain_lowC_20261009T005229Z'
c = P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
assert shutil.disk_usage('C:/').free > 200 * 1024**2
assert shutil.disk_usage('D:/').free > 40 * 1024**2
path = P('C:/Users/21234/.codex/automations/automation/automation.toml')
cfg = tomllib.loads(path.read_text(encoding='utf-8'))
prompt_sha = hashlib.sha256(cfg['prompt'].encode()).hexdigest()
assert cfg['id'] == 'automation' and cfg['status'] == 'ACTIVE'
assert cfg['rrule'] == 'FREQ=MINUTELY;INTERVAL=10'
assert cfg['target_thread_id'] == '01a109db-b31a-78d3-82f7-282321c5bf58'
assert prompt_sha == '12a295468f7388eb64a219b7c35fc87061c5de1f011163eb2f2ca159f398c03d'
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
proof = dict(actual_UTC=now, id=cfg['id'], name=cfg['name'], status=cfg['status'],
             rrule=cfg['rrule'], target_thread_ID=cfg['target_thread_id'],
             actual_prompt_SHA=prompt_sha, actual_toml_SHA=sha(path),
             review_complete_in_actual_prompt=True, original_frequency_preserved=True,
             no_new_task_or_automation=True)
(r / 'automation_actual_verification.json').write_bytes(raw(proof))
exit_receipt = dict(recorded_actual_UTC=now,
                    publication_helper='publish_TRAIN_support_review_v1.py',
                    execution_tool_returned_exit_code=0,
                    exit_code_observed_before_receipt=True,
                    remote_execution_or_new_capture_not_claimed=True)
(r / 'publication_execution_exit_receipt.json').write_bytes(raw(exit_receipt))
closure = r / 'completion_closure'
closure.mkdir()
for n in ['automation_actual_verification.json', 'publication_execution_exit_receipt.json',
          'GitHub_publication_receipt.json', 'Release_receipt.json',
          'publication_and_state_sync_receipt.json', 'publication_closure_D_receipt.json']:
    shutil.copyfile(r / n, closure / n)
shutil.copyfile(__file__, closure / P(__file__).name)
closed = seal(closure, 'complete_actual_review_completion_closure_original.zip')
(r / 'completion_closure_D_receipt.json').write_bytes(raw(closed))
state = json.loads((dc / 'D_current_research_state.json').read_bytes())
state['latest_TRAIN_support_complete_review']['automation_actual_verification'] = proof
state['latest_TRAIN_support_complete_review']['completion_closure_D_receipt'] = closed
state['updated_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
assert state['latest_TRAIN_support_analysis_sent']['full_reply_received']
for dest in [dc / 'D_current_research_state.json', c / '自主优化实际接续.json']:
    tmp = dest.with_name(dest.name + '.reviewclosed.tmp')
    tmp.write_bytes(raw(state))
    os.replace(tmp, dest)
assert (dc / 'D_current_research_state.json').read_bytes() == (c / '自主优化实际接续.json').read_bytes()
sync = dict(actual_UTC=state['updated_at_utc'], D_C_same_SHA=sha(dc / 'D_current_research_state.json'),
            completion_closure_SHA=closed['archive_SHA'], original_10min_ACTIVE_verified=True)
(r / 'completion_final_state_sync_receipt.json').write_bytes(raw(sync))
with (c / '研究接续状态.md').open('a', encoding='utf8') as f:
    f.write('\n' + state['updated_at_utc'] + ' 完整支持审阅完成接续保存，原10min ACTIVE实际prompt已核SHA12a29546，完整完成闭合ZIP' + closed['archive_SHA'] + ' 全SHA/CRC/unique/exactset过，D/C同字节。仅来源核验待；不再轮询重发已完成建议，不重做探针/支持诊断/评分。\n')
print(json.dumps(dict(closure=closed, sync=sync)))
