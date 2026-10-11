import datetime,hashlib,json,os,pathlib,shutil,sys,tomllib
P=pathlib.Path;sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw,sha,seal
r=P(sys.argv[1]);dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
path=P('C:/Users/21234/.codex/automations/automation/automation.toml');cfg=tomllib.loads(path.read_text(encoding='utf-8'))
assert cfg['id']=='automation' and cfg['status']=='ACTIVE' and cfg['rrule']=='FREQ=MINUTELY;INTERVAL=10'
assert cfg['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58'
assert cfg['prompt'].startswith('最新Oct10实际UTC01:33主源配方适用范围核验优先')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
sent=dict(thread_ID='01a10fcb-6663-70a2-9a76-60e5634d0c03',input_original_archive_SHA='c3f0c4e5371663314fc682aaa4556b8b4ebcd427c4619fda52d974ee4f3d83ce',included_predecessor_SHA='aa31b202ea877b682461ee588c64a80a2ed7bb624d794f6505e31a14f1aae9ef',send_once_acknowledged=True,send_receipt_recorded_UTC=now,exact_send_timestamp_not_claimed=True,turn_ID='01a12370-f4a9-7c33-b28e-f9f45502162a',cursor='f8668b33-ccbc-454f-96ee-d12086ff737c:1',last_tool_snapshot_observed_status='inProgress',full_reply_received=False,no_same_batch_resend=True,scope='ANALYSIS_ONLY_NO_GPU_CREDENTIALS_OR_MAIN_SOURCE_CHANGES')
(r/'advisor_actual_send_receipt.json').write_bytes(raw(sent))
proof=dict(actual_UTC=now,status=cfg['status'],rrule=cfg['rrule'],actual_prompt_SHA=hashlib.sha256(cfg['prompt'].encode()).hexdigest(),actual_toml_SHA=sha(path),recipe_scope_in_actual_prompt=True)
(r/'automation_actual_verification.json').write_bytes(raw(proof))
(r/'execution_exit_observation.json').write_bytes(raw(dict(recorded_actual_UTC=now,source_fetch_tool_exit=0,PR_fetch_tool_exit=0,decision_AST_tool_exit=0,publisher_tool_exit=0,observed_before_receipt=True,no_new_remote_execution_claim=True)))
closure=r/'completion_closure';closure.mkdir()
for n in ['advisor_actual_send_receipt.json','automation_actual_verification.json','execution_exit_observation.json','GitHub_publication_receipt.json','publication_closure_D_receipt.json','publication_and_state_sync_receipt.json']:
    shutil.copyfile(r/n,closure/n)
shutil.copyfile(__file__,closure/P(__file__).name);closed=seal(closure,'complete_actual_recipe_scope_completion_original.zip');(r/'completion_D_preservation_receipt.json').write_bytes(raw(closed))
state=json.loads((dc/'D_current_research_state.json').read_bytes());state['latest_raw_TRAIN_recipe_analysis_sent']=sent;state['latest_raw_TRAIN_recipe_scope']['automation_actual_verification']=proof;state['latest_raw_TRAIN_recipe_scope']['completion_D_receipt']=closed;state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for p in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
    tmp=p.with_name(p.name+'.recipeclosed.tmp');tmp.write_bytes(raw(state));os.replace(tmp,p)
assert (dc/'D_current_research_state.json').read_bytes()==(c/'自主优化实际接续.json').read_bytes()
adv=json.loads((c/'研究建议交流接续.json').read_bytes());adv['latest_raw_TRAIN_recipe_analysis_sent']=sent;(c/'研究建议交流接续.json').write_bytes(raw(adv))
(r/'completion_final_state_sync_receipt.json').write_bytes(raw(dict(actual_UTC=state['updated_at_utc'],D_C_same_SHA=sha(dc/'D_current_research_state.json'))))
with (c/'研究接续状态.md').open('a',encoding='utf-8') as f:f.write('\n'+state['updated_at_utc']+' 新recipe来源批SHAc3f0c4e5连前驱aa31b202已向原建议聊天只发送一次仅分析，首次snapshot新turn01a12370 inProgress/full_reply_received假，不等回复停保存，不机械扩邻近仓库。原10min ACTIVE实际prompt已核，新完成闭合ZIP'+closed['archive_SHA']+'全SHA/CRC/unique/exactset过，D/C同字节；本次官方pickle生成身份仍未知，无新数值执行。\n')
print(json.dumps(dict(completion=closed,D_C_same_SHA=sha(dc/'D_current_research_state.json'),automation=proof)))
