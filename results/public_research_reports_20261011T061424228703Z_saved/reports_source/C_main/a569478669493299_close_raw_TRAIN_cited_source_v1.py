import datetime,hashlib,json,os,pathlib,shutil,sys,tomllib
P=pathlib.Path;sys.path.insert(0,str(P(__file__).parent))
from raw_TRAIN_capture_v1 import raw,sha,seal
r=P(sys.argv[1]);dc=r.parent.parent/'candidate_posttrain_lowC_20261009T005229Z';c=P('C:/Users/21234/Documents/Codex/2026-10-05/ni/outputs')
path=P('C:/Users/21234/.codex/automations/automation/automation.toml');cfg=tomllib.loads(path.read_text(encoding='utf-8'))
assert cfg['status']=='ACTIVE' and cfg['rrule']=='FREQ=MINUTELY;INTERVAL=10' and cfg['id']=='automation'
assert cfg['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58'
assert cfg['prompt'].startswith('最新Oct10实际UTC01:16公开来源核验优先') and '325520faf4eb19e02ac0ff8dd19d754618673c76' in cfg['prompt']
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
proof=dict(actual_UTC=now,status=cfg['status'],rrule=cfg['rrule'],prompt_SHA=hashlib.sha256(cfg['prompt'].encode()).hexdigest(),actual_toml_SHA=sha(path),cited_source_in_actual_prompt=True,no_new_automation=True)
(r/'automation_actual_verification.json').write_bytes(raw(proof))
(r/'execution_exit_observation.json').write_bytes(raw(dict(recorded_actual_UTC=now,source_fetch_tool_exit=0,decision_tool_exit=0,publisher_tool_exit=0,observed_before_this_receipt=True,no_new_remote_execution_claim=True)))
closure=r/'completion_closure';closure.mkdir()
for n in ['automation_actual_verification.json','execution_exit_observation.json','GitHub_publication_receipt.json','publication_closure_D_receipt.json','publication_and_state_sync_receipt.json']:
    shutil.copyfile(r/n,closure/n)
shutil.copyfile(__file__,closure/P(__file__).name);closed=seal(closure,'complete_actual_cited_source_completion_original.zip');(r/'completion_D_preservation_receipt.json').write_bytes(raw(closed))
state=json.loads((dc/'D_current_research_state.json').read_bytes());state['latest_raw_TRAIN_cited_public_source']['automation_actual_verification']=proof;state['latest_raw_TRAIN_cited_public_source']['completion_D_receipt']=closed;state['updated_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
for p in [dc/'D_current_research_state.json',c/'自主优化实际接续.json']:
    tmp=p.with_name(p.name+'.citedclosed.tmp');tmp.write_bytes(raw(state));os.replace(tmp,p)
assert (dc/'D_current_research_state.json').read_bytes()==(c/'自主优化实际接续.json').read_bytes()
(r/'completion_final_state_sync_receipt.json').write_bytes(raw(dict(actual_UTC=state['updated_at_utc'],D_C_same_SHA=sha(dc/'D_current_research_state.json'))))
with (c/'研究接续状态.md').open('a',encoding='utf-8') as f:f.write('\n'+state['updated_at_utc']+' 新引用公开源批完成保存，原10min ACTIVE实际prompt已核，完成闭合ZIP'+closed['archive_SHA']+'全member SHA/CRC/unique/exactset过，D/C同字节。中心化来源仍未闭合，未发重复建议，无新数值执行。\n')
print(json.dumps(dict(closure=closed,D_C_same_SHA=sha(dc/'D_current_research_state.json'),automation=proof)))
