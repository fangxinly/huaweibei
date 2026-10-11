"""Verify the existing heartbeat was refreshed without changing its schedule."""
import datetime as dt
import hashlib
import json
import pathlib
import shutil
import sys
import tomllib

P=pathlib.Path
BASE=P(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import digest,write,seal
from raw_TRAIN_save_and_publish_v1 import publish
from group5_publish_exact_local_v1 import publish_tree
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync

pointer=BASE/'work/group5_current_continuation_pointer.json'
prepared=json.loads(pointer.read_bytes())
root=P(prepared['root'])/'heartbeat_continuation_closure'
root.mkdir()
config=P('C:/Users/21234/.codex/automations/automation/automation.toml')
actual=tomllib.loads(config.read_text(encoding='utf-8'))
before=json.loads((BASE/'work/group5_automation_before_20261010T1155Z.json').read_bytes())
expected=(BASE/'work/group5_heartbeat_updated_prompt.txt').read_text(encoding='utf8').rstrip('\n')
assert actual['prompt']==expected
assert actual['id']=='automation' and actual['status']==before['status']=='ACTIVE'
assert actual['rrule']==before['rrule']=='FREQ=MINUTELY;INTERVAL=10'
assert actual['target_thread_id']==before['target_thread_id']=='01a109db-b31a-78d3-82f7-282321c5bf58'
fixed={k:v for k,v in before.items() if k not in ('prompt','updated_at')}
assert fixed=={k:v for k,v in actual.items() if k not in ('prompt','updated_at')}
verification=dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
 status='EXISTING_HEARTBEAT_CURRENT_PROMPT_VERIFIED_SCHEDULE_UNCHANGED',automation_id=actual['id'],
 name=actual['name'],active=True,interval_minutes=10,target_thread_id=actual['target_thread_id'],
 prompt_SHA=hashlib.sha256(actual['prompt'].encode()).hexdigest(),actual_TOML_SHA=digest(config),
 old_prompt_SHA=hashlib.sha256(before['prompt'].encode()).hexdigest(),
 original_schedule_target_name_and_other_fields_unchanged=True,no_new_automation_or_chat=True,
 latest_preparation_original_SHA=prepared['proof']['archive_SHA'],
 current_formal_outputs_preserved_unscored=1,current_outer_scores=0,new_train_dispatch=False,
 human_new_lease_question_pending=True,quiet_on_unchanged_or_preparation=True)
write(root/'actual_automation_verification.json',verification)
shutil.copyfile(BASE/'work/group5_heartbeat_updated_prompt.txt',root/'actual_prompt.txt')
shutil.copyfile(P(__file__),root/P(__file__).name)
write(root/'continuation_original_reference.json',dict(root=prepared['root'],proof=prepared['proof'],github=prepared['github']))
proof=seal(root,root/'complete_actual_heartbeat_continuation_closure.zip')
write(root.parent/'heartbeat_D_receipt.json',proof)
publish(P(proof['archive']),proof['archive_SHA'],'group5-heartbeat-current-closure-'+proof['archive_SHA'][:12]+'.zip',root.parent/'heartbeat_Release_receipt.json')
github=publish_tree(root,'results/group5_current_heartbeat_20261010T1214','Refresh and verify existing quiet ten-minute research heartbeat; preserve actual lease gate and original closed outputs')
write(root.parent/'heartbeat_GitHub_receipt.json',github)
state=json.loads((DC/'D_current_research_state.json').read_bytes())
state['latest_human_TEST_selected_group5']['latest_no_new_lease_continuation']['automation_actual_verification']=verification
state['latest_human_TEST_selected_group5']['latest_no_new_lease_continuation']['heartbeat_closure']=dict(proof=proof,github=github)
state.update(updated_at_utc=github['actual_UTC'],github_source=github)
sync(state)
prepared.update(automation_actual_verification=verification,heartbeat_closure=dict(proof=proof,github=github));write(pointer,prepared)
assert (DC/'D_current_research_state.json').read_bytes()==(OUT/'自主优化实际接续.json').read_bytes()
with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:
 f.write('\n'+github['actual_UTC']+' 原10min heartbeat仅更新最新真实结果/租期门接续，ACTIVE/原频率/目标/其余字段逐项未变；无新自动化或聊天。实际TOML prompt SHA'+verification['prompt_SHA']+'已核，原件与回执D/Release/GitHub'+github['commit']+'闭合、D/C同字节。\n')
print(json.dumps(dict(root=str(root),verification=verification,proof=proof,github=github)),flush=True)
