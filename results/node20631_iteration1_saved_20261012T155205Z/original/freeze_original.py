import datetime as dt,json,pathlib,shutil,sys,zipfile
P=pathlib.Path;B=P(__file__).resolve().parent.parent;W=B/'work';sys.path.insert(0,str(W))
from group5_release_transport_v1 import digest,write,verify_zip
from freeze_node23668_careflow1_v4 import publish_complete_small
from publish_test_selected_group5_preparation_v1 import DC,sync
ptr=json.loads((W/'node20631_iteration_pointer.json').read_bytes());prep=P(ptr['root']);old=json.loads((W/'node20631_official_queue_pointer.json').read_bytes());assert not old['training_dispatched'] and not old['once_consumed']
z=prep/'synthetic_closed_original.zip';verify_zip(z)
now=dt.datetime.now(dt.timezone.utc);r=DC.parent/('node20631_iteration1_saved_'+now.strftime('%Y%m%dT%H%M%SZ'));r.mkdir();pay=r/'payload';pay.mkdir()
with zipfile.ZipFile(z) as f:f.extractall(pay/'synthetic_original')
q=pay/'synthetic_original';assert json.loads((q/'synthetic_natural_exit.json').read_bytes())['natural_exit']==0
assert digest(q/'iterative_method_v1.py')==digest(prep/'source/iterative_method_v1.py')
shutil.copyfile(z,pay/'synthetic_closed_original.zip');shutil.copytree(prep/'source',pay/'source')
for n in ['prepared_plan.json','prior_not_executed_pointer.json','实验协议.md']:shutil.copyfile(prep/n,pay/n)
shutil.copyfile(W/'freeze_node20631_iteration_v1.py',pay/'freeze_original.py');shutil.copyfile(W/'prepare_node20631_iteration_v1.py',pay/'prepare_original.py')
shutil.copyfile(P(old['payload'])/'runtime_and_route_closed_original.zip',pay/'runtime_and_route_closed_original.zip')
shutil.copyfile('C:/Users/21234/.codex/automations/automation/automation.toml',pay/'automation_current_original.toml')
p=json.loads((prep/'prepared_plan.json').read_bytes());p.update(execution_enabled=True,frozen_UTC=now.isoformat(),synthetic_original_SHA=digest(z),local_space_capture_UTC=now.isoformat(),local_C_free_bytes=shutil.disk_usage('C:/').free,local_D_free_bytes=shutil.disk_usage('D:/').free)
assert (dt.datetime.fromisoformat(p['lease_end_UTC'])-now).total_seconds()>p['per_job_budget_seconds']+7200
assert p['local_C_free_bytes']>200*1024**2 and p['local_D_free_bytes']>40*1024**2
write(pay/'plan.json',p);h=digest(pay/'plan.json');proof=publish_complete_small(r,pay,'node20631-VAL-only-iteration1-frozen-plan')
ptr.update(frozen_root=str(r),payload=str(pay),plan_SHA=h,python=p['python'],assets_root=p['assets_root'],runtime_passed=True,synthetic_natural_exit=0,synthetic_original_SHA=digest(z),preservation=proof,lease_end_UTC=p['lease_end_UTC'],training_dispatched=False,once_consumed=False)
write(W/'node20631_iteration_pointer.json',ptr)
state=json.loads((DC/'D_current_research_state.json').read_bytes());state['latest_human_research_direction']=dict(actual_UTC=now.isoformat(),scope='OFFICIAL_FIXED_SPLIT_VAL_ONLY_ITERATION',human_confirmation='不用之前内容的五折验证，只管方法不断更新迭代',old_pooled_CV_and_previous15queue_not_to_dispatch=True,methods=p['methods'],seeds=p['seeds'],TEST_scoring_enabled=False,TRAIN_VAL_only_iteration=True,all_old_results_once_preserved=True)
state['latest_node20631_iteration']=ptr;state['latest_node20631_official_queue']['superseded_before_dispatch']=True;sync(state);write(DC/'node20631_iteration_pointer.json',ptr)
print(json.dumps(dict(root=str(r),plan_SHA=h,remote=ptr['remote'],GitHub=proof['GitHub']['commit'],training_dispatched=False)))
