"""Instrumentation-only new candidate after original budget-gate failure."""
import ast,datetime,hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parents[1];src=root/'work/second_lease_fixed_execution_v1_20261006T1349Z';utc=datetime.datetime.now(datetime.timezone.utc)
dest=root/'work'/('second_lease_budget_trace_v4_'+utc.strftime('%Y%m%dT%H%M%SZ'));shutil.copytree(src,dest)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=(src/'precheck_minimal_fixed_v3.py').read_text(encoding='utf-8')
needle='        if elapsed > 1200 or max(allocated, reserved) > 6*1024**3:\n'
insert="        record={'stage':stage,'actual_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'elapsed_seconds':elapsed,'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,'budget_seconds':1200,'budget_Torch_bytes':6*1024**3}\n        with (output/'actual_budget_trace.log').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\\n')\n"
assert old.count(needle)==1;new=old.replace(needle,insert+needle)
needle='    snapshots={\'construction_and_initial_full_save\':budget(\'construction\')}\n'
insert="    (output/'partial_actual_progress.json').write_text(json.dumps({'scope':'PARTIAL_PRECHECK_NOT_GPU_PASS','pid':os.getpid(),'construction':session.construction_receipt,'initial_full':initial,'optimizer_steps_completed':0,'full_INNER_replay_completed':False},indent=2),encoding='utf-8')\n"
assert new.count(needle)==1;new=new.replace(needle,insert+needle)
needle='        gradients.append(report);snapshots[key]=budget(key)\n'
insert="        gradients.append(report)\n        (output/'partial_actual_progress.json').write_text(json.dumps({'scope':'PARTIAL_PRECHECK_NOT_GPU_PASS','actual_utc':datetime.now(timezone.utc).isoformat(),'pid':os.getpid(),'construction':session.construction_receipt,'initial_full':initial,'optimizer_steps_completed':index+1,'step_times':times,'component_gradients':component_gradients,'normal_and_mixed_gradients':gradients,'initial_donor_mechanism':mechanism,'full_INNER_replay_completed':False,'formal100_complete':False},indent=2),encoding='utf-8')\n        snapshots[key]=budget(key)\n"
assert new.count(needle)==1;new=new.replace(needle,insert)
(dest/'precheck_minimal_fixed_v4.py').write_text(new,encoding='utf-8')
wrapper=(src/'second_lease_precheck_wrapper_v1.py').read_text().replace('from precheck_minimal_fixed_v3 import run_precheck','from precheck_minimal_fixed_v4 import run_precheck')
(dest/'second_lease_precheck_wrapper_v2.py').write_text(wrapper)
capture=(src/'capture_new_fixed_precheck_v20.py').read_text().replace("'source/second_lease_precheck_wrapper_v1.py'","'source/second_lease_precheck_wrapper_v2.py'").replace("'source/precheck_minimal_fixed_v3.py'","'source/precheck_minimal_fixed_v4.py'").replace("'source/capture_new_fixed_precheck_v20.py'","'source/capture_new_fixed_precheck_v21.py'")
(dest/'capture_new_fixed_precheck_v21.py').write_text(capture)
plan=json.loads((src/'second_lease_precheck_plan.json').read_text());plan.update({'status':'INSTRUMENTATION_V4_UNEXECUTED_SAME_SCIENCE_AND_6GIB_GATE','frozen_actual_utc':utc.isoformat(),'runner':'precheck_minimal_fixed_v4.py','parent_plan_sha256':sha(src/'second_lease_precheck_plan.json'),'changes':'Only persists stage elapsed/allocated/reserved peaks before gate, construction and per-step raw gradient/timing progress before gate. No budget relaxation, batch/objective/init/optimizer/role change. Failed v3 retained.','actual_GPU_executed':False})
plan['source_sha256']['precheck_minimal_fixed_v4.py']=sha(dest/'precheck_minimal_fixed_v4.py');(dest/'second_lease_precheck_plan_v4.json').write_text(json.dumps(plan,indent=2)+'\n')
# Original plan remains immutable; new wrapper points to the new v4 plan.
p=dest/'second_lease_precheck_wrapper_v2.py';p.write_text(p.read_text().replace("bundle/'second_lease_precheck_plan.json'","bundle/'second_lease_precheck_plan_v4.json'"))
ep=json.loads((src/'deployment_execution_plan_v1.json').read_text());ep.update({'status':'BUDGET_TRACE_V4_SOURCE_ONLY_NOT_RETRY_OR_GPU_PASS','frozen_actual_utc':utc.isoformat(),'parent_execution_plan_sha256':sha(src/'deployment_execution_plan_v1.json'),'parent_precheck_plan_sha256':sha(dest/'second_lease_precheck_plan_v4.json')})
for n in ('precheck_minimal_fixed_v4.py','second_lease_precheck_wrapper_v2.py','capture_new_fixed_precheck_v21.py'):ep['source_sha256'][n]=sha(dest/n)
(dest/'deployment_execution_plan_v2.json').write_text(json.dumps(ep,indent=2)+'\n')
for n,h in ep['source_sha256'].items():assert sha(dest/n)==h
for p in dest.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
assert new.count('session.optimizer.step()')==old.count('session.optimizer.step()')==1
receipt={'actual_frozen_utc':utc.isoformat(),'directory':str(dest),'plan_sha256':sha(dest/'second_lease_precheck_plan_v4.json'),'execution_plan_sha256':sha(dest/'deployment_execution_plan_v2.json'),'runner_sha256':sha(dest/'precheck_minimal_fixed_v4.py'),'budget_changed':False,'batch_objective_optimizer_init_or_role_changed':False,'actual_GPU_executed':False,'original_v3_retained':True,'next_dependency':'Complete original full initial D and other-node CPU preservation before retry; true fresh gate and source coverage required.'}
(dest/'local_candidate_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(root/'outputs/第二租期预算计量v4本地候选最新.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
