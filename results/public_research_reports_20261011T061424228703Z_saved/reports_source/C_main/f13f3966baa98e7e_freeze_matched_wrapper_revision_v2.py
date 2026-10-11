from pathlib import Path
import datetime,hashlib,json,shutil,ast
w=Path(__file__).parent;parent=Path('D:/CodexBackups/selective_flow_20261003_1105');old=parent/'matched_message_pools_actual_20261006T064950Z'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();now=datetime.datetime.now(datetime.timezone.utc);stamp=now.strftime('%Y%m%dT%H%M%SZ')
new=parent/('matched_message_pools_actual_v2_'+stamp);new.mkdir();remote='/data/coding/group_teacher_v1_20261005T1650Z/matched_message_pools_v2_'+stamp
p=json.loads((old/'plan.json').read_text());r=p['new_root']
keep=['diagnose_matched_message_pools_v1.py','audit_matched_message_pools_v1.py','mu_input.npz','roles.npz','cal_parameters.json','run_same_teacher_capture_v1.py','evaluation_plan.json','evaluate_matched_message_pools_v1.py','audit_matched_pool_eval_v1.py']
for f in keep:shutil.copy2(old/f,new/f)
ast.parse((w/'run_matched_message_pools_phase_v2.py').read_text());shutil.copy2(w/'run_matched_message_pools_phase_v2.py',new/'run_matched_message_pools_phase_v2.py')
pins={k:v for k,v in p['pinned_files'].items() if not k.startswith(r+'/')}
for f in keep[:6]+['run_matched_message_pools_phase_v2.py']:pins[remote+'/'+f]=sha(new/f)
p.update(new_root=remote,pinned_files=pins,frozen_utc=now.isoformat(),wrapper_revision='Only startup exec-argv wait; science source, array definitions, selector, numerical parameters identical.',previous_precheck='GPU numerical check passed; original wrapper14488/child14489 exit0 but actual_proc_argv empty race caused independent audit refusal. Retain failed provenance; do not authorize execute.')
(new/'plan.json').write_text(json.dumps(p,indent=2))
ev=json.loads((new/'evaluation_plan.json').read_text());ev['gpu_plan_sha256']=sha(new/'plan.json');ev['utc']=now.isoformat();(new/'evaluation_plan.json').write_text(json.dumps(ev,indent=2))
assert sha(new/'diagnose_matched_message_pools_v1.py')==sha(old/'diagnose_matched_message_pools_v1.py')
failure=dict(status='V1_ORIGINAL_ARGV_GATE_REFUSED_NOT_NUMERICAL_FAILURE',recorded_utc=now.isoformat(),exit_code=1,child_pid=14489,actual_proc_argv=[],formal_executed=False,revision='new wrapperv2 wait, immutable science')
(old/'precheck/independent_audit_refusal.json').write_text(json.dumps(failure,indent=2))
(new/'local_freeze.json').write_text(json.dumps(dict(utc=now.isoformat(),old_root=str(old),science_sha256=sha(new/'diagnose_matched_message_pools_v1.py'),wrapper_sha256=sha(new/'run_matched_message_pools_phase_v2.py'),plan_sha256=sha(new/'plan.json'),C_free=shutil.disk_usage('C:/').free,D_free=shutil.disk_usage('D:/').free),indent=2))
(w.parent/'outputs/匹配消息候选短实验执行接续.json').write_text(json.dumps(dict(stage='WRAPPER_V2_FROZEN_NOT_GPU_EXECUTED',local_root=str(new),remote_root=remote,original_failed_gate_retained=str(old/'precheck/independent_audit_refusal.json')),indent=2))
print(json.dumps(dict(local_root=str(new),remote_root=remote)))
