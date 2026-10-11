"""Preserve observed native precheck and terminal-review rejection without SSH."""
import datetime as dt,json,pathlib,shutil,sys
P=pathlib.Path
BASE=P(__file__).resolve().parent.parent
sys.path.insert(0,str(BASE/'work'))
from raw_TRAIN_capture_v1 import raw,sha,seal
from publish_test_selected_group5_preparation_v1 import sync,DC,OUT

now=dt.datetime.now(dt.timezone.utc)
root=DC.parent/('g5_native_review_block_'+now.strftime('%Y%m%dT%H%M%SZ'))
root.mkdir()
assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2
pointer=json.loads((BASE/'work/group5_native_pointer.json').read_bytes())
error='write_stdin rejected: terminal input and permission details are too large to review safely; use a smaller input or start a new terminal with fewer grants'
(root/'tool_review_rejection.txt').write_text(error+'\n',encoding='utf8')
record=dict(recorded_actual_UTC=now.isoformat(),failure_timestamp_not_reconstructed=True,
    tool='write_stdin',session_id=23105,action='Upload separate CPU audit source and invoke CPU-only original checkpoint audit',
    exact_error=error,remote_execution_confirmed=False,CPU_audit_executed=False,
    authentication_failure=False,smaller_input_or_other_connection_not_attempted=True,
    later_human_or_permission_change_required_before_remote_retry=True,
    record_origin='Exact rejection text retained across context compaction; no full original failed tool object available')
(root/'rejection_scope.json').write_bytes(raw(record))
observation=dict(recorded_actual_UTC=now.isoformat(),origin='Previously observed terminal result summarized across context compaction; not a newly retrieved remote receipt',
    remote_root='/data/coding/g5native_20261010T032338Z',SSH_session=23105,
    method='careflow',fold=0,native_precheck_updates=3,native_precheck_observed_natural_exit=0,
    status='NATIVE_PRECHECK3_EXIT_PENDING_CPU_TRANSPORT',
    plan_SHA='598dcd8ef0d80f1e10eb8f59b26ae296669af1e1a0f7ca266f6b522541026119',
    checkpoint_remote='/data/coding/g5native_20261010T032338Z/out/precheck_full.pt',
    checkpoint_observed_SHA='8883bc80222c968c61d6a3cd40d5f8eb4a17fe8ddc8149557da431cce627bb71',
    checkpoint_observed_bytes=2223516942,peak_GPU_bytes=4232052736,
    original_precheck_once_consumed=True,dummy_labels_prediction_state_rng_invariance_observed=True,
    full_checkpoint_reload_prediction_equality_observed=True,
    conservative_FIT_update_only_seconds=8487.095864117146,
    estimate_excludes_INNER_and_saving=True,formal_training_started=False,final_outputs_completed=0,
    native_independent_CPU_qualified=False,real_large_checkpoint_Release_preserved=False,
    remote_original_retained=True,precheck_must_not_be_repeated=True)
(root/'native_precheck_terminal_observation.json').write_bytes(raw(observation))
for rel in ['group5_direct_CPU_audit_v1.py','group5_composite_components_v1.py','group5_teacher_optimizer_origin.json','group5_pipeline_pointer.json','group5_native_pointer.json']:
    shutil.copyfile(BASE/'work'/rel,root/rel)
shutil.copyfile(P(pointer['plan']),root/'frozen_precheck_plan.json')
shutil.copyfile(P(__file__),root/P(__file__).name)
proof=seal(root,'complete_actual_native_precheck_block_metadata.zip')
(root/'D_metadata_preservation_receipt.json').write_bytes(raw(proof))
state=json.loads((DC/'D_current_research_state.json').read_bytes())
entry=state['latest_human_TEST_selected_group5']
entry.update(status='CAREFLOW_PRECHECK3_OBSERVED_EXIT0_CPU_INPUT_REVIEW_BLOCKED',
    training_started=True,training_started_scope='Only three-update native qualification; no formal100-epoch stage',
    formal_training_started=False,final_outputs_completed=0,execution_enabled=False,
    latest_native_precheck=observation,latest_remote_review_block=record,latest_metadata_D_receipt=proof)
state['updated_at_utc']=now.isoformat()
state['next_gate']='Preserve current native original remotely. New CPU-audit terminal input review rejected, no remote audit execution confirmed. No SSH retry, command splitting, connection/tool switch until subsequent human instruction or permission change. Independent local implementation/publication may continue. Native precheck once consumed; never retrain it. CPU-original state and actual large Release restoration remain pending before any formal100 stage.'
sync(state)
with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:
    f.write('\n'+now.isoformat()+' CaReFlow fold0三步原生预检此前已观察自然0，完整检查点2223516942B/SHA8883bc80远端保留，尚无正式100轮或25外折结果。新write_stdin23105上传CPU审核源码被输入/权限信息过大无法安全审阅拒绝，未确认该调用远端执行，非SSH认证失败。拒绝后不拆命令/换工具连接重试，等后续人类或权限变化；CPU/真实大状态Release还原待。此处为既有终端观察和原拒绝文本落盘，不冒新远端receipt读取。D元数据ZIP'+proof['archive_SHA']+'全member SHA/CRC/unique/exactset过，D/C同字节。\n')
(BASE/'work/group5_native_block_pointer.json').write_bytes(raw(dict(root=str(root),proof=proof)))
print(json.dumps(dict(root=str(root),proof=proof),ensure_ascii=False))
