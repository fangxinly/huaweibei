"""Preserve a new denied read-only query without any further remote operation."""
import datetime as dt,json,pathlib,shutil,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import write,seal
from raw_TRAIN_save_and_publish_v1 import publish
from group5_publish_exact_local_v1 import publish_tree
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync

def main():
    now=dt.datetime.now(dt.timezone.utc).isoformat();root=DC.parent/('g5_formal_progress_policy_block_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir();meta=root/'metadata';meta.mkdir()
    failure=dict(status='NEW_READ_ONLY_FORMAL_PROGRESS_QUERY_AUTO_APPROVAL_REJECTED',recorded_actual_UTC=now,
        action='write_stdin SSH23105 read original wrapper exit/failure/history and last FIT step',SSH_session=23105,
        exact_error='write_stdin rejected: approval required by policy, but AskForApproval is set to Never',
        remote_execution_confirmed=False,authentication_failure=False,training_failure_observed=False,
        remote_training_stopped=False,healthy_task_interrupted=False,remote_root='/data/coding/g5careflow100_20261010T043810Z',
        last_confirmed_original_dispatch=dict(child=2302,actual_UTC='2026-10-10T04:47:40.128116+00:00'),
        last_prior_terminal_observation=dict(actual_UTC='2026-10-10T04:50:37.332036+00:00',completed_history_epoch=2,FIT_updates=141,FIT_epoch=3,FIT_batch=47,complete_epoch_state_bytes=2965811069,
            observation_origin='Existing previous-turn terminal output, not a new remote capture; no full checkpoint SHA asserted'),
        new_remote_capture=False,current_training_state_after_denial_unknown=True,
        next_action='No remote query/retry, splitting, tool/command/connection change until a human instruction after this rejection or permission-state change. Keep healthy training untouched; independent local work allowed.',
        final_outputs_completed=0,formal_once_consumed=True,do_not_repeat_training_or_native_precheck=True)
    write(meta/'exact_policy_failure_and_prior_observation.json',failure)
    shutil.copyfile(BASE/'work/save_group5_progress_policy_block_v1.py',meta/'save_group5_progress_policy_block_v1.py')
    original=json.loads((BASE/'work/group5_formal_dispatch_pointer.json').read_bytes());write(meta/'prior_original_dispatch_preservation_pointer.json',original)
    proof=seal(meta,meta/'complete_progress_policy_rejection_metadata.zip');write(root/'D_preservation_receipt.json',proof)
    publish(P(proof['archive']),proof['archive_SHA'],'group5-formal-progress-policy-block-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
    github=publish_tree(meta,'results/group5_formal_progress_policy_block_20261010','Preserve new rejected read-only progress query; leave live training untouched')
    write(root/'GitHub_receipt.json',github)
    state=json.loads((DC/'D_current_research_state.json').read_bytes());e=state['latest_human_TEST_selected_group5']
    e.update(status=failure['status'],latest_formal_progress_policy_block=dict(root=str(root),failure=failure,preservation=proof,github=github),formal_training_started=True,final_outputs_completed=0)
    state.update(updated_at_utc=now,next_gate=failure['next_action'],github_source=github,current_research_execution_blocker=failure['exact_error'])
    sync(state);write(BASE/'work/group5_formal_progress_block_pointer.json',dict(root=str(root),failure=failure,proof=proof,github=github))
    with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+now+' 新write_stdin23105只读正式训练进度查询被自动审批拒绝：'+failure['exact_error']+'。本次远端执行未确认、非认证/训练失败，健康child2302未停止；此前UTC04:50:37已2完整轮/141FIT更新只是既有观察。拒绝后不重试/拆分/换工具连接，等本次拒绝后人类或权限变化；新元数据原ZIP'+proof['archive_SHA']+'和GitHub'+github['commit']+'完整保存、D/C同步，正式最终仍0/25，原once不重跑。\n')
    print(json.dumps(dict(root=str(root),archive_SHA=proof['archive_SHA'],github=github),ensure_ascii=False))

if __name__=='__main__':main()
