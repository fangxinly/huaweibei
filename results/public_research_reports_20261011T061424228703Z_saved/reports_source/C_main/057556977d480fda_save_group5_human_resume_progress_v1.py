"""Preserve original post-authorization progress; no training dispatch or score."""
import datetime as dt,hashlib,io,json,pathlib,shutil,sys,zipfile
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import write,seal
from raw_TRAIN_save_and_publish_v1 import publish
from group5_publish_exact_local_v1 import publish_tree
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync

def main():
    root=DC.parent/('g5_human_resume_progress_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));root.mkdir();meta=root/'metadata';meta.mkdir()
    b=bytes.fromhex((BASE/'work/group5_human_resume_progress_original.hex').read_text());h=hashlib.sha256(b).hexdigest()
    assert h=='cdf23fd7651715f338a9de24c7466262547135a065cb4eb7a21c6eabd0bf3a39';(meta/'remote_progress_original.zip').write_bytes(b)
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
        assert 'wrapper_exit.json' not in z.namelist() and 'out/failed_natural_exit.json' not in z.namelist()
        for n in z.namelist():
            p=meta/'original'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(z.read(n))
    snapshot=json.loads((meta/'original/capture_actual.json').read_bytes());history=json.loads((meta/'original/out/history.json').read_bytes());last=history[-1]
    assert last['epoch']==10 and last['updates']==470
    receipt=dict(status='HUMAN_CONTINUE_SAME_PROTOCOL_REAL_PROGRESS10',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        original_remote_capture=snapshot,original_SHA=h,last_completed_epoch=last['epoch'],logical_updates_at_completed_epoch=last['updates'],
        remote_root='/data/coding/g5careflow100_20261010T043810Z',SSH_session=23105,child=2302,
        later_human_continue_after_0452_rejection=True,permission_profile_changed=True,new_approval_rejection=False,
        no_claim_underlying_approval_policy_fixed=True,training_restarted=False,new_train_once_consumed=False,
        original_wrapper_exit_absent=True,formal_final_outputs_completed=0,outer_scores_computed=False,
        permission_grant_scope='This turn only: D preservation write and network approved by request_permissions')
    write(meta/'actual_progress_receipt.json',receipt);shutil.copyfile(BASE/'work/save_group5_human_resume_progress_v1.py',meta/'save_group5_human_resume_progress_v1.py')
    proof=seal(meta,meta/'complete_actual_progress_original.zip');write(root/'D_receipt.json',proof)
    publish(P(proof['archive']),proof['archive_SHA'],'group5-progress-human-continue-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
    github=publish_tree(meta,'results/group5_human_continue_progress_20261010','Preserve actual tenth-epoch progress after human continuation; no repeated training')
    write(root/'GitHub_receipt.json',github)
    state=json.loads((DC/'D_current_research_state.json').read_bytes());e=state['latest_human_TEST_selected_group5']
    e.update(status=receipt['status'],latest_human_resume_progress=dict(root=str(root),receipt=receipt,preservation=proof,github=github),formal_training_started=True,final_outputs_completed=0)
    e['latest_formal_progress_policy_block']['human_continue_after_rejection']=True;e['latest_formal_progress_policy_block']['currently_blocks_same_protocol']=False
    state.update(updated_at_utc=receipt['actual_UTC'],github_source=github,current_research_execution_blocker='No new approval rejection after human continue and changed permission profile. Same original child2302 running; old04:52 rejection retained as history.',next_gate='Observe original formal100 naturally; complete independent local multi-method runners. CPU original-state audit and full Release original restoration after natural0 before outer scoring. Preserve once and source/lease gates.')
    sync(state);write(BASE/'work/group5_human_resume_progress_pointer.json',dict(root=str(root),receipt=receipt,proof=proof,github=github))
    with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+receipt['actual_UTC']+' 本次新拒绝后人类继续且权限profile变化，同SSH23105原协议只读成功，无新审批拒绝，不冒底层修复；原child2302继续运行，UTC04:58:07原history完整10轮/470更新，无wrapper退出/失败记录，未重启/重复once。完整原进度ZIP'+proof['archive_SHA']+'全SHA/CRC/member过、Release/GitHub'+github['commit']+'保存D/C同步；最终0/25、OUTER未评分。\n')
    print(json.dumps(dict(root=str(root),archive_SHA=proof['archive_SHA'],github=github,last_completed_epoch=10),ensure_ascii=False))

if __name__=='__main__':main()
