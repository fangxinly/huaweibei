"""Publish the already sealed progress capture through the original Git helper."""
import json,pathlib,shutil,sys
P=pathlib.Path;BASE=P(__file__).resolve().parent.parent;sys.path.insert(0,str(BASE/'work'))
from group5_release_transport_v1 import write,digest
from raw_TRAIN_save_and_publish_v1 import publish
from group5_publish_exact_local_v1 import publish_tree
from publish_test_selected_group5_preparation_v1 import DC,OUT,sync

def main():
    root=DC.parent/'g5_human_resume_progress_20261010T045836Z';meta=root/'metadata'
    proof=json.loads((root/'D_receipt.json').read_bytes());assert digest(P(proof['archive']))==proof['archive_SHA']
    failure=root/'Release_failure_original.json'
    if not failure.exists():shutil.copyfile(root/'Release_receipt.json',failure)
    publish(P(proof['archive']),proof['archive_SHA'],'group5-progress-human-continue-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
    github=publish_tree(meta,'results/group5_human_continue_progress_20261010','Preserve actual tenth-epoch progress after human continuation; no repeated training')
    write(root/'GitHub_receipt.json',github);receipt=json.loads((meta/'actual_progress_receipt.json').read_bytes())
    state=json.loads((DC/'D_current_research_state.json').read_bytes());e=state['latest_human_TEST_selected_group5']
    e.update(status=receipt['status'],latest_human_resume_progress=dict(root=str(root),receipt=receipt,preservation=proof,github=github),formal_training_started=True,final_outputs_completed=0)
    e['latest_formal_progress_policy_block']['human_continue_after_rejection']=True;e['latest_formal_progress_policy_block']['currently_blocks_same_protocol']=False
    state.update(updated_at_utc=github['actual_UTC'],github_source=github,current_research_execution_blocker='No new remote approval rejection after human continue and changed permission profile. Same original child2302 running; old04:52 rejection retained as history.',next_gate='Observe original formal100 naturally; complete independent local multi-method runners. CPU original-state audit and full Release original restoration after natural0 before outer scoring. Preserve once and source/lease gates.')
    sync(state);write(BASE/'work/group5_human_resume_progress_pointer.json',dict(root=str(root),receipt=receipt,proof=proof,github=github))
    with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:f.write('\n'+github['actual_UTC']+' 本次新拒绝后人类继续且权限profile变化，同SSH23105原协议只读成功，无新远端审批拒绝，不冒底层修复；原child2302继续，UTC04:58:07原history完整10轮/470更新，无wrapper退出/失败，未重启/重复once。进度原件D先保存；默认sandbox Git凭据助手RuntimeError的失败回执留存后，通过同已授权Git助手发布原封ZIP'+proof['archive_SHA']+'，Release/GitHub'+github['commit']+'与D/C同步过。最终0/25、OUTER未评分。\n')
    print(json.dumps(dict(root=str(root),archive_SHA=proof['archive_SHA'],github=github,last_completed_epoch=10),ensure_ascii=False))

if __name__=='__main__':main()
