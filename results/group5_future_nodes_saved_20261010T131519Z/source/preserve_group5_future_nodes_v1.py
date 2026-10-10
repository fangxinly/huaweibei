"""Preserve explicit future-node timing and bounded local source preparation."""
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tomllib

BASE = Path(__file__).resolve().parent.parent
WORK = BASE/'work'
sys.path.insert(0,str(WORK))
from group5_release_transport_v1 import digest,write,seal

POINTER = WORK/'group5_future_nodes_pointer.json'


def prepare():
    assert shutil.disk_usage('C:/').free>200*1024**2
    check=json.loads((WORK/'group5_composite_recovery_v2_check_pointer.json').read_bytes())
    report=Path(check['report']);assert digest(report)==check['SHA']
    result=json.loads(report.read_bytes())
    assert result['status']=='COMPOSITE_V2_METADATA_SYNTHETIC_AND_AST_CHECKS_PASS'
    assert result['passed_cases']==28 and not result['scientific_imports'] and not result['real_recovery_qualified']
    stamp=dt.datetime.now(dt.timezone.utc)
    root=BASE/'outputs'/('group5_future_nodes_preparation_'+stamp.strftime('%Y%m%dT%H%M%SZ'))
    payload=root/'payload';payload.mkdir(parents=True)
    names=('group5_composite_runtime_v2.py','group5_composite_epoch_resume_v2.py',
           'group5_composite_CPU_audit_v2.py','group5_composite_resume_contract_v2.py',
           'prepare_group5_composite_recovery_v2.py','check_group5_composite_recovery_contract_v2.py',
           'group5_composite_runtime_v1.py','group5_composite_epoch_resume_v1.py',
           'group5_composite_CPU_audit_v1.py',Path(__file__).name)
    inventory=[]
    for name in names:
        source=WORK/name;ast.parse(source.read_text(encoding='utf-8-sig'),filename=name)
        destination=payload/'source'/name;destination.parent.mkdir(exist_ok=True);shutil.copyfile(source,destination)
        assert digest(source)==digest(destination)
        inventory.append(dict(name=name,SHA=digest(destination),bytes=destination.stat().st_size))
    write(payload/'source_inventory.json',inventory)
    shutil.copyfile(report,payload/'actual_synthetic_report.json')
    write(payload/'synthetic_tool_exit_observation.json',dict(observed_record_UTC=stamp.isoformat(),
          source='Actual exec_command tool return for the already executed check; no rerun or invented exact process-exit timestamp.',
          tool_chunk_id='caef4d',exit_code=0,wall_time_seconds=6.7373425000000005,
          report_SHA=check['SHA'],new_execution_by_this_record=False))
    human=dict(observed_record_UTC=stamp.isoformat(),human_answer='24小时后提供新节点',
          preceding_clarification='Current nodes extended, or provide new nodes after24h?',
          interpretation='Human will provide new nodes after24h; current N1/N2/N3 leases are not extended.',
          exact_future_node_arrival_not_confirmed=True,new_endpoints_ports_UUID_or_actual_lease_not_yet_supplied=True,
          current_leases_extended=False,renewal_or_purchase_authorized=False,
          future_timing_is_not_a_valid_new_lease_or_execution_proof=True,repeat_lease_question_now=False)
    write(payload/'human_future_node_availability.json',human)
    continuation=json.loads((WORK/'group5_current_continuation_pointer.json').read_bytes())
    write(payload/'previous_closed_continuation_reference.json',dict(root=continuation['root'],proof=continuation['proof'],
          github=continuation.get('heartbeat_closure',{}).get('github',continuation['github'])))
    scope=dict(actual_UTC=stamp.isoformat(),status='LOCAL_COMPOSITE_V2_SOURCE_PREPARED_FUTURE_NODES_NOT_CURRENT_LEASE_EXTENSION',
          preserved_unscored_method_fold_outputs=1,outer_score_outputs=0,required_outputs=25,required_training_stages=35,
          source_AST_passed_files=len(inventory),metadata_synthetic_passed_cases=28,unchanged_science_functions=result['unchanged_science_functions'],
          composite_recovery_branch_integrated_as_prepared_source=True,original_partial_CPU_audit_implemented=False,
          real_composite_model_Adam_RNG_CUDA_recovery_qualified=False,real_composite_recovery_execution=False,
          old_v1_sources_retained=True,scientific_imports_or_task_arrays_decoded=False,new_model_fit_inference_score=False,
          new_training_scoring_or_native_once_consumed=False,current_node_leases_unchanged=True,new_valid_lease_proof_received=False,
          actual_source_checks_scope='Standard-library synthetic metadata and AST only; positive fixtures are not original audit evidence.')
    write(payload/'preparation_scope.json',scope)
    (payload/'continuation.md').write_text(
        '# Five-fold continuation while awaiting new nodes\n\n'
        'The human clarified: “24小时后提供新节点”. This is future node availability; '
        'it does not extend the current three nodes or supply a new endpoint, UUID or actual lease. '
        'The current measured complete-stage budget plus saving reserve still forbids another formal dispatch.\n\n'
        'CaReFlow fold0 remains complete and preserved with 437 unscored outer predictions (1/25 outputs, 0/25 scores). '
        'Fold1 native3/CPU/full Release restoration remains complete; its formal training has not started. '
        'No completed original or consumed token was rerun.\n\n'
        'The separate composite v2 runtime now connects complete-epoch recovery to SHA-bound original provenance, '
        'a new token, whole-original restoration, explicit replay-cost accounting and full state qualification. '
        'Its metadata gate passed 28 synthetic cases; AST comparison preserved nine science-bearing functions. '
        'No Torch/NumPy, task arrays, model or training execution was used in those checks.\n\n'
        'This is source preparation. The original nonfinal-epoch CPU audit is not implemented, '
        'and real composite model/Adam/RNG/CUDA recovery is not qualified. The new recovery gate requires '
        'those separate proofs and currently cannot authorize original recovery. It accepts only one '
        'fresh-origin recovery, with no recovery-chain ancestry. All unchanged v1 source is retained.\n\n'
        'On receipt of actual new nodes, verify endpoint/UUID/runtime/source/assets and fresh physical resources, '
        'actual lease and at least two hours of saving reserve before freezing any new stage. '
        'Old task weights are not fresh same-fold teachers/parents; every new composite method still needs '
        'its own original native/CPU and whole-original transport qualification. Historical TEST-selected '
        'all-data five-fold CV remains exploratory, not a new blind test.\n',encoding='utf8')
    proof=seal(payload,payload/'complete_future_nodes_preparation_original.zip')
    write(root/'C_preparation_receipt.json',proof)
    pointer=dict(local_root=str(root),payload=str(payload),human_availability=human,scope=scope,local_proof=proof)
    write(POINTER,pointer)
    print(json.dumps(dict(root=str(root),proof=proof,scope=scope),ensure_ascii=False))


def publish():
    from publish_test_selected_group5_preparation_v1 import DC,OUT,sync
    from raw_TRAIN_save_and_publish_v1 import publish as publish_small
    from group5_publish_exact_local_v1 import publish_tree
    pointer=json.loads(POINTER.read_bytes());local=Path(pointer['payload']);proof=pointer['local_proof']
    assert digest(proof['archive'])==proof['archive_SHA']
    assert shutil.disk_usage('D:/').free>40*1024**2+sum(p.stat().st_size for p in local.rglob('*') if p.is_file())
    root=DC.parent/Path(pointer['local_root']).name.replace('_preparation_','_saved_')
    payload=root/'payload';shutil.copytree(local,payload)
    for original in local.rglob('*'):
        if original.is_file():assert digest(original)==digest(payload/original.relative_to(local))
    saved=dict(proof,archive=str(payload/Path(proof['archive']).name),D_exact_copy_verified=True,
               D_verification_UTC=dt.datetime.now(dt.timezone.utc).isoformat())
    write(root/'D_receipt.json',saved)
    pointer.update(root=str(root),D_proof=saved);write(POINTER,pointer)
    publish_small(Path(saved['archive']),saved['archive_SHA'],'group5-future-nodes-source-'+saved['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
    github=publish_tree(payload,'results/'+root.name,'Preserve human future-node clarification and bounded composite recovery source; no new training')
    write(root/'GitHub_receipt.json',github);pointer.update(github=github);write(POINTER,pointer)
    state=json.loads((DC/'D_current_research_state.json').read_bytes())
    group=state['latest_human_TEST_selected_group5']
    group['latest_future_node_availability_and_source_preparation']=pointer
    group.update(method_fold_predictions_preserved=1,outer_scores_computed=False)
    state.update(updated_at_utc=github['actual_UTC'],github_source=github,
          next_gate='Human will provide new nodes after24h; current leases NOT extended. Await actual new endpoint/UUID/runtime/lease, preserve all originals. Bounded local preparation only; no new training/score/once dispatch.')
    sync(state)
    with (OUT/'研究接续状态.md').open('a',encoding='utf8') as stream:
        stream.write('\n'+github['actual_UTC']+' 人类明确24小时后提供新节点，非延长现有租期、非新有效执行证明；不重复索要、不冻新formal/once。复合v2恢复分支已接入准备源码，28合成元数据反例/9函数AST过，真实非最终epoch CPU审核未实现、模型Adam/RNG/CUDA恢复未资格，不冒执行。原1/25未评分预测/0评分不变；新原ZIP'+saved['archive_SHA']+'与GitHub'+github['commit']+'保存，D/C同字节。\n')
    print(json.dumps(dict(root=str(root),proof=saved,github=github),ensure_ascii=False))


def heartbeat():
    from publish_test_selected_group5_preparation_v1 import DC,OUT,sync
    from raw_TRAIN_save_and_publish_v1 import publish as publish_small
    from group5_publish_exact_local_v1 import publish_tree
    pointer=json.loads(POINTER.read_bytes());root=Path(pointer['root'])/'heartbeat_closure';root.mkdir()
    config=Path('C:/Users/21234/.codex/automations/automation/automation.toml')
    actual=tomllib.loads(config.read_text(encoding='utf8'))
    before=json.loads((WORK/'group5_automation_before_human_future_nodes.json').read_bytes())
    expected=(WORK/'group5_heartbeat_future_nodes_prompt.txt').read_text(encoding='utf8').rstrip('\n')
    assert actual['prompt']==expected
    assert {k:v for k,v in actual.items() if k not in ('prompt','updated_at')}=={k:v for k,v in before.items() if k not in ('prompt','updated_at')}
    assert actual['rrule']=='FREQ=MINUTELY;INTERVAL=10' and actual['status']=='ACTIVE'
    verify=dict(status='EXISTING_HEARTBEAT_FUTURE_NODES_PROMPT_VERIFIED_SCHEDULE_UNCHANGED',
          actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),prompt_SHA=hashlib.sha256(expected.encode()).hexdigest(),
          actual_TOML_SHA=digest(config),id=actual['id'],interval_minutes=10,active=True,
          all_other_fields_unchanged=True,new_automation_or_chat=False,human_future_nodes_not_lease_extension=True,
          new_node_details_pending=True,no_repeated_lease_question_now=True,preparation_original_SHA=pointer['D_proof']['archive_SHA'])
    write(root/'actual_heartbeat_verification.json',verify)
    shutil.copyfile(WORK/'group5_heartbeat_future_nodes_prompt.txt',root/'actual_prompt.txt')
    write(root/'source_preparation_and_human_reference.json',dict(proof=pointer['D_proof'],human_availability=pointer['human_availability'],github=pointer['github']))
    proof=seal(root,root/'complete_heartbeat_future_nodes_closure.zip');write(root.parent/'heartbeat_D_receipt.json',proof)
    publish_small(Path(proof['archive']),proof['archive_SHA'],'group5-future-nodes-heartbeat-'+proof['archive_SHA'][:12]+'.zip',root.parent/'heartbeat_Release_receipt.json')
    github=publish_tree(root,'results/'+root.parent.name+'_heartbeat','Keep existing quiet ten-minute heartbeat aligned with human future-node availability; current leases unchanged')
    write(root.parent/'heartbeat_GitHub_receipt.json',github)
    pointer['heartbeat_closure']=dict(verification=verify,proof=proof,github=github);write(POINTER,pointer)
    state=json.loads((DC/'D_current_research_state.json').read_bytes())
    state['latest_human_TEST_selected_group5']['latest_future_node_availability_and_source_preparation']=pointer
    state.update(updated_at_utc=github['actual_UTC'],github_source=github);sync(state)
    with (OUT/'研究接续状态.md').open('a',encoding='utf8') as stream:
        stream.write('\n'+github['actual_UTC']+' 原10min ACTIVE heartbeat同步人类24小时后新节点答复，当前三节点租期未延长；原频率/目标/其余字段逐项未变，实际TOML和原回执D/Release/GitHub'+github['commit']+'过，D/C同字节。\n')
    print(json.dumps(dict(verification=verify,proof=proof,github=github),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('prepare','publish','heartbeat'),required=True)
    args=parser.parse_args();{'prepare':prepare,'publish':publish,'heartbeat':heartbeat}[args.stage]()
