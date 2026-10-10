"""Preserve new inert driver and real standard-library synthetic observations."""
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tomllib

BASE=Path(__file__).resolve().parent.parent; WORK=BASE/'work'
sys.path.insert(0,str(WORK))
from group5_release_transport_v1 import digest,write,seal,verify_zip
POINTER=WORK/'group5_epoch_CPU_driver_source_pointer.json'


def prepare():
    stamp=dt.datetime.now(dt.timezone.utc)
    root=BASE/'outputs'/('group5_epoch_CPU_driver_source_prepared_'+stamp.strftime('%Y%m%dT%H%M%SZ'))
    payload=root/'payload'; payload.mkdir(parents=True)
    check=json.loads((WORK/'group5_stopped_epoch_CPU_driver_check_pointer.json').read_bytes())
    assert digest(check['report'])==check['SHA']; report=json.loads(Path(check['report']).read_bytes())
    assert report['passed_cases']==51 and not report['scientific_imports'] and not report['real_original_checkpoint_CPU_audit']
    names=('group5_stopped_epoch_CPU_driver_v1.py','check_group5_stopped_epoch_CPU_driver_v1.py',
           'group5_composite_stopped_epoch_CPU_audit_v1.py','group5_composite_epoch_resume_v2.py',
           'group5_release_transport_v1.py','group5_test_selected_contract_v1.py',Path(__file__).name)
    inventory=[]
    for name in names:
        source=WORK/name; ast.parse(source.read_text(encoding='utf8'))
        dest=payload/'source'/name; dest.parent.mkdir(exist_ok=True); shutil.copyfile(source,dest)
        inventory.append(dict(name=name,SHA=digest(dest),bytes=dest.stat().st_size))
    write(payload/'source_inventory.json',inventory)
    shutil.copyfile(check['report'],payload/'actual_final51_synthetic_report.json')
    old=WORK/'group5_CPU_driver_synthetic_20261010T161339316664Z/actual_synthetic_report.json'
    assert json.loads(old.read_bytes())['passed_cases']==49
    shutil.copyfile(old,payload/'actual_prior49_synthetic_report.json')
    current=(WORK/'group5_stopped_epoch_CPU_driver_v1.py').read_text(encoding='utf8')
    added="    if (not {'numpy', 'torch'} <= set(plan['runtime_versions']) or\n            plan['runtime_versions'] != origin['runtime_versions']):\n        raise PermissionError('Exact original scientific runtime package inventory required')\n"
    assert current.count(added)==1
    prior=current.replace(added,'',1)
    assert hashlib.sha256(prior.encode()).hexdigest()==json.loads(old.read_bytes())['source_SHA']
    (payload/'prior49_driver_reconstructed_by_exact_runtime_gate_reversal.py').write_text(prior,encoding='utf8',newline='\n')
    write(payload/'actual_check_observations.json',dict(record_UTC=stamp.isoformat(),
        prior_check=dict(tool_chunk_id='9c4e72',exit_code=0,passed_cases=49),
        final_check=dict(tool_chunk_id='76bcbf',exit_code=0,passed_cases=51),
        reason_for_final_check='Source review added complete original Torch/NumPy runtime inventory and two counterexamples.',
        no_test_rerun_for_preservation=True,no_invented_exact_exit_timestamp=True,
        prior_source_is_explicit_reconstruction_not_capture=True,prior_source_SHA_matches_original_report=True))
    previous=json.loads((WORK/'group5_composite_epoch_CPU_source_pointer.json').read_bytes())
    human=json.loads((WORK/'group5_future_nodes_pointer.json').read_bytes())
    write(payload/'original_predecessor_reference.json',dict(root=previous['root'],proof=previous['D_proof'],
        github=previous['heartbeat_closure']['github'],human_availability=human['human_availability']))
    template=dict(status='GROUP5_STOPPED_EPOCH_CPU_AUDIT_PREPARED',execution_enabled=False,
        reason='No actual stopped composite original or future human-provided node/lease is available.',
        actual_plan_must_bind=['method','fold','original_root','original_child','origin_plan','process_exit','checkpoint',
            'restoration','member_manifest','original_archive_SHA','original_archive_bytes','original_source_SHA','original_assets',
            'split_SHA','audit_source_SHA','human_node_lease','endpoint','GPU_UUID','lease_end_UTC','python','runtime_versions',
            'local_space','measured_remote_required_bytes','measured_audit_seconds','measured_saving_seconds',
            'audit_output_root','new_audit_once_token','no_fit_inference_score','no_new_target_decode','CPU_CUDA_recovery_qualification'],
        no_endpoint_UUID_or_lease_guessed=True,no_new_actual_audit_token_created=True)
    write(payload/'inert_plan_template.json',template)
    scope=dict(actual_UTC=stamp.isoformat(),status='STOPPED_EPOCH_CPU_DRIVER_SOURCE_PREPARATION_ONLY',
        independent_CPU_driver_source_implemented=True,actual_audit_plan_frozen=False,
        full_transitive_execution_capsule_assembly_pending=True,actual_original_checkpoint_CPU_audit=False,
        CPU_CUDA_recovery_qualification=False,metadata_synthetic_cases_passed=51,
        scientific_imports=False,task_array_or_target_decode=False,new_actual_audit_or_training_or_score_once_consumed=False,
        outputs_preserved_unscored=1,outer_scores=0,current_leases_unchanged=True,new_valid_node_details_not_received=True)
    write(payload/'preparation_scope.json',scope)
    (payload/'preparation.md').write_text(
        '# Prepared stopped-epoch CPU audit driver\n\n'
        'The independent driver is now prepared. A usable instance is still not frozen: no stopped composite original or new node has been provided. '
        'The included inert template rejects before physical commands, scientific imports or a real audit token.\n\n'
        'Fifty-one standard-library synthetic gate cases passed. They cover original method/fold and naturally observed exit, '
        'whole-original Release restoration/member identity, fresh node/runtime/source/assets/RAM/space, measured audit time plus at least two hours of saving, '
        'transitive source inventory, original live-child rejection and strict flags. AST checks put the gate before the new token and auditor import. '
        'The initial49-case report is retained; its prior driver source is explicitly reconstructed by reversing the runtime-inventory gate, with SHA matched.\n\n'
        'The future actual instance must bind exact originals, all transitive execution dependencies, the human-provided new node/actual lease, '
        'runtime and fresh resources, measured saving budget and a distinct audit token. This preparation capsule is not a runnable scientific dependency bundle. '
        'No original tensor audit, continuous-versus-restored CPU/CUDA comparison, recovery dispatch, task array/target decode, training, prediction or scoring occurred.\n\n'
        'Current leases are unchanged. The human will provide new nodes after24h. The fixed five-method,93-video comparison remains exploratory; '
        'one of25 unscored outputs is preserved, and no OUTER score has been computed.\n',encoding='utf8')
    proof=seal(payload,payload/'complete_CPU_driver_source_original.zip'); write(root/'C_receipt.json',proof)
    write(POINTER,dict(local_root=str(root),payload=str(payload),proof=proof,scope=scope))
    before=tomllib.loads(Path('C:/Users/21234/.codex/automations/automation/automation.toml').read_text(encoding='utf8'))
    write(WORK/'group5_automation_before_epoch_CPU_driver.json',before)
    print(json.dumps(dict(root=str(root),proof=proof,scope=scope)))


def publish():
    from publish_test_selected_group5_preparation_v1 import DC,OUT,sync
    from raw_TRAIN_save_and_publish_v1 import publish as small
    from group5_publish_exact_local_v1 import publish_tree
    p=json.loads(POINTER.read_bytes()); local=Path(p['payload'])
    root=DC.parent/Path(p['local_root']).name.replace('_prepared_','_saved_')
    needed=sum(v.stat().st_size for v in local.rglob('*') if v.is_file())
    assert shutil.disk_usage('C:/').free>200*1024**2 and shutil.disk_usage('D:/').free>40*1024**2+needed
    payload=root/'payload'; shutil.copytree(local,payload)
    for original in local.rglob('*'):
        if original.is_file(): assert digest(original)==digest(payload/original.relative_to(local))
    archive=payload/Path(p['proof']['archive']).name
    assert digest(archive)==p['proof']['archive_SHA']; actual=verify_zip(archive)
    proof=dict(p['proof'],archive=str(archive),D_exact_copy_verified=True,
               D_verification_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),**actual)
    write(root/'D_receipt.json',proof); p.update(root=str(root),D_proof=proof); write(POINTER,p)
    small(archive,proof['archive_SHA'],'group5-stopped-epoch-CPU-driver-source-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
    github=publish_tree(root,'results/'+root.name,'Prepare independent stopped-epoch CPU audit driver; original audit and recovery remain unexecuted')
    write(root/'GitHub_receipt.json',github); p['github']=github; write(POINTER,p)
    state=json.loads((DC/'D_current_research_state.json').read_bytes())
    state['latest_human_TEST_selected_group5']['latest_stopped_epoch_CPU_driver_preparation']=p
    state.update(updated_at_utc=github['actual_UTC'],github_source=github); sync(state)
    with (OUT/'研究接续状态.md').open('a',encoding='utf8') as stream:
        stream.write('\n'+github['actual_UTC']+' 独立完整epoch CPU审核driver源码仅准备，51项标准库合成门和惰性入口拒绝过；实际原件/新节点/冻结审核plan、真实CPU审核及CPU/CUDA连续恢复资格仍待。未运行SSH、任务模型/数组、训练推理评分或消费实际once；原1/25未评分预测与未延长租期不变。原ZIP'+proof['archive_SHA']+'、GitHub'+github['commit']+'保存，D/C同字节。\n')
    print(json.dumps(dict(root=str(root),proof=proof,github=github)))


def close():
    from publish_test_selected_group5_preparation_v1 import DC,sync
    from raw_TRAIN_save_and_publish_v1 import publish as small
    from group5_publish_exact_local_v1 import publish_tree
    p=json.loads(POINTER.read_bytes()); root=Path(p['root'])/'heartbeat_closure'; root.mkdir()
    config=Path('C:/Users/21234/.codex/automations/automation/automation.toml')
    actual=tomllib.loads(config.read_text(encoding='utf8'))
    before=json.loads((WORK/'group5_automation_before_epoch_CPU_driver.json').read_bytes())
    expected=(WORK/'group5_heartbeat_epoch_CPU_driver_prompt.txt').read_text(encoding='utf8').rstrip('\n')
    assert actual['prompt']==expected
    assert {k:v for k,v in actual.items() if k not in ('prompt','updated_at')}=={k:v for k,v in before.items() if k not in ('prompt','updated_at')}
    verify=dict(status='EXISTING_HEARTBEAT_DRIVER_PREPARATION_CURRENT_PROMPT_SCHEDULE_UNCHANGED',
        actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),prompt_SHA=hashlib.sha256(expected.encode()).hexdigest(),
        actual_TOML_SHA=digest(config),id=actual['id'],active=actual['status']=='ACTIVE',rrule_unchanged=True,
        target_and_other_fields_unchanged=True,no_new_automation_or_chat=True,no_new_original_training_audit_or_score=True)
    write(root/'actual_automation_verification.json',verify)
    shutil.copyfile(WORK/'group5_heartbeat_epoch_CPU_driver_prompt.txt',root/'actual_prompt.txt')
    write(root/'preparation_reference.json',dict(proof=p['D_proof'],scope=p['scope'],github=p['github']))
    proof=seal(root,root/'complete_heartbeat_closure.zip'); write(root.parent/'heartbeat_D_receipt.json',proof)
    small(Path(proof['archive']),proof['archive_SHA'],'group5-epoch-CPU-driver-heartbeat-'+proof['archive_SHA'][:12]+'.zip',root.parent/'heartbeat_Release_receipt.json')
    github=publish_tree(root,'results/'+root.parent.name+'_heartbeat','Preserve quiet heartbeat continuation for prepared CPU-audit driver; no original execution')
    write(root.parent/'heartbeat_GitHub_receipt.json',github)
    p['heartbeat_closure']=dict(verification=verify,proof=proof,github=github); write(POINTER,p)
    state=json.loads((DC/'D_current_research_state.json').read_bytes())
    state['latest_human_TEST_selected_group5']['latest_stopped_epoch_CPU_driver_preparation']=p
    state.update(updated_at_utc=github['actual_UTC'],github_source=github); sync(state)
    print(json.dumps(dict(proof=proof,github=github,verification=verify)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--stage',choices=('prepare','publish','close'),required=True)
    args=parser.parse_args(); {'prepare':prepare,'publish':publish,'close':close}[args.stage]()
