"""Preserve prepared epoch-audit source and real metadata-check observations."""
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tomllib

BASE=Path(__file__).resolve().parent.parent;WORK=BASE/'work'
sys.path.insert(0,str(WORK))
from group5_release_transport_v1 import digest,write,seal
POINTER=WORK/'group5_composite_epoch_CPU_source_pointer.json'


def prepare():
    stamp=dt.datetime.now(dt.timezone.utc)
    root=BASE/'outputs'/('group5_epoch_CPU_source_prepared_'+stamp.strftime('%Y%m%dT%H%M%SZ'))
    payload=root/'payload';payload.mkdir(parents=True)
    checks=json.loads((WORK/'group5_stopped_epoch_CPU_audit_preparation_pointer.json').read_bytes())
    assert digest(checks['report'])==checks['SHA'];report=json.loads(Path(checks['report']).read_bytes())
    assert report['passed_cases']==34 and not report['scientific_imports'] and not report['real_original_checkpoint_audit_executed']
    names=('group5_composite_stopped_epoch_CPU_audit_v1.py','check_group5_stopped_epoch_CPU_metadata_v1.py',
           'group5_composite_epoch_resume_v2.py',Path(__file__).name)
    inventory=[]
    for name in names:
        source=WORK/name;ast.parse(source.read_text(encoding='utf8'))
        dest=payload/'source'/name;dest.parent.mkdir(exist_ok=True);shutil.copyfile(source,dest)
        inventory.append(dict(name=name,SHA=digest(dest),bytes=dest.stat().st_size))
    write(payload/'source_inventory.json',inventory)
    shutil.copyfile(checks['report'],payload/'actual_final34_synthetic_report.json')
    old=WORK/'group5_stopped_epoch_CPU_metadata_20261010T134333327033Z/actual_synthetic_report.json'
    assert json.loads(old.read_bytes())['passed_cases']==33
    shutil.copyfile(old,payload/'actual_prior33_synthetic_report.json')
    current=(WORK/'group5_composite_stopped_epoch_CPU_audit_v1.py').read_text(encoding='utf8')
    prior=current.replace("or plan.get('execution_enabled') is not True:","or not plan.get('execution_enabled'):",1)
    prior=prior.replace("or plan.get('execution_enabled') is not True or (plan.get('resume')", "or (plan.get('resume')",1)
    assert hashlib.sha256(prior.encode()).hexdigest()==json.loads(old.read_bytes())['source_SHA']
    (payload/'prior_source_reconstructed_by_exact_two_line_reversal.py').write_text(prior,encoding='utf8',newline='\n')
    write(payload/'synthetic_observation_scope.json',dict(observed_record_UTC=stamp.isoformat(),
          prior_check=dict(tool_chunk_id='fad8fc',exit_code=0,passed_cases=33),
          final_check=dict(tool_chunk_id='2a7e75',exit_code=0,passed_cases=34),
          no_test_rerun_by_preservation=True,no_invented_exact_exit_timestamp=True,
          prior_source_is_reconstruction_not_capture=True,prior_source_SHA_matches_original_report=True,
          reason_for_final_check='Added strict boolean execution-enabled gate and an adversarial string-flag case after source review.',
          no_new_real_model_or_array_execution=True))
    previous=json.loads((WORK/'group5_future_nodes_pointer.json').read_bytes())
    write(payload/'previous_source_and_human_reference.json',dict(root=previous['root'],proof=previous['D_proof'],
          human_availability=previous['human_availability'],github=previous['heartbeat_closure']['github']))
    scope=dict(actual_UTC=stamp.isoformat(),status='STOPPED_EPOCH_CPU_AUDIT_SOURCE_PREPARATION_ONLY',
          nonfinal_epoch_CPU_audit_source_implemented=True,separately_frozen_audit_driver_pending=True,
          original_stopped_checkpoint_CPU_audit_executed=False,real_recovery_qualified=False,CUDA_recovery_qualified=False,
          metadata_synthetic_cases_passed=34,scientific_imports=False,new_fit_inference_score_or_task_target_decode=False,
          new_training_native_audit_or_score_once_consumed=False,outputs_preserved_unscored=1,outer_scores=0,
          current_leases_unchanged=True,future_new_node_details_not_received=True)
    write(payload/'preparation_scope.json',scope)
    (payload/'preparation.md').write_text(
        '# Prepared stopped-epoch composite CPU audit\n\n'
        'The new source audits a bound, naturally stopped fresh-origin composite checkpoint at a nonfinal complete epoch. '
        'It checks exact role IDs, same-fold parent/cache, durable current/selected state, all addon Adam ownership/moments/steps, '
        'RNG inventory, complete FIT orders/statistics, fixed tail/teacher and old-A frozen INNER history without recomputing a score. '
        'Message20 rejects INNER target access or checkpoint selection.\n\n'
        'Thirty-four standard-library synthetic metadata cases passed. AST checks place the actual 6GiB available-RAM gate '
        'before scientific imports and find no fit/forward/predict/backward/optimizer-step calls. '
        'The earlier 33-case report is retained; final source adds a strict boolean execution flag and an adversarial case. '
        'The prior source is explicitly reconstructed by reversing those two lines, with its SHA checked against the earlier report.\n\n'
        'This is preparation only. No original checkpoint, scientific library or task array was loaded for these checks. '
        'A separately frozen audit driver, exact original files and new audit token, runtime/resources/lease gates and real original CPU execution remain pending. '
        'This source supplies neither whole Release restoration nor continuous-versus-recovered CPU/CUDA qualification. '
        'No composite recovery or new formal training was dispatched.\n\n'
        'The human will provide new nodes after24h; current leases are unchanged. '
        'CaReFlow fold0 remains the one of25 preserved unscored outputs; fold1 native qualification is retained without formal training.\n',encoding='utf8')
    proof=seal(payload,payload/'complete_epoch_CPU_source_original.zip');write(root/'C_receipt.json',proof)
    write(POINTER,dict(local_root=str(root),payload=str(payload),proof=proof,scope=scope))
    config=Path('C:/Users/21234/.codex/automations/automation/automation.toml')
    write(WORK/'group5_automation_before_epoch_CPU_source.json',tomllib.loads(config.read_text(encoding='utf8')))
    print(json.dumps(dict(root=str(root),proof=proof,scope=scope)))


def publish():
    from publish_test_selected_group5_preparation_v1 import DC,OUT,sync
    from raw_TRAIN_save_and_publish_v1 import publish as small
    from group5_publish_exact_local_v1 import publish_tree
    p=json.loads(POINTER.read_bytes());local=Path(p['payload']);root=DC.parent/Path(p['local_root']).name.replace('_prepared_','_saved_')
    needed=sum(v.stat().st_size for v in local.rglob('*') if v.is_file())
    assert shutil.disk_usage('D:/').free>40*1024**2+needed and shutil.disk_usage('C:/').free>200*1024**2
    payload=root/'payload';shutil.copytree(local,payload)
    for old in local.rglob('*'):
        if old.is_file():assert digest(old)==digest(payload/old.relative_to(local))
    proof=dict(p['proof'],archive=str(payload/Path(p['proof']['archive']).name),D_exact_copy_verified=True,
               D_verification_UTC=dt.datetime.now(dt.timezone.utc).isoformat())
    write(root/'D_receipt.json',proof);p.update(root=str(root),D_proof=proof);write(POINTER,p)
    small(Path(proof['archive']),proof['archive_SHA'],'group5-stopped-epoch-CPU-source-'+proof['archive_SHA'][:12]+'.zip',root/'Release_receipt.json')
    github=publish_tree(payload,'results/'+root.name,'Prepare stopped-epoch composite CPU audit source with synthetic metadata checks; no original model execution')
    write(root/'GitHub_receipt.json',github);p['github']=github;write(POINTER,p)
    state=json.loads((DC/'D_current_research_state.json').read_bytes())
    state['latest_human_TEST_selected_group5']['latest_stopped_epoch_CPU_audit_source_preparation']=p
    state.update(updated_at_utc=github['actual_UTC'],github_source=github);sync(state)
    with (OUT/'研究接续状态.md').open('a',encoding='utf8') as f:
        f.write('\n'+github['actual_UTC']+' 新非最终完整epoch CPU审核源码仅准备，34项标准库合成元数据检查过；独立冻结driver、真实原件CPU与CPU/CUDA恢复资格仍待。未导入科学库/任务数组、未训练推理评分或消费once；原1/25未评分预测、租期未延长不变。原ZIP'+proof['archive_SHA']+'与GitHub'+github['commit']+'已保存，D/C同字节。\n')
    print(json.dumps(dict(root=str(root),proof=proof,github=github)))


def close():
    from publish_test_selected_group5_preparation_v1 import DC,OUT,sync
    from raw_TRAIN_save_and_publish_v1 import publish as small
    from group5_publish_exact_local_v1 import publish_tree
    p=json.loads(POINTER.read_bytes());root=Path(p['root'])/'heartbeat_closure';root.mkdir()
    config=Path('C:/Users/21234/.codex/automations/automation/automation.toml');actual=tomllib.loads(config.read_text(encoding='utf8'))
    before=json.loads((WORK/'group5_automation_before_epoch_CPU_source.json').read_bytes())
    expected=(WORK/'group5_heartbeat_epoch_CPU_source_prompt.txt').read_text(encoding='utf8').rstrip('\n')
    assert actual['prompt']==expected
    assert {k:v for k,v in actual.items() if k not in ('prompt','updated_at')}=={k:v for k,v in before.items() if k not in ('prompt','updated_at')}
    verify=dict(status='EXISTING_HEARTBEAT_SOURCE_PREPARATION_CURRENT_PROMPT_SCHEDULE_UNCHANGED',actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
          prompt_SHA=hashlib.sha256(expected.encode()).hexdigest(),actual_TOML_SHA=digest(config),id=actual['id'],
          active=actual['status']=='ACTIVE',rrule_unchanged=True,target_and_other_fields_unchanged=True,
          no_new_automation_or_chat=True,no_new_original_training_audit_or_score=True)
    write(root/'actual_automation_verification.json',verify)
    shutil.copyfile(WORK/'group5_heartbeat_epoch_CPU_source_prompt.txt',root/'actual_prompt.txt')
    write(root/'preparation_reference.json',dict(proof=p['D_proof'],scope=p['scope'],github=p['github']))
    proof=seal(root,root/'complete_heartbeat_closure.zip');write(root.parent/'heartbeat_D_receipt.json',proof)
    small(Path(proof['archive']),proof['archive_SHA'],'group5-epoch-CPU-source-heartbeat-'+proof['archive_SHA'][:12]+'.zip',root.parent/'heartbeat_Release_receipt.json')
    github=publish_tree(root,'results/'+root.parent.name+'_heartbeat','Keep quiet ten-minute heartbeat current with prepared stopped-epoch audit source; no new original execution')
    write(root.parent/'heartbeat_GitHub_receipt.json',github)
    p['heartbeat_closure']=dict(verification=verify,proof=proof,github=github);write(POINTER,p)
    state=json.loads((DC/'D_current_research_state.json').read_bytes())
    state['latest_human_TEST_selected_group5']['latest_stopped_epoch_CPU_audit_source_preparation']=p
    state.update(updated_at_utc=github['actual_UTC'],github_source=github);sync(state)
    print(json.dumps(dict(proof=proof,github=github,verification=verify)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=('prepare','publish','close'),required=True)
    args=parser.parse_args();{'prepare':prepare,'publish':publish,'close':close}[args.stage]()
