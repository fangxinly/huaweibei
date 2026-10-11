"""Preserve bounded preparation and an actual no-dispatch resource decision.

This script imports no scientific library, opens no dataset and consumes no
training/scoring token. Earlier synthetic observations are identified as such.
"""
import ast
import datetime as dt
import json
import pathlib
import shutil
import sys

P = pathlib.Path
BASE = P(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'work'))
from group5_release_transport_v1 import digest, seal, write
from group5_publish_exact_local_v1 import publish_tree
from raw_TRAIN_save_and_publish_v1 import publish
from publish_test_selected_group5_preparation_v1 import DC, OUT, sync


def main():
    native = json.loads((BASE / 'work/group5_careflow1_native_capture_pointer.json').read_bytes())
    formal = json.loads((BASE / 'work/group5_careflow0_complete_pointer.json').read_bytes())
    gate = json.loads((BASE / 'work/group5_careflow_next_gate_pointer.json').read_bytes())
    assert native['qualification']['status'] == 'SAME_METHOD_FOLD_NATIVE3_CPU_FULL_RELEASE_RESTORE_CLOSED'
    assert formal['closed_receipt']['status'] == 'CAREFLOW_FOLD0_FORMAL100_CPU_FULL_RELEASE_RESTORE_CLOSED_OUTER_UNSCORED'
    assert not gate['formal_training_dispatched'] and not gate['gate']['new_training_once_consumed']
    assert gate['gate']['status'] == 'NEW_TRAIN_NOT_FROZEN_LEASE_OR_STAGING_MARGIN'
    assert not gate['gate']['execution_enabled']
    assert shutil.disk_usage('C:/').free > 200 * 1024**2
    assert shutil.disk_usage('D:/').free > 40 * 1024**2
    stamp = dt.datetime.now(dt.timezone.utc)
    root = DC.parent / ('g5_continuation_no_new_lease_' + stamp.strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir()
    payload = root / 'payload'
    payload.mkdir()
    names = (
        'group5_composite_components_v1.py', 'group5_composite_runtime_v1.py',
        'group5_composite_CPU_audit_v1.py', 'group5_composite_epoch_resume_v1.py',
        'check_group5_composite_resume_history_v1.py', 'group5_pooled_score_v1.py',
        'check_group5_pooled_score_v1.py', 'freeze_group5_careflow_after_native_v1.py',
        P(__file__).name,
    )
    inventory = []
    for name in names:
        source = BASE / 'work' / name
        ast.parse(source.read_text(encoding='utf-8-sig'), filename=name)
        target = payload / 'source' / name
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        inventory.append(dict(name=name, SHA=digest(target), bytes=target.stat().st_size))
    write(payload / 'source_inventory.json', inventory)
    write(payload / 'actual_measured_stage_gate.json', gate['gate'])
    earlier = json.loads((BASE / 'work/group5_careflow_gate_before_exit_bound_fix.json').read_bytes())
    write(payload / 'earlier_receipt_time_gate_retained.json', earlier)
    write(payload / 'gate_exit_bound_correction.json', dict(
        original_gate_retained_unchanged=True,
        reason='Initial estimate ended at the result receipt. The current gate includes the original complete observed wrapper lifetime through natural exit.',
        current_total_stage_budget_seconds=gate['gate']['total_stage_budget_seconds'],
        original_total_stage_budget_seconds=earlier['gate']['total_stage_budget_seconds'],
        dispatch_decision_changed=False, new_scientific_execution=False,
    ))
    write(payload / 'earlier_exit_bound_gate_staging_state.json', json.loads(
        (BASE / 'work/group5_careflow_gate_before_transport_relocation.json').read_bytes()))
    relocation = json.loads((BASE / 'work/group5_N1_runtime_complete_pointer.json').read_bytes())
    write(payload / 'new_transport_duplicate_relocation.json', relocation['second_relocation'])
    write(payload / 'closed_original_references.json', dict(
        formal_fold0=formal['closed_receipt'], formal_preservation_root=formal['preservation_root'],
        native_fold1=native['qualification'], native_preservation_root=native['preservation_root'],
        native_publication=native['github'],
    ))
    scope = dict(
        actual_UTC=stamp.isoformat(), status='PREPARATION_PRESERVED_NO_NEW_FORMAL_TRAIN_DISPATCH',
        methods=['anchored_message20', 'old_fixed_A', 'factorized_aux', 'regression_aux', 'careflow'],
        rows=2195, videos=93, final_method_fold_outputs_required=25, training_stages_required=35,
        preserved_unscored_method_fold_predictions=1, outer_score_outputs=0,
        historical_TEST_selected_exploratory_repartition_CV=True, new_blind_test=False,
        source_AST_passed_files=len(inventory), scientific_source_imported=False,
        new_dataset_array_or_target_decode=False, new_model_fit_inference_score=False,
        new_training_or_scoring_once_consumed=False,
        prior_tool_observations=dict(
            composite_history_10_synthetic_cases_passed=True,
            pooled_score_10_synthetic_cases_passed=True,
            scope='Earlier tool executions in this turn, not new executions or reconstructed original receipts.',
        ),
        composite_full_model_Adam_RNG_resume_executed=False,
        composite_CUDA_resume_executed=False,
        composite_resume_helper_integrated_into_runtime=False,
        composite_runtime_v1_continues_to_reject_resume=True,
        composite_native_and_parent_original_qualification_pending=True,
        all_25_original_scoring_preflight_not_passed=True,
        human_new_valid_lease_question_already_pending=True,
        no_inferred_extension_or_renewal=True,
    )
    write(payload / 'preparation_scope.json', scope)
    text = (
        '# Current five-fold continuation\n\n'
        'CaReFlow fold0 completed 100 epochs / 4700 updates, original CPU audit, '
        'complete immutable Release publication and actual whole download restoration. '
        'Its 437 outer predictions remain unscored. This is 1 of 25 required method/fold outputs.\n\n'
        'CaReFlow fold1 native3 and original CPU audit are separately complete, including '
        'actual whole Release restoration. Native3 is qualification cost; formal fold1 was not dispatched.\n\n'
        'The measured stage gate includes target native FIT projection and actual fold0 '
        'non-FIT costs, with explicit margins, plus at least 7200 seconds for saving. '
        'The current conservative lease cannot cover that total. No new formal plan/token '
        'was frozen or consumed. A question for a new valid human-provided lease is pending.\n\n'
        'Composite and scoring source here is preparation. The composite resume helper '
        'is not integrated or qualified on full model/Adam/RNG/CUDA state. The original '
        'runtime still rejects resume. Prior synthetic history/scoring checks are not '
        'original-model qualification and are not rerun by this capture.\n\n'
        'Continue with explicit current lease/endpoint/UUID binding, fresh runtime/source/assets '
        'and physical resource checks before any new native or formal stage. Fresh same-fold '
        'parents/teachers and independent CPU plus whole original GitHub restore remain required. '
        'Keep every old result, original and consumed token. Do not change this queue from outer scores.\n'
    )
    (payload / 'continuation.md').write_text(text, encoding='utf8')
    proof = seal(payload, payload / 'complete_current_continuation_original.zip')
    write(root / 'D_receipt.json', proof)
    publish(P(proof['archive']), proof['archive_SHA'],
            'group5-continuation-no-new-lease-' + proof['archive_SHA'][:12] + '.zip',
            root / 'Release_receipt.json')
    github = publish_tree(payload, 'results/group5_current_continuation_' + stamp.strftime('%Y%m%dT%H%M%SZ'),
                          'Preserve actual measured lease gate and bounded five-fold preparation; one unscored output, no new training')
    write(root / 'GitHub_receipt.json', github)
    result = dict(root=str(root), preparation=scope, gate=gate['gate'], proof=proof, github=github)
    state = json.loads((DC / 'D_current_research_state.json').read_bytes())
    group = state['latest_human_TEST_selected_group5']
    group['latest_no_new_lease_continuation'] = result
    group.update(method_fold_predictions_preserved=1, outer_scores_computed=False)
    state.update(updated_at_utc=github['actual_UTC'], github_source=github,
                 next_gate='Current measured full stage plus at least2h saving exceeds conservative lease. New valid human lease pending; local bounded preparation only, no training/scoring dispatch or consumed original once repeated.')
    sync(state)
    write(BASE / 'work/group5_current_continuation_pointer.json', result)
    with (OUT / '研究接续状态.md').open('a', encoding='utf8') as f:
        f.write('\n' + github['actual_UTC'] + ' 第0折正式100轮4700更新及CPU/完整GitHub还原已闭合，1/25未评分预测；第1折native3/CPU/整件还原闭合，但实测下一阶段预算与至少2h保存余量超过当前保守租期，未冻新formal/未消费once。新租期人类问题待答；复合恢复与统一评分源码仅准备，全部旧结果/原件不重做。新接续原ZIP' + proof['archive_SHA'] + '与D bare exactblob GitHub' + github['commit'] + '保存、D/C同字节。\n')
    print(json.dumps(dict(root=str(root), proof=proof, github=github, status=scope['status'])), flush=True)


if __name__ == '__main__':
    main()
