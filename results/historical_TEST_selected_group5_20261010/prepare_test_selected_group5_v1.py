"""Preserve the human-authorized historical-TEST-selected pooled video CV design.

Reads saved metric JSON and label-free IDs only. It never loads a task checkpoint,
pickle, target array, SSH client, scientific runtime, or training process.
"""
import ast
import datetime as dt
import hashlib
import importlib.util
import json
import pathlib
import shutil
import sys

P = pathlib.Path
BASE = P(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE / 'work'))
from raw_TRAIN_capture_v1 import raw, sha, seal

EV = P('D:/CodexBackups/selective_flow_20261003_1105/candidate_final100_actual_20261009T012656Z')
DC = EV.parent / 'candidate_posttrain_lowC_20261009T005229Z'
OLD = BASE / 'work/group5_flow_preparation_20261007T084136Z'
LATEST = BASE / 'work/A_candidate_recovery400_qualified_20261009T002805Z'
MESSAGE = BASE / 'work/anchored_increment_source_20261008T024739Z'
METRICS = ('Acc7', 'Acc2', 'F1', 'MAE', 'Corr')


def write(p, v):
    p.write_bytes(raw(v))


def five(d):
    return {k: d[k] for k in METRICS}


def dominates(a, b):
    signs = {k: -1 if k == 'MAE' else 1 for k in METRICS}
    differences = [signs[k] * (a[k] - b[k]) for k in METRICS]
    return min(differences) >= 0 and max(differences) > 0


def main():
    assert shutil.disk_usage('C:/').free > 200 * 1024**2
    assert shutil.disk_usage('D:/').free > 40 * 1024**2
    now = dt.datetime.now(dt.timezone.utc)
    root = EV / ('historical_TEST_selected_group5_prepared_actual_' + now.strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir()
    evidence = root / 'selection_evidence'
    evidence.mkdir()
    references = {
        'fixed_models.json': BASE / 'outputs/所有固定模型VAL_TEST五项实际对齐结果.json',
        'message_increment.json': BASE / 'outputs/anchored_VAL_TEST_review_20261008T052013Z/actual_VAL_TEST_five_metrics.json',
        'latest_AB.json': EV / 'AB_official_five_complete_saved_actual_20261009T180914Z/actual_verified_results.json',
        'old_preparation.json': OLD / 'preparation_protocol.json',
        'old_fixed_A_training.json': BASE / 'work/finite_formal_plan_v1.json',
        'message_training.json': MESSAGE / 'increment_train_protocol.json',
    }
    for name, path in references.items():
        shutil.copyfile(path, evidence / name)
    fixed = json.loads(references['fixed_models.json'].read_bytes())['models']
    message = json.loads(references['message_increment.json'].read_bytes())
    ab = json.loads(references['latest_AB.json'].read_bytes())
    roster = [
        dict(method='anchored_message20', label='原 AnchoredFlow + 固定20轮消息增量',
             historical_TEST=five(message['roles']['TEST']['metrics']['new_fixed20_upgrade']['overall']),
             historical_scope='POOLED_FOLD0_DESCRIPTIVE_WITH_FIT_AND_INNER_OVERLAP',
             historical_TEST_overlap=message['roles']['TEST']['training_overlap'],
             source_family='anchored_message', selection_reason='Best apparent historical TEST scores; overlap explicitly retained'),
        dict(method='old_fixed_A', label='旧固定 A / finite fixed',
             historical_TEST=five(fixed['old_A']['TEST']), historical_scope='OFFICIAL_SPLIT_TEST',
             source_family='old_fixed', selection_reason='Best official-split Acc2/F1/Corr among the selected in-house methods'),
        dict(method='factorized_aux', label='新 A / factorized_aux',
             historical_TEST=five(ab['A_other_node_original']['result']['roles']['TEST']['new']),
             historical_scope='OFFICIAL_SPLIT_TEST', source_family='latest_AB',
             selection_reason='Official-split classification and regression tradeoff; not dominated by old A or new B'),
        dict(method='regression_aux', label='新 B / regression_aux',
             historical_TEST=five(ab['B_author_five']['result']['roles']['TEST']['new']),
             historical_scope='OFFICIAL_SPLIT_TEST', source_family='latest_AB',
             selection_reason='Best official-split MAE and Acc7 among these in-house finalists'),
        dict(method='careflow', label='CaReFlow', historical_TEST=five(fixed['careflow']['TEST']),
             historical_scope='OFFICIAL_SPLIT_TEST', source_family='author', selection_reason='Required baseline'),
    ]
    selected = {r['method']: r for r in roster}
    excluded = []
    for name, winner in [('old_B', 'old_fixed_A'), ('old_C2', 'old_fixed_A'), ('minimal_fixed_F', 'regression_aux')]:
        loser = five(fixed[name]['TEST'])
        assert dominates(selected[winner]['historical_TEST'], loser)
        excluded.append(dict(method=name, historical_TEST=loser, dominated_by=winner,
                             same_historical_evaluation_scope=True, all_five_checked=True))
    assert message['original_test_overlap_in_training_disclosed'] and message['not_independent_official_benchmark']
    write(root / 'candidate_ledger.json', dict(
        actual_UTC=now.isoformat(), human_requests=['test表现最好的几次', '全数据约 93 视频五折', '把内容都上传github'],
        candidate_selection_uses_historical_TEST=True, new_blind_test=False,
        criterion='Retain three official-TEST nondominated representatives across all five saved metrics, plus the strongest overlapping pooled message architecture, and required CaReFlow',
        complete_historical_search_claimed=False, roster=roster, excluded=excluded,
        apparent_pooled_scores_not_ranked_as_independent_official_TEST=True,
        no_old_weights_or_cached_teacher_may_be_reused=True,
        original_once_tokens_untouched=True,
        saved_evidence={n: dict(origin=str(p), SHA=sha(p), bytes=p.stat().st_size) for n, p in references.items()}))

    source = root / 'source'
    source.mkdir()
    # These are read-only reference parents, not a dispatchable replacement.
    shutil.copytree(OLD / 'source', source / 'group_reference')
    dst = source / 'latest_AB_reference'
    dst.mkdir()
    for name in ['candidate_adapter.py', 'polarity_intensity_flow.py', 'controlled_flow.py',
                 'official_upgrade.py', 'incremental_message.py', 'anchored_flow.py']:
        shutil.copyfile(LATEST / name, dst / name)
    dst = source / 'message_reference'
    dst.mkdir()
    for name in ['train_increment.py', 'incremental_message.py', 'cache_original.py']:
        shutil.copyfile(MESSAGE / name, dst / name)
    dst = source / 'old_fixed_reference'
    dst.mkdir()
    old_plan = json.loads(references['old_fixed_A_training.json'].read_bytes())
    for name, expected in {**old_plan['source_sha256'], **{k:v for k,v in old_plan['base_source_sha256'].items() if '/' not in k}}.items():
        p = BASE / 'work' / name
        assert sha(p) == expected
        shutil.copyfile(p, dst / name)
    frozen = BASE / 'work/inflow_counterfactual_v5_deployment_20261005T0520Z'
    for rel, expected in old_plan['base_source_sha256'].items():
        if '/' not in rel:
            continue
        p = frozen / P(rel).name
        assert sha(p) == expected
        shutil.copyfile(p, dst / P(rel).name)
    shutil.copyfile(BASE / 'work/group5_test_selected_contract_v1.py', source / 'group5_test_selected_contract_v1.py')
    shutil.copyfile(P(__file__), root / P(__file__).name)
    split_path = source / 'group_reference/split.json'
    assert sha(split_path) == '1b49d2e3d4f790c3c88e1412f139f2b523396f8f4d755d270863700f5dc1ac48'
    split = json.loads(split_path.read_bytes())
    spec = importlib.util.spec_from_file_location('contract', source / 'group_reference/fold_contract.py')
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)
    contract.validate_folds(split['canonical_row_ids'], split['folds'])
    assert contract.make_folds(split['canonical_row_ids'], 128) == split['folds']
    assert split['rows'] == 2195 and split['videos'] == 93 and not split['labels_used_to_split']
    ast_count = 0
    for p in source.rglob('*.py'):
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
        ast_count += 1
    folds = [dict(fold=f['fold'], **f['rows'], videos={r: len(v) for r,v in f['video_ids'].items()},
                  steps_per_100_epoch_stage=((f['rows']['fit']+31)//32)*100,
                  steps_per_message20_stage=((f['rows']['fit']+31)//32)*20) for f in split['folds']]
    queue = []
    for f in folds:
        n = f['fold']
        # Parent stages are explicit costs, not independent repeats or free frozen teachers.
        for stage, method, parent, epochs in [
            ('parent', 'anchored_parent', None, 100),
            ('final', 'anchored_message20', 'anchored_parent', 20),
            ('teacher', 'old_A_teacher', None, 100),
            ('final', 'old_fixed_A', 'old_A_teacher', 100),
            ('final', 'factorized_aux', None, 100),
            ('final', 'regression_aux', None, 100),
            ('final', 'careflow', None, 100),
        ]:
            queue.append(dict(fold=n, stage=stage, method=method,
                              parent=None if parent is None else f'{parent}:fold{n}',
                              epochs=epochs, expected_updates=((f['fit']+31)//32)*epochs,
                              old_task_checkpoint_reuse=False, status='NOT_DISPATCHED',
                              once_consumed=False, requires_same_fold_parent=True))
    plan = dict(
        schema_version=1, status='HUMAN_TEST_SELECTED_POOLED_GROUP5_DESIGN_FIXED_EXECUTION_NOT_QUALIFIED',
        actual_UTC=now.isoformat(), execution_enabled=False,
        methods=[r['method'] for r in roster], method_fold_outputs=25, explicit_training_stages=35,
        end_to_end_parent_teacher_and_direct_stages=25, added_head_or_message_stages=10,
        formal_design_new_human_authorization=True, inherited_TRAIN_only_next_step_superseded_for_this_new_comparison=True,
        all_original_results_and_once_contracts_preserved=True,
        exploratory_historical_TEST_selected_repartitioned_CV=True, not_five_independent_seeds=True,
        data_scope=dict(rows=2195, video_groups=93, original_roles=split['original_role_rows'],
                        split_SHA=sha(split_path), raw_input_SHA='5c3cc6ab6b43c97d3e34b9767b456d8e48f9bc5cc6f02089292a29d0e0aafd4b',
                        input_identity_parent_SHA=split['input_identity_parent_sha256'],
                        unknown_upstream_recipe_units_sentinels_precision_exposure=True),
        task_seed=128, batch_size=32, drop_last=False, eval_batch_size=128,
        selection='Separate within-outer-fold video INNER; strict earliest minimum sample-weighted FP64 MSE over100. Fixed-final20 message stage has no further epoch selection.',
        method_preservation='Original architecture/objectives/LR preserved. Fresh random task initialization and public encoder for every fold. Normalize own models on FIT only; preserve and disclose author per-batch normalization. Per-fold new schedules use actual tail-inclusive update count.',
        old_A_teacher='Fresh same-fold teacher from public pretraining; prior fullTRAIN teacher, fitted scales and cached coordinates forbidden. Original teacher/adaptation pipeline needs an additional qualified fold constructor; do not substitute another teacher.',
        message_parent='Fresh AnchoredFlow100 INNER-selected parent for the same fold; freeze parent then fixed20 message-only stage. Original best36/20 checkpoints forbidden.',
        inner_scope='INNER may select each fold parent and final checkpoint; no OUTER-based tuning. No claimed nested search correction for historical candidate selection.',
        scoring='Freeze and preserve all25 final OUTER predictions plus model/optimizer/scheduler/RNG/orders/stats/source/identities and independent CPU checks before one new pooled OOF evaluation per method. Report all5 metrics, per-fold mean/SD and concatenated OOF. No outer-result-based queue edits.',
        folds=folds, queue=queue, complete_queue_updates=sum(q['expected_updates'] for q in queue),
        storage=dict(human_authorization='把内容都上传github', repository='fangxinly/huaweibei',
                     existing_release_tag='autonomous-flow-health-20261008',
                     permanent_repository='Source, protocols, IDs, hashes, receipts, summaries',
                     permanent_release='Exact original complete checkpoint/capture ZIP bytes, split into sub2GiB ranges when necessary, plus reassembly manifest and whole/archive/member SHA',
                     upload_chunk_bytes=512*1024**2, delete_old_evidence=False,
                     all25_D_full_state_reservation_required=False,
                     actual_serial_staging_capacity_required=True,
                     remote_original_keep_until_GitHub_complete_digest_and_restore_verified=True,
                     original_ZIP_permanent_preservation_in_GitHub_required=True,
                     CPU_available_RAM_min_bytes=6*1024**3, C_floor_bytes=200*1024**2, D_floor_bytes=40*1024**2,
                     available_C_bytes=shutil.disk_usage('C:/').free, available_D_bytes=shutil.disk_usage('D:/').free,
                     capacity_estimate_is_not_capture=True,
                     checkpoint_estimate_parent_reference_SHA=sha(references['old_preparation.json']),
                     estimated_full_states_25_bytes=25*3708000000,
                     extra_head_caches_and_prechecks_and_transfer_margin_not_included=True,
                     streaming_transport_and_restoration_still_need_qualification=True),
        pending=['New multi-method fold training/teacher/message constructors and scorer',
                 'Source/synthetic/native3step/fullcheckpoint CPU qualification',
                 'Fresh batch5 UUID/fullargv/runtime/assets/RAM/remote and serial staging space',
                 'Measured stage queue time + at least2h actual save/transfer reserve before lease end',
                 'New GitHub chunk upload and restoration qualification without SSH credentials in files/argv',
                 'Separate precheck/train/score execution plans and fresh once locks'],
        no_new_data_array_or_target_decode=True, no_new_fit_inference_scoring_SSH=True,
        local_source_manifest={p.relative_to(source).as_posix():dict(SHA=sha(p), bytes=p.stat().st_size) for p in sorted(source.rglob('*')) if p.is_file()})
    write(root / 'group5_design.json', plan)
    report = '''# 历史 TEST 优势方案与 CaReFlow：93 视频五折

按本聊天人类答复，采用全部2195条、93个视频；按历史TEST表现选择结构，并明确记为探索性重新划分CV。旧TEST已经参与候选挑选，不能称新的盲测。旧训练/诊断/评分及其once不重做。

| 方案 | 历史TEST MAE | Acc2 % | 历史评分范围 |
|---|---:|---:|---|
'''
    for r in roster:
        report += f"| {r['label']} | {r['historical_TEST']['MAE']:.8f} | {r['historical_TEST']['Acc2']*100:.4f} | {'包含FIT436/INNER88/OUTER161' if r['method']=='anchored_message20' else '原官方划分TEST'} |\n"
    report += '''
消息增量版的0.3663包含拟合/选模重叠，保留其结构参加新的五折，不能把旧分数当独立TEST优势。旧A、新A、新B在官方TEST的五项指标上各有取舍；旧B/C2被旧A覆盖，固定F被新B覆盖。只依据已列实际报告收敛代表方案，未声称穷尽所有历史模型。

沿用已保存的无标签视频划分：外折437/435/441/441/441条，每条恰好外折一次；每折另划按视频隔离的INNER。每折从公共DeBERTa初态重训，统计只用FIT。CaReFlow保持作者按批归一化的行为，样本顺序与评估批次固定。100轮、batch32、保留尾批；INNER使用全行FP64 MSE严格最早最优选模。消息分支保持固定20轮。

共25个最终方法×外折结果。实际训练有35个阶段：25个端到端/父模型/教师阶段，加5个消息增量及5个旧A增量阶段。旧A必须重新训练本折教师，消息版必须重新训练本折父模型；它们的成本单列。原上游处理与预训练曝光仍未知，分折重训不消除这一来源限制。

人类指定全部上传GitHub，采用现有仓库和Release：源码/报告/索引进Git，完整原件进入Release，每片512MiB并记录单片及整体SHA、ZIP CRC、成员SHA及还原索引。GitHub每资产须小于2GiB，单Release可有1000资产；本次片数实际检查后确定。原大件不重复上传；新原件在GitHub完整核验前继续保留。D不再被要求同时容纳全部模型，但单个新任务的临时保存、CPU审核、上传和还原门仍须真实通过。

现在已固定候选与划分，完成元数据/源码静态核验；没有加载本任务数组或标签，没有新训练、模型推理、计分或SSH。新增执行源码、教师复建、合成/真实预检和串行GitHub保存资格仍待，不能用旧preparation的过期UUID/租期或旧两方法训练器直接启动。每阶段只有资格与新once通过才调度，失败原件先保存。
'''
    (root / '五折名单与执行设计.md').write_text(report, encoding='utf-8')
    write(root / 'local_static_receipt.json', dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
          scope='Saved metrics/IDs/source AST only', historical_metrics_recomputed=False,
          source_AST_passed_files=ast_count, exact_split_regenerated=True,
          outer_rows_once=True, video_disjoint_FIT_INNER_OUTER=True,
          original_dataset_or_labels_loaded=False, torch_runtime_used=False,
          design_SHA=sha(root / 'group5_design.json')))
    proof = seal(root, 'complete_actual_TEST_selected_group5_preparation.zip')
    proof['actual_UTC'] = dt.datetime.now(dt.timezone.utc).isoformat()
    write(root / 'D_preservation_receipt.json', proof)
    pointer = BASE / 'work/test_selected_group5_pointer.json'
    write(pointer, dict(root=str(root), design_SHA=sha(root/'group5_design.json'), archive_SHA=proof['archive_SHA']))
    print(json.dumps(dict(root=str(root), proof=proof, queue_stages=len(queue), execution_enabled=False)))


if __name__ == '__main__':
    main()
