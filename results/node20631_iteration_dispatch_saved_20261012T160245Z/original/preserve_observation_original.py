"""Preserve observed live metadata; never relaunch training or decode model arrays."""
import datetime as dt
import json
import pathlib
import shutil
import sys

P = pathlib.Path
BASE = P(__file__).resolve().parent.parent
WORK = BASE / 'work'
sys.path.insert(0, str(WORK))
from group5_release_transport_v1 import digest, verify_zip, write
from freeze_node23668_careflow1_v4 import publish_complete_small
from publish_test_selected_group5_preparation_v1 import DC, sync


def main():
    local = BASE / 'outputs/node20631_iteration_actual_20261012T155936Z'
    observed = local / 'remote_original'
    original = observed / 'original'
    capture = json.loads((observed / 'capture_receipt.json').read_bytes())
    archive = original / 'complete_observation_original.zip'
    assert digest(archive) == capture['archive_SHA']
    verification = verify_zip(archive)
    inventory = json.loads((original / 'observation_inventory.json').read_bytes())
    assert inventory['scope'] == 'LIVE_TRAINING_SMALL_METADATA_OBSERVATION_NOT_COMPLETE_RUN'
    assert not inventory['model_arrays_decoded'] and not inventory['training_restarted']
    for row in inventory['source_files']:
        q = original / row['name']
        assert q.stat().st_size == row['bytes'] and digest(q) == row['SHA']

    ptr = json.loads((WORK / 'node20631_iteration_pointer.json').read_bytes())
    assert digest(original / 'root/plan.json') == ptr['plan_SHA']
    assert (original / 'root/plan.json').read_bytes() == (P(ptr['payload']) / 'plan.json').read_bytes()
    queue = json.loads((original / 'root/queue_status.json').read_bytes())
    assert queue['protocol'] == 'OFFICIAL_FIXED_SPLIT_VAL_ONLY_ITERATION'
    assert not queue['failure_hold'] and not queue['pending']
    assert {x['key'] for x in queue['active']} == {
        'careflow_seed4', 'adaptive_gated_aux_seed4', 'adaptive_gated_residual_seed4'}
    assert len({x['pid'] for x in queue['active']}) == 3
    actual_jobs = []
    for job in queue['active']:
        folder = original / 'jobs' / job['key']
        dispatch = json.loads((folder / 'actual_dispatch.json').read_bytes())
        once = json.loads((folder / 'out/train.once').read_bytes())
        progress = json.loads((folder / 'out/progress.json').read_bytes())
        health = json.loads((folder / 'out/health.json').read_bytes())
        assert dispatch['pid'] == job['pid'] and dispatch['plan_SHA'] == ptr['plan_SHA']
        assert once['plan_SHA'] == ptr['plan_SHA'] and once['method'] == job['method']
        assert health['step'] >= 40 and progress['steps'] >= 40
        assert health['missing_gradients'] == ([] if job['method'] != 'careflow' else [
            'dberta.pooler.dense.weight', 'dberta.pooler.dense.bias'])
        actual_jobs.append(dict(**job, dispatch_UTC=dispatch['actual_UTC'], once_UTC=once['actual_UTC'],
                                health=health, progress_observed=progress, training_complete=False))
    physical = json.loads((original / 'actual_physical_observation.json').read_bytes())
    assert physical['GPU'].split(',')[0] == 'GPU-e5ee2aee-86ad-49e1-58f5-f0d319129198'
    transport_failure = json.loads((observed / 'capture_transport_failure_observed.json').read_bytes())
    assert transport_failure['previous_shell_exit'] == 1 and not transport_failure['training_affected']
    assert shutil.disk_usage('C:/').free > 200 * 1024**2
    assert shutil.disk_usage('D:/').free > 40 * 1024**2

    now = dt.datetime.now(dt.timezone.utc)
    saved = DC.parent / ('node20631_iteration_dispatch_saved_' + now.strftime('%Y%m%dT%H%M%SZ'))
    saved.mkdir()
    pay = saved / 'payload'
    shutil.copytree(observed, pay / 'remote_observation_original')
    shutil.copyfile(__file__, pay / 'preserve_observation_original.py')
    shutil.copyfile('C:/Users/21234/.codex/automations/automation/automation.toml', pay / 'automation_current_original.toml')
    write(pay / 'observed_execution_summary.json', dict(
        actual_saved_UTC=now.isoformat(), observation_UTC=inventory['actual_UTC'],
        plan_SHA=ptr['plan_SHA'], protocol=queue['protocol'], jobs=actual_jobs,
        physical=physical, raw_observation_archive_SHA=capture['archive_SHA'],
        raw_observation_verification=verification, training_complete=False,
        CPU_final_audit_complete=False, full_model_Release_restoration_complete=False,
        TEST_inference_or_scoring_performed=False, old_fivefold_and15queue_not_started=True,
        metadata_capture_path_failure_preserved_and_corrected=True))
    (pay / '实际训练与方法迭代.md').write_text(
        '# 方法迭代实际启动记录\n\n'
        '官方 TRAIN/VAL/TEST 划分不变。本轮 seed4 的 CaReFlow 参照、输入自适应消息门控、'
        '门控加零初始化有界残差三个训练已实际同时运行；旧五折及三方法×五种子队列未启动。\n\n'
        '以 TRAIN 优化、VAL229 两批等权 FP64 MSE strict earliest 选择；本轮不索引 TEST 角色、'
        '不做 TEST 推理或评分。整个可信 pickle 会实体化其他角色，历史 TEST 也已见，不能声称新盲测。\n\n'
        '本包仅保存各文件分别观察到的进行中元数据，不能冒充训练完成、CPU 最终审核、'
        '完整模型 Release 下载还原或方法收益。三组最终自然退出及所有负结果仍须保存。\n\n'
        '快照归档首次因输出目录不符合 seal 的路径要求而退出 1；原源码与观察记录已保留，'
        '随后只对已复制的元数据在正确目录封装，未重启训练、未消费新的科学 once。\n', encoding='utf8')
    proof = publish_complete_small(saved, pay, 'node20631-iteration1-three-active-original-observation')
    supervisor = json.loads((original / 'root/supervisor_actual_dispatch.json').read_bytes())
    ptr.update(training_dispatched=True, once_consumed=True,
               supervisor_actual_dispatch=supervisor, supervisor_once_UTC=(original / 'root/supervisor.once').read_text(),
               observed_jobs=actual_jobs, observation_UTC=inventory['actual_UTC'],
               live_physical_observation=physical, actual_observation_preservation=proof,
               training_complete=False, final_CPU_audit_complete=False,
               full_model_Release_restoration_complete=False, TEST_scoring_performed=False)
    write(WORK / 'node20631_iteration_pointer.json', ptr)
    write(DC / 'node20631_iteration_pointer.json', ptr)
    state = json.loads((DC / 'D_current_research_state.json').read_bytes())
    state['latest_node20631_iteration'] = ptr
    state['latest_node20631_iteration_metadata_capture_failure'] = dict(
        root=str(saved), original=transport_failure, corrected_metadata_archive_SHA=capture['archive_SHA'],
        training_restarted=False, scientific_failure=False)
    if 'current_execution_before_node20631_iteration' not in state:
        state['current_execution_before_node20631_iteration'] = state.get('current_execution')
    state['current_execution'] = dict(protocol=queue['protocol'], training_active=3,
        methods=[x['method'] for x in queue['active']], seeds=[4],
        observation_UTC=inventory['actual_UTC'], pointer=str(WORK / 'node20631_iteration_pointer.json'),
        completion_or_improvement_not_claimed=True)
    state['next_gate'] = ('Continue the actual iteration1 three active jobs without rerunning once. '
        'Observe natural exits, preserve all full states including failures through local-credential Release relay '
        'and actual whole restoration, audit final CPU state once. Compare final VAL only, then freeze a bounded '
        'next method version; never launch old fivefold or15job queue or use TEST for iteration. '
        'Respect current20631 lease and at least7200s saving reserve.')
    state['updated_at_utc'] = now.isoformat()
    sync(state)
    with (BASE / 'outputs/研究接续状态.md').open('a', encoding='utf8') as f:
        f.write('\n最新 ' + now.isoformat() + '：方法迭代第1轮 seed4 三组已实际并行训练，'
            'CaReFlow/自适应消息门控/门控加有界残差；原各 train.once 与派发原件闭合，'
            '快照 GPU 实测100%，约13.3GiB显存。旧五折和15组队列未启动，TEST 无推理评分。'
            '仅进行中快照，不能冒完整训练、CPU最终审核或提升。元数据归档路径原失败1已保留修正，'
            '训练未受影响。D完整保存 ' + str(saved) + '；GitHub ' + proof['GitHub']['commit'] + '。\n')
    write(saved / 'state_sync_receipt.json', dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        D_C_identical=(DC / 'D_current_research_state.json').read_bytes() == (BASE / 'outputs/自主优化实际接续.json').read_bytes(),
        synced_state_SHA=digest(DC / 'D_current_research_state.json'), old_C_git_HEAD_untouched=True))
    print(json.dumps(dict(root=str(saved), GitHub=proof['GitHub']['commit'], Release=proof['Release']['url'],
        observation_UTC=inventory['actual_UTC'], active_jobs=3, training_complete=False, D_C_identical=True)))


if __name__ == '__main__':
    main()
