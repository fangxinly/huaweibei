"""Seal actual 13:07 snapshots; scheduled 13:00 is not the capture timestamp."""
import hashlib, json, shutil, sys, zipfile
from pathlib import Path

base = Path('D:/CodexBackups/selective_flow_20261003_1105')
root = base / 'lease_dynamic_actual_20261007T130645Z'
clock = sys.argv[1]
read = lambda p: json.loads(p.read_text(encoding='utf8'))
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
uuids = {'A': 'GPU-53696803-875e-eec8-2231-29db63579891',
         'B': 'GPU-8897bd58-4213-b200-8801-6ac6bfc55f3f',
         'C': 'GPU-417d3577-0525-788b-7296-0808a0f52012'}
qualified = {'5a58f50c554890baa4324d49437155649a92c68032dcbc6d88138a6269e6ed31': 2966458931}
prior = read(Path('outputs/正式双方fullTRAIN100完成与完整保存实际结果.json'))
for method in prior['methods'].values():
    for state in method['whole_D_files'].values():
        qualified[state['sha256']] = state['bytes']
captures = {}
for node in ('A', 'B', 'C'):
    receipt = read(root / (node + '_dynamic_receipt.json'))
    archive = root / (node + '_dynamic_capture.zip')
    assert sha(archive) == receipt['sha256']
    assert archive.stat().st_size == receipt['bytes']
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist())) == receipt['members']
        m = json.loads(z.read('dynamic_manifest.json'))
        p = json.loads(z.read('physical.json'))
        plan = json.loads(z.read('pinned_asset_runtime_plan.json'))
        for name, value in m['member_sha256'].items():
            assert hashlib.sha256(z.read(name)).hexdigest() == value
        assert p['uuid'] == uuids[node] and p['node'] == node
        assert p['operator_source_sha256'] == 'b64dfc4a91eafca4e5da9dccb8e0bca5246b2d7f41de49b4fb63cfefb9387a0a'
        assert p['native_runtime'] == plan['runtime_exact_versions']
        assert p['public_asset_sha256'] == plan['asset_sha256']
        assert p['fullargv'][1].endswith('/lease_dynamic_completed_refs_capture_v2.py') and p['pid'] > 0
        for ref in m['large_original_refs']:
            assert not ref['new_large_original_requires_D_transfer']
            assert qualified[ref['sha256']] == ref['bytes']
        captures[node] = dict(actual_start_utc=m['actual_capture_start_utc'], actual_finish_utc=m['actual_capture_finish_utc'],
            archive_SHA=receipt['sha256'], archive_bytes=receipt['bytes'], members=receipt['members'], uuid=p['uuid'],
            compute=p['compute'], fullargv=p['fullargv'], large_original_refs=m['large_original_refs'],
            source_SHA=m['source_file_sha256'], free_bytes=p['free_bytes'], original_roots=m['original_roots'],
            operator_exit0_observed_in_original_tool_output=True)
report = dict(status='THREE_REAL_DYNAMIC_NODE_CAPTURES_D_SHA_CRC_UNIQUE_SOURCE_UUID_RUNTIME_QUALIFIED_COMPLETED_STATE_REFS',
    actualclock_D_record_UTC=clock, scheduled_window_UTC='2026-10-07T13:00:00+00:00', captures=captures,
    capture_late_actual_time_not_backfilled=True, large_originals_previously_D_qualified_no_duplicate_download=True,
    original_workers_or_saved_states_changed=False, conservative_lease_end_UTC='2026-10-07T13:30:00+00:00',
    platform_expiry_confirmed=False, D_free_bytes=shutil.disk_usage('D:/').free,
    earlier_SFTP_sessions_closed_exit1=[73248,27032,54572], earlier_SFTP_failures_did_not_change_remote_or_D_originals=True)
text = json.dumps(report, ensure_ascii=False, indent=2)
(root / 'actual_D_dynamic_joint.json').write_text(text, encoding='utf8')
Path('outputs/第二租期13点真实动态保存实际接续.json').write_text(text, encoding='utf8')
Path('outputs/第二租期13点真实动态保存实际接续.md').write_text(
    '13:00计划窗口的三机真实动态保存已完成D核验。\n\n'
    f'本地记录 {clock}。实际采集在13:07进行，A/B/C回执分别13:07:17、13:07:18、13:07:21；不回填13:00。'
    '三件ZIP的SHA、CRC、唯一成员、原source/fullargv、物理UUID、公共资产和runtime通过。'
    '新40轮以及CaReFlow完整模型状态fresh SHA匹配此前D完整原件，以refs保存，未重下载大权重。'
    '动态快照不替代原任务自然退出证明。旧等待认证的三SFTP连接实际exit1，新连接成功后完成下载，旧ID禁复用。\n\n'
    '13:30为保守执行界限，平台到期未核；不续租、释放、关机、停止健康训练或修改原资料。\n', encoding='utf8')
print(json.dumps(dict(status=report['status'], joint_SHA=sha(root/'actual_D_dynamic_joint.json'), D_free_bytes=report['D_free_bytes'])))
