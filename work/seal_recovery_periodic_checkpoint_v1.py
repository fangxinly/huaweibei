"""Preserve small actual rolling-checkpoint receipts without fetching large states."""
import argparse, hashlib, json, os, shutil, zipfile
from pathlib import Path

base = Path(__file__).resolve().parents[1]
ev = base / 'outputs/polarity_intensity_prefix_v2_20261008T203225Z'
p = argparse.ArgumentParser()
p.add_argument('--stamp', required=True)
p.add_argument('--tag', required=True)
a = p.parse_args()

def read(path):
    return json.loads(path.read_text(encoding='utf8'))

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

files, proofs = [], {}
for node in ('A', 'B'):
    ready_path = ev / (node + '_recovery400_epoch20_checkpoint_ready.json')
    observed_path = ev / (node + '_recovery400_epoch20_checkpoint_observed.json')
    progress_path = ev / (node + '_recovery400_epoch20_observed_progress.json')
    old_path = ev / (node + '_interrupted_history_original.json')
    ready, observed, progress, old = map(read, (ready_path, observed_path, progress_path, old_path))
    assert ready['steps'] == 800 and ready['full_model_Adam_scheduler_RNG_orders_history']
    assert observed['ready'] == ready and observed['checkpoint_exists'] and observed['child_process_exists']
    assert observed['checkpoint_actual_bytes'] == ready['checkpoint_bytes'] == 2960840167
    assert len(ready['checkpoint_SHA']) == 64 and observed['free_bytes'] > 10_000_000_000
    assert progress['epoch'] >= 20 and progress['steps'] == progress['epoch'] * 40
    expected = old[progress['epoch'] - 1]
    for field in ('state_SHA', 'prediction_SHA', 'DEV_batch_MSE'):
        assert progress[field] == expected[field], (node, field)
    proofs[node] = dict(checkpoint_ready_original=ready, remote_stat_original=observed,
                        observed_progress_original=progress, recorded_trajectory_exact=True,
                        large_checkpoint_not_downloaded_or_independently_CPU_audited=True,
                        no_final_VAL_TEST_five=True)
    files.extend((ready_path, observed_path, progress_path, old_path))

assert shutil.disk_usage('C:/').free > 200_000_000
assert shutil.disk_usage('D:/').free > 500_000_000
dest = Path('D:/CodexBackups/selective_flow_20261003_1105') / ('matched_recovery_checkpoint800_' + a.tag)
dest.mkdir()
archive = dest / 'checkpoint800_small_actual_receipts.zip'
members = {f.name: sha(f) for f in files}
with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        z.write(f, f.name)
    z.writestr('member_SHA.json', json.dumps(members))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist()) == len(set(z.namelist()))
    for name, digest in members.items():
        assert hashlib.sha256(z.read(name)).hexdigest() == digest
receipt = dict(status='BOTH_PERIODIC800_CHECKPOINTS_REAL_READY_AND_TRAJECTORY_RECEIPTS_C_D_VERIFIED',
               actualclock_UTC=a.stamp, nodes=proofs, D_archive=str(archive),
               archive_SHA=sha(archive), archive_bytes=archive.stat().st_size,
               original_large_interrupted_states_Release_and_peer_CPU_already_saved=True,
               periodic_large_states_remote_only=True, overall_complete=False)
(ev / 'recovery_checkpoint800_C_D_verification.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
state_path = base / 'outputs/自主优化实际接续.json'
state = read(state_path)
pair = state['polarity_intensity_pair_v2']
pair['periodic800_actual'] = receipt
for node, row in proofs.items():
    pair['recovery400_actual'][node]['latest_progress_original'] = row['observed_progress_original']
    pair['recovery400_actual'][node]['latest_mutable_checkpoint_ready_original'] = row['checkpoint_ready_original']
pair['full100_complete'] = False
pair['no_new_final_VAL_TEST_five'] = True
state['updated_at_utc'] = a.stamp
state['actualclock_before_snapshot_UTC'] = a.stamp
state['current_capacity'] = {node: shutil.disk_usage(node + ':/').free for node in ('C', 'D')}
state['next_gate'] = 'Healthy detached recovery -> real100 natural0 full originals Release -> independent full4000 CPU -> same selected official VAL/TEST predictions saved -> author all5 once and independent NPZ audit. No TEST structure selection.'
temp = state_path.with_name('自主优化实际接续_checkpoint800.tmp')
temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf8')
os.replace(temp, state_path)
epochs = '/'.join(str(proofs[n]['observed_progress_original']['epoch']) for n in ('A', 'B'))
short = (a.stamp + ' 两臂脱离SSH恢复训练真实继续，最新原进度A/B第' + epochs + '轮；第20轮800步各2960840167B完整可变检查点真实落盘，原writer SHA与独立文件stat一致，小原回执C/D SHA/CRC/唯一成员过。恢复参数/预测SHA与中断前历史完全一致。此800大状态仅远端，未冒新Release或800异节点CPU；中断完整原件Release及400异节点CPU已通过。原中断原因未知、实际额外重放3493/3473更新仍计成本；旧冻源不删。GitHub main ' + state['github_source']['commit'][:8] + '。尚无新最终VAL/TEST五项；100自然0后立即完整Release+异节点4000CPU再官方五项。整体未完成。\n\n')
status_path = base / 'outputs/研究接续状态.md'
status_path.write_text(short + status_path.read_text(encoding='utf8'), encoding='utf8')
print(json.dumps(dict(status=receipt['status'],latest_epochs=epochs,D_archive=str(archive),archive_SHA=receipt['archive_SHA'],archive_bytes=receipt['archive_bytes']), ensure_ascii=True))
