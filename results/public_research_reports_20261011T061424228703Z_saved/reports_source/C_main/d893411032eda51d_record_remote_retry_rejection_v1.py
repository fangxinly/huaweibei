"""Preserve the actual original-tool rejection; never contact a remote node."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import zipfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


root = Path(__file__).resolve().parents[1]
now = datetime.datetime.now(datetime.timezone.utc)
stamp = now.strftime('%Y%m%dT%H%M%SZ')
free = {drive: shutil.disk_usage(drive).free for drive in ('C:/', 'D:/')}
assert free['D:/'] >= 1024**3
out = root / 'outputs' / ('用户请求GPU重试实际拒绝_' + stamp + '.json')
assert not out.exists()
record = {
    'status': 'HUMAN_REQUESTED_ORIGINAL_WRITE_STDIN_RETRY_REJECTED_BEFORE_TRANSMISSION',
    'recorded_utc': now.isoformat(),
    'human_request': '你自己不能连接gpu做实验吗现在',
    'tool': 'write_stdin', 'session_id': 11935,
    'attempted_command': 'nvidia-smi --query-gpu=uuid,name,memory.used,memory.total --format=csv',
    'actual_error': 'write_stdin rejected: approval required by policy, but AskForApproval is set to Never',
    'remote_command_transmitted': False, 'GPU_experiment_started': False,
    'new_connection_or_alternate_command_client_tool_after_rejection': False,
    'current_GPU_UUID_compute_argv_space': 'unknown; no successful current query',
    'SSH_session_exit': 'unknown; no exit0 claim',
    'server_or_password_failure_established': False,
    'tool_repaired_or_approval_restored': False, 'fresh_local_free_bytes': free,
    'latest_science_unchanged': 'Three-stage design and label-known output-simplex diagnostic already independently verified and D-preserved; not GPU/CAL/generalization.',
    'source_sha256': sha(__file__)}
out.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
state = root / 'outputs/研究接续状态.md'
prefix = ('最新远端工具重试UTC' + now.isoformat() + '：用户明确要求当前GPU连接核查后，原write_stdin向C会话11935发送nvidia-smi查询仍被实际拒绝：'
          'approval required by policy, but AskForApproval is set to Never。命令未传远端，无新GPU实验；会话退出/当前UUID、compute、argv、空间未知。'
          '未换命令/连接/客户端/工具绕过，未认定服务器或密码异常，未称工具已修复。证据' + out.name + '，本轮永久D保存；'
          '没有新的正式恢复证据不继续重试。已完成科学证据不变，新校准控制未执行。\n\n')
state.write_text(prefix + state.read_text(encoding='utf-8'), encoding='utf-8')
destination = Path('D:/CodexBackups/selective_flow_20261003_1105') / ('human_gpu_retry_rejection_' + stamp)
destination.mkdir(exist_ok=False)
files = {'original_tool_rejection.json': out, 'continuation.md': state, 'record_source.py': Path(__file__)}
manifest = {}
for name, source in files.items():
    target = destination / name
    shutil.copyfile(source, target)
    assert sha(target) == sha(source)
    manifest[name] = {'sha256': sha(target), 'bytes': target.stat().st_size}
archive = destination / 'evidence.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for name in files:
        z.write(destination / name, name)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and set(z.namelist()) == set(files)
    assert len(z.namelist()) == len(set(z.namelist()))
    for name in files:
        assert hashlib.sha256(z.read(name)).hexdigest() == manifest[name]['sha256']
proof = {'status': 'LOCAL_ACTUAL_REJECTION_EVIDENCE_D_SHA_CRC_UNIQUE_MEMBERS_PASSED',
         'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
         'backup_directory': str(destination), 'manifest': manifest, 'zip_sha256': sha(archive),
         'remote_capture_or_GPU_work': False, 'source_query_received_remote': False}
proof_path = out.with_name(out.stem + '_永久保存.json')
proof_path.write_text(json.dumps(proof, indent=2, ensure_ascii=False), encoding='utf-8')
shutil.copyfile(proof_path, destination / 'preservation_proof.json')
assert sha(proof_path) == sha(destination / 'preservation_proof.json')
print(json.dumps({'status': proof['status'], 'evidence': str(out), 'D': str(destination),
                  'zip_sha256': proof['zip_sha256'], 'query_sent': False, 'new_GPU_work': False}))
