"""Pure file/association gates shared by prospective original CPU and capture tools.

No model, Torch, dataset, or label imports. Large references never imply downloads.
"""
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile

FROZEN = 'PAIRED_FULLTRAIN_RUNTIME_EXECUTION_PROTOCOL_FROZEN'
PRECHECK_JOINT = 'ACTUAL_PAIRED_METHOD_PRECHECK_GPU_D_OTHER_CPU_CAPTURE_JOINT_PASSED'
TRAIN_JOINT = 'ACTUAL_PAIRED_METHOD_FULLTRAIN_GPU_D_OTHER_CPU_CAPTURE_JOINT_PASSED_NO_FINAL_TEST'

def require(condition, message):
    if not condition:
        raise ValueError(message)

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    p = Path(path)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(p)

def safe_member(name):
    p = PurePosixPath(name)
    require(not p.is_absolute() and '\\' not in name and ':' not in name and
            all(x not in ('', '.', '..') for x in name.split('/')), 'Unsafe ZIP member')
    return name

def zip_audit(path, expected_sha=None):
    if expected_sha is not None:
        require(sha(path) == expected_sha, 'Whole ZIP SHA mismatch')
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) and names, 'Duplicate/empty ZIP')
        for n in names:
            safe_member(n.rstrip('/'))
        require(z.testzip() is None, 'Original whole ZIP CRC failure')
    return {'bytes': Path(path).stat().st_size, 'sha256': sha(path), 'unique_members': len(names), 'CRC': True}

def plan_gate(bundle, expected_sha):
    bundle = Path(bundle)
    p = bundle / 'paired_fulltrain_execution_plan.json'
    require(sha(p) == expected_sha, 'Original execution plan SHA mismatch')
    plan = read(p)
    require(plan.get('status') == FROZEN, 'Local preparation cannot audit or capture actual research')
    for n, h in plan['source_sha256'].items():
        safe_member(n)
        require(sha(bundle / n) == h, 'Frozen source SHA mismatch: ' + n)
    require(plan.get('original_CPU_and_root_specific_capture_sources_frozen') is True,
            'Complete original preservation source dependencies absent')
    return plan

def stage_association(root, plan_sha, expected_method=None):
    """root may be the original root or its physical saved copy; argv names the original."""
    root = Path(root)
    receipt = read(root / 'out/actual_stage_receipt.json')
    ex = read(root / 'natural_exit.json')
    launch = read(root / 'actual_child_launch.json')
    wrapper = read(root / 'wrapper_actual_start.json')
    command = ex.get('full_argv')
    require(isinstance(command, list) and command == launch.get('full_argv') == wrapper.get('child_full_argv'),
            'Original complete child argv association mismatch')
    require(ex.get('natural_exit') is True and ex.get('exit_code') == 0 and
            ex.get('child_pid') == launch.get('pid'), 'Original natural completion absent')
    require(ex.get('receipt_sha256') == sha(root / 'out/actual_stage_receipt.json'), 'Original receipt/exit SHA mismatch')
    require(command[1:] == receipt.get('argv'), 'Original receipt full argv mismatch')
    require(receipt.get('plan_sha256') == plan_sha, 'Original plan association mismatch')
    for flag, value in (('--root', receipt.get('root')), ('--bundle', receipt.get('source_bundle')), ('--plan-sha', plan_sha)):
        require(command.count(flag) == 1 and command[command.index(flag) + 1] == value, 'Original argv field mismatch: ' + flag)
    if expected_method is not None:
        require(receipt.get('method') == expected_method, 'Original method mismatch')
    return receipt, ex

def precheck_training_gate(args, plan):
    """Executed before Torch/model construction and before any TRAIN label access."""
    require(args.precheck_root and args.precheck_joint and args.precheck_joint_sha, 'Original completed precheck joint required')
    require(sha(args.precheck_joint) == args.precheck_joint_sha, 'Precheck joint whole SHA mismatch')
    joint = read(args.precheck_joint)
    parent, _ = stage_association(args.precheck_root, args.plan_sha, args.method)
    require(parent['status'] == 'ACTUAL_PAIRED_METHOD_PRECHECK2_COMPLETE_NOT_TRAINED100', 'Wrong original precheck status')
    require(joint.get('status') == PRECHECK_JOINT and joint.get('method') == args.method and
            joint.get('plan_sha256') == args.plan_sha and joint.get('original_root') == parent['root'], 'Precheck joint identity mismatch')
    require(joint.get('original_stage_receipt_sha256') == sha(args.precheck_root / 'out/actual_stage_receipt.json') and
            joint.get('original_stage_exit_sha256') == sha(args.precheck_root / 'natural_exit.json'), 'Joint/original receipt-exit mismatch')
    require(joint.get('whole_checkpoint_D_SHA_CRC_passed') is True and
            joint.get('original_other_CPU_receipt_sha256') and joint.get('captures'), 'Incomplete original preservation joint')
    require(joint.get('orders_sha256') == plan['orders_sha256'] and
            joint.get('official_row_ID_identity_sha256') == plan['official_train_dev_ID_identity_sha256'], 'Joint order/official identity mismatch')
    return parent

def stable_file(path):
    p = Path(path)
    require(p.is_file() and not p.is_symlink(), 'Nonregular/symlink capture input')
    a = p.stat()
    h = sha(p)
    b = p.stat()
    require((a.st_dev, a.st_ino, a.st_size, a.st_mtime_ns) == (b.st_dev, b.st_ino, b.st_size, b.st_mtime_ns),
            'File changed during original SHA capture')
    return {'original_path': str(p), 'bytes': a.st_size, 'sha256': h, 'inode': a.st_ino,
            'device': a.st_dev, 'mtime_ns': a.st_mtime_ns, 'stable_before_after': True}

def capture_association(base):
    base = Path(base)
    c = read(base / 'capture_receipt.json')
    ex = read(base / 'capture_actual_exit.json')
    m = read(base / 'member_manifest.json')
    require(c.get('status') == 'ACTUAL_PAIRED_CAPTURE_COMPLETE' and ex.get('exit_code') == 0 and
            ex.get('natural_wait_verified') is True and ex.get('child_pid') == c.get('pid') and
            ex.get('child_full_argv', [])[1:] == c.get('argv') and
            ex.get('original_receipt_sha256') == sha(base / 'capture_receipt.json'), 'Capture original natural0/argv association')
    zip_audit(base / 'snapshot.zip', c['snapshot_sha256'])
    with zipfile.ZipFile(base / 'snapshot.zip') as z:
        require(set(z.namelist()) == set(m['small_members']) | {'member_manifest.json'}, 'Capture exact member set mismatch')
        require(z.read('member_manifest.json') == (base / 'member_manifest.json').read_bytes(), 'Original capture manifest mismatch')
        for n, item in m['small_members'].items():
            safe_member(n)
            require(hashlib.sha256(z.read(n)).hexdigest() == item['sha256'] == sha(base / n), 'Complete ZIP/member/original SHA mismatch: ' + n)
    require(c['members'] == len(m['small_members']) + 1 and
            c['original_receipt_sha256'] == m['original_receipt_sha256'] and
            c['original_exit_sha256'] == m['original_exit_sha256'] and
            c['large_references_not_downloads']==m['large_references_not_downloads'], 'Capture member/receipt/exit/large-reference binding mismatch')
    return c, m
