"""Prepared, separately gated driver for one original stopped composite epoch.

No usable audit plan is created here. Future original files, a human-provided
node/lease, runtime, fresh resources and a new audit token must be frozen first.
Import and provenance checks use only the standard library. This driver neither
trains nor qualifies continuous-versus-restored CPU/CUDA state.
"""
import argparse
import datetime as dt
import importlib.metadata
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys

from group5_release_transport_v1 import digest, write

STATUS = 'GROUP5_STOPPED_COMPOSITE_EPOCH_CPU_AUDIT_FROZEN'
REQUIRED_SOURCE = {
    'group5_stopped_epoch_CPU_driver_v1.py',
    'group5_composite_stopped_epoch_CPU_audit_v1.py',
    'group5_composite_epoch_resume_v2.py',
    'group5_release_transport_v1.py',
    'group5_test_selected_contract_v1.py',
    'fixed_flow_components_candidate.py',
    'encoder_adapter.py',
    'minimal_fixed_flow_v2.py',
    'legacy_flow_model.py',
    'finite_single_token_reader_v1.py',
}


def utc_value(value):
    result = dt.datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() != dt.timedelta(0):
        raise PermissionError('Actual observations and lease must use explicit UTC')
    return result


def positive(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise PermissionError('Measured budget/size must be finite and positive')
    return value


def gate(plan, origin, exited, restoration, members, lease, physical, now):
    """Pure metadata gate; synthetic fixtures cannot constitute actual evidence."""
    if plan.get('status') != STATUS or plan.get('execution_enabled') is not True:
        raise PermissionError('Preparation is not a frozen original CPU audit')
    method, fold = plan['method'], plan['fold']
    if method not in ('anchored_message20', 'old_fixed_A') or type(fold) is not int or not 0 <= fold < 5:
        raise PermissionError('Fixed composite method/fold required')
    if (origin.get('status') != 'GROUP5_COMPOSITE_TRAIN_FROZEN' or origin.get('execution_enabled') is not True or
            origin.get('method') != method or type(origin.get('fold')) is not int or origin['fold'] != fold or
            (origin.get('resume') is not None and origin.get('resume') is not False) or
            origin.get('old_task_weight_reuse') is not False):
        raise PermissionError('Only a bound fresh original same-fold composite is eligible')
    if type(exited.get('child')) is not int or exited['child'] <= 0 or type(exited.get('natural_exit')) is not int:
        raise PermissionError('Original process needs an observed natural exit')
    if type(plan['original_child']) is not int or exited['child'] != plan['original_child']:
        raise PermissionError('Original natural-exit child differs')
    if origin['new_once_token'] == plan['new_audit_once_token']:
        raise PermissionError('An original consumed training token cannot become an audit token')
    if (plan.get('no_fit_inference_score') is not True or plan.get('no_new_target_decode') is not True or
            plan.get('CPU_CUDA_recovery_qualification') is not False):
        raise PermissionError('Original-state audit scope must be explicit')
    if (plan['split_SHA'] != origin['split_SHA'] or plan['original_source_SHA'] != origin['source_SHA'] or
            {name: spec['SHA'] for name, spec in plan['original_assets'].items()} != origin['asset_SHA']):
        raise PermissionError('Bound original source/split differs')
    if not REQUIRED_SOURCE <= set(plan['audit_source_SHA']):
        raise PermissionError('Complete independent auditor dependency inventory required')
    if (restoration.get('status') != 'RELEASE_FULL_ORIGINAL_RESTORED_SHA_ZIP_VERIFIED' or
            restoration.get('all_member_SHA_CRC_unique_exact_set_passed') is not True or
            restoration.get('whole_SHA') != plan['original_archive_SHA'] or
            restoration.get('bytes') != plan['original_archive_bytes'] or
            restoration.get('member_manifest_SHA') != plan['member_manifest']['SHA']):
        raise PermissionError('Bound actual whole-original Release restoration absent')
    positive(plan['original_archive_bytes'])
    inventory = {}
    for row in members:
        name = row['name']; member = PurePosixPath(name)
        if (not name or member.is_absolute() or '..' in member.parts or '\\' in name or ':' in name or
                name in inventory or type(row['bytes']) is not int or row['bytes'] < 0):
            raise PermissionError('Original restoration member inventory is unsafe or ambiguous')
        inventory[name] = row
    for key, name in (('origin_plan', 'plan.json'), ('process_exit', 'wrapper_exit.json'),
                      ('checkpoint', 'out/complete_composite_full.pt')):
        spec = plan[key]
        if name not in inventory or inventory[name]['sha256'] != spec['SHA'] or inventory[name]['bytes'] != spec['bytes']:
            raise PermissionError('Original plan/exit/checkpoint is not in the restored whole original')
    if (lease.get('status') != 'HUMAN_PROVIDED_NEW_NODE_LEASE' or lease.get('human_provided') is not True or
            lease.get('GPU_UUID') != plan['GPU_UUID'] or lease.get('endpoint') != plan['endpoint'] or
            lease.get('lease_end_UTC') != plan['lease_end_UTC']):
        raise PermissionError('New human-provided node/actual lease identity is absent')
    if not 0 <= (now - utc_value(physical['actual_UTC'])).total_seconds() <= 300:
        raise PermissionError('Actual node/source/runtime/resource observation older than five minutes')
    if physical['GPU_UUID'] != plan['GPU_UUID'] or physical['compute'].strip() or not physical['fullargv'].strip():
        raise PermissionError('Actual UUID/full argv/empty compute gate failed')
    # On the origin node, the naturally exited child cannot still be present.
    if plan['GPU_UUID'] == origin['GPU_UUID']:
        pids = {int(line.split()[0]) for line in physical['fullargv'].splitlines()
                if line.split() and line.split()[0].isdigit()}
        if exited['child'] in pids:
            raise PermissionError('Original child still present; never stop it to enable recovery')
    if physical['python'] != plan['python'] or physical['runtime_versions'] != plan['runtime_versions']:
        raise PermissionError('Actual exact CPU runtime differs')
    if physical.get('all_bound_files_SHA_verified') is not True:
        raise PermissionError('Original files and complete audit-source SHA checks absent')
    if physical['available_RAM_bytes'] < 6 * 1024**3:
        raise PermissionError('Actual CPU available RAM below6GiB')
    if (physical['C_free_bytes'] < 200 * 1024**2 or physical['D_free_bytes'] < 40 * 1024**2 or
            not 0 <= (now - utc_value(physical['local_space_actual_UTC'])).total_seconds() <= 300):
        raise PermissionError('Fresh local preservation floors absent')
    if physical['remote_free_bytes'] < positive(plan['measured_remote_required_bytes']):
        raise PermissionError('Original audit/saving space insufficient')
    remaining = (utc_value(plan['lease_end_UTC']) - now).total_seconds()
    if remaining < positive(plan['measured_audit_seconds']) + max(7200, positive(plan['measured_saving_seconds'])):
        raise PermissionError('Measured original CPU audit plus at least two-hour saving exceeds lease')
    return True


def exact(spec):
    path = Path(spec['path'])
    if type(spec['bytes']) is not int or path.stat().st_size != spec['bytes'] or digest(path) != spec['SHA']:
        raise PermissionError('Bound original file bytes differ')
    return path


def inside(root, name):
    member = PurePosixPath(name)
    if member.is_absolute() or '..' in member.parts or '\\' in name or ':' in name:
        raise PermissionError('Bound source path escapes its capsule')
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise PermissionError('Bound source path escapes through a link')
    return path


def run(a):
    if digest(a.plan) != a.plan_sha:
        raise PermissionError('Frozen original CPU-audit plan bytes differ')
    plan = json.loads(a.plan.read_bytes())
    # Reject the inert template before file access, physical commands or a token.
    if plan.get('status') != STATUS or plan.get('execution_enabled') is not True:
        raise PermissionError('No executable original CPU-audit plan has been frozen')
    root, capsule = Path(plan['original_root']).resolve(), Path(__file__).resolve().parent
    output = Path(plan['audit_output_root']).resolve()
    token = Path(plan['new_audit_once_token']).resolve()
    if output.exists() or token.exists() or output == root or output.is_relative_to(root) or token.is_relative_to(root):
        raise PermissionError('Fresh audit output/token must not overwrite an original')
    if not output.parent.exists() or not token.parent.exists():
        raise PermissionError('Separately reviewed audit/token parent directories required')
    loaded = {key: json.loads(exact(plan[key]).read_bytes()) for key in
              ('origin_plan', 'process_exit', 'restoration', 'member_manifest', 'human_node_lease', 'local_space')}
    for key, rel in (('origin_plan', 'plan.json'), ('process_exit', 'wrapper_exit.json'),
                     ('checkpoint', 'out/complete_composite_full.pt')):
        if exact(plan[key]).resolve() != (root / rel).resolve():
            raise PermissionError('The selected original files are outside the exact original root')
    for rel, sha in plan['audit_source_SHA'].items():
        if digest(inside(capsule, rel)) != sha:
            raise PermissionError('Independent auditor source inventory differs')
    for rel, sha in plan['original_source_SHA'].items():
        if digest(inside(root / 'source', rel)) != sha:
            raise PermissionError('Frozen original scientific source inventory differs')
    for spec in plan['original_assets'].values():
        exact(spec)  # Stream SHA/size only; no pickle/array/target decode.
    if digest(root / 'source/split.json') != plan['split_SHA']:
        raise PermissionError('Bound original split differs')
    mem = Path('/proc/meminfo').read_text()
    local = loaded['local_space']
    physical = dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        GPU_UUID=subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip(),
        compute=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv,noheader'], text=True),
        fullargv=subprocess.check_output(['ps', '-eo', 'pid,ppid,args', '--width', '10000'], text=True),
        available_RAM_bytes=int(next(line.split()[1] for line in mem.splitlines() if line.startswith('MemAvailable:'))) * 1024,
        remote_free_bytes=shutil.disk_usage(output.parent).free, python=sys.executable,
        runtime_versions={name: importlib.metadata.version(name) for name in plan['runtime_versions']},
        C_free_bytes=local['C_free_bytes'], D_free_bytes=local['D_free_bytes'], local_space_actual_UTC=local['actual_UTC'],
        all_bound_files_SHA_verified=True)
    gate(plan, loaded['origin_plan'], loaded['process_exit'], loaded['restoration'], loaded['member_manifest'],
         loaded['human_node_lease'], physical, dt.datetime.now(dt.timezone.utc))
    # All provenance/resource checks precede the one new audit token and the
    # original auditor import. The token stays consumed even on a natural failure.
    with token.open('x', encoding='utf8') as stream:
        json.dump(dict(actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(), audit_plan_SHA=a.plan_sha,
                       method=plan['method'], fold=plan['fold'], original_child=plan['original_child']), stream)
    output.mkdir()
    write(output / 'fresh_physical.json', physical)
    from group5_composite_stopped_epoch_CPU_audit_v1 import audit_stopped_epoch
    result = audit_stopped_epoch(root, plan['origin_plan']['SHA'], plan['checkpoint']['SHA'],
                                 plan['checkpoint']['bytes'], plan['process_exit']['SHA'])
    if result['method'] != plan['method'] or result['fold'] != plan['fold']:
        raise PermissionError('Audited original method/fold differs from the new audit dispatch')
    result.update(audit_dispatch_plan_SHA=a.plan_sha, audit_source_SHA=plan['audit_source_SHA'],
                  independent_audit_not_recovery_dispatch=True)
    write(output / 'actual_original_CPU_audit.json', result)
    print('INDEPENDENT_ORIGINAL_STOPPED_EPOCH_CPU_AUDIT_PASS')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha', required=True)
    run(parser.parse_args())
