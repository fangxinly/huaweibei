"""Standard-library provenance gate for a separately qualified epoch recovery.

This gate does not qualify a model or manufacture a CPU-audit receipt. Original
partial-epoch CPU audit and CUDA recovery qualification remain separate work.
"""
import json
import pathlib
import shutil
from group5_release_transport_v1 import digest

P = pathlib.Path
METHODS = ('anchored_message20', 'old_fixed_A')
CPU_STATUS = 'INDEPENDENT_CPU_ORIGINAL_COMPOSITE_EPOCH_STATE_AUDIT_PASS'


def exact_json(spec):
    path = P(spec['path'])
    if digest(path) != spec['SHA']:
        raise PermissionError('Original recovery evidence SHA differs')
    return json.loads(path.read_bytes())


def qualify(plan, method, fold, stage):
    """Validate original metadata before any new token or scientific import."""
    spec = plan.get('resume')
    if spec is None or spec is False:
        return None
    if not isinstance(spec, dict) or stage != 'train' or method not in METHODS:
        raise PermissionError('Only a separately frozen composite training recovery is allowed')
    required = ('complete_epoch_recovery_qualified', 'full_Adam_RNG_recovery_qualified',
                'CUDA_recovery_qualified')
    if any(spec.get(key) is not True for key in required):
        raise PermissionError('Actual full-state and CUDA recovery qualification is absent')
    origin = exact_json(spec['origin_plan'])
    cpu = exact_json(spec['CPU_audit'])
    restored = exact_json(spec['restoration'])
    members = exact_json(spec['member_manifest'])
    exited = exact_json(spec['process_exit'])
    cost = exact_json(spec['cost_accounting'])
    qualification = exact_json(spec['state_recovery_qualification'])
    if origin.get('status') != 'GROUP5_COMPOSITE_TRAIN_FROZEN':
        raise PermissionError('The origin was not a frozen fresh composite training stage')
    if origin.get('resume') is not None and origin.get('resume') is not False:
        raise PermissionError('Recovery ancestry beyond one fresh origin is not qualified by this version')
    for key in ('source_SHA', 'asset_SHA', 'split_SHA', 'orders_SHA', 'updates', 'task_seed'):
        if origin[key] != plan[key]:
            raise PermissionError('Original recovery source/data/budget identity differs: ' + key)
    if origin['method'] != method or origin['fold'] != fold or plan['method'] != method or plan['fold'] != fold:
        raise PermissionError('Cross-family or cross-fold recovery is forbidden')
    if origin['old_task_weight_reuse'] or plan['old_task_weight_reuse']:
        raise PermissionError('Historical task-weight scope is forbidden')
    if plan['new_once_token'] == origin['new_once_token']:
        raise PermissionError('A consumed original token cannot be used for recovery')
    if (qualification.get('status') != 'ORIGINAL_COMPOSITE_COMPLETE_STATE_CPU_CUDA_RECOVERY_QUALIFICATION_PASS' or
            qualification.get('method') != method or qualification.get('fold') != fold or
            qualification.get('source_SHA') != plan['source_SHA'] or
            qualification.get('parent_checkpoint_SHA') != plan['parent_checkpoint']['SHA'] or
            qualification.get('cache_SHA') != plan['cache_reference']['files_SHA']):
        raise PermissionError('Bound original CPU/CUDA complete-state recovery evidence is absent')
    checks = ('current_model', 'selected_model', 'all_Adam', 'Python_NumPy_Torch_CUDA_RNG',
              'FIT_orders_statistics', 'next_update_matches_continuous')
    if any(qualification.get('checks', {}).get(key) is not True for key in checks):
        raise PermissionError('Actual continuous-versus-recovered full-state checks are incomplete')
    # Partial originals must be observed stopped; a healthy child is never stopped
    # to make this recovery gate pass. Boolean False must not masquerade as exit0.
    if type(exited.get('natural_exit')) is not int or exited.get('child') != spec['original_child']:
        raise PermissionError('The original child has no bound observed natural exit')
    if cpu.get('status') != CPU_STATUS or cpu.get('original_child') != spec['original_child']:
        raise PermissionError('Original nonfinal-epoch independent CPU qualification is absent')
    if cpu.get('method') != method or cpu.get('fold') != fold or cpu.get('origin_plan_SHA') != spec['origin_plan']['SHA']:
        raise PermissionError('Original checkpoint CPU identity differs')
    checkpoint = spec['checkpoint']
    file = P(checkpoint['path'])
    if file.stat().st_size != checkpoint['bytes'] or digest(file) != checkpoint['SHA'] or cpu['checkpoint_SHA'] != checkpoint['SHA']:
        raise PermissionError('Original full checkpoint bytes differ')
    if cpu.get('source_SHA') != plan['source_SHA'] or cpu.get('split_SHA') != plan['split_SHA']:
        raise PermissionError('Original full-state CPU source/split differs')
    if cpu.get('outer_labels_decoded') is not False or (method == 'anchored_message20' and cpu.get('inner_labels_decoded') is not False):
        raise PermissionError('Held-out label scope was violated')
    if (restored.get('status') != 'RELEASE_FULL_ORIGINAL_RESTORED_SHA_ZIP_VERIFIED' or
            restored.get('all_member_SHA_CRC_unique_exact_set_passed') is not True or
            restored.get('whole_SHA') != spec['original_archive_SHA'] or
            restored.get('bytes') != spec['original_archive_bytes']):
        raise PermissionError('The original whole Release restoration is absent')
    if restored.get('member_manifest_SHA') != spec['member_manifest']['SHA']:
        raise PermissionError('The restored original manifest is not bound to the restoration receipt')
    manifest = {row['name']: row for row in members}
    if len(manifest) != len(members):
        raise PermissionError('Duplicate original member identity')
    for member, sha, size in ((spec['checkpoint_member'], checkpoint['SHA'], checkpoint['bytes']),
                              ('plan.json', spec['origin_plan']['SHA'], P(spec['origin_plan']['path']).stat().st_size)):
        if member not in manifest or manifest[member]['sha256'] != sha or manifest[member]['bytes'] != size:
            raise PermissionError('Checkpoint/plan is not bound to the restored original')
    if (digest(spec['FIT_steps']['path']) != spec['FIT_steps']['SHA'] or
            manifest['out/FIT_steps.jsonl']['sha256'] != spec['FIT_steps']['SHA'] or
            manifest['out/FIT_steps.jsonl']['bytes'] != P(spec['FIT_steps']['path']).stat().st_size):
        raise PermissionError('Original partial FIT log is not preserved')
    steps = [json.loads(line) for line in P(spec['FIT_steps']['path']).read_text().splitlines()]
    offset = origin.get('resume', {}).get('restored_updates', 0) if isinstance(origin.get('resume'), dict) else 0
    if any(type(row.get('updates')) is not int or row['updates'] != offset + index
           for index, row in enumerate(steps, 1)):
        raise PermissionError('Observed original partial update log is noncontiguous')
    completed = cpu.get('completed_epochs')
    total = 20 if method == 'anchored_message20' else 100
    if type(plan['updates']) is not int or plan['updates'] <= 0 or plan['updates'] % total:
        raise PermissionError('The original full-epoch update budget is invalid')
    per = plan['updates'] // total
    if type(completed) is not int or not 0 < completed < total or cpu.get('updates') != completed * per:
        raise PermissionError('Only an actual nonfinal durable complete epoch can recover')
    observed = offset + len(steps)
    extra = observed - cpu['updates']
    if extra < 0 or cost.get('origin_FIT_log_SHA') != spec['FIT_steps']['SHA'] or cost.get('logical_updates_restored') != cpu['updates']:
        raise PermissionError('Original checkpoint and observed cost differ')
    if cost.get('observed_extra_completed_updates') != extra or cost.get('unlogged_inflight_update_possible') is not True:
        raise PermissionError('Discarded/replayed and possible unlogged in-flight cost must be disclosed')
    if (cpu.get('parent_checkpoint_SHA') != plan['parent_checkpoint']['SHA'] or
            origin['parent_checkpoint']['SHA'] != plan['parent_checkpoint']['SHA'] or
            cpu.get('cache_SHA') != plan['cache_reference']['files_SHA']):
        raise PermissionError('Original same-fold parent/cache recovery identity differs')
    if spec.get('restored_updates') != cpu['updates']:
        raise PermissionError('Frozen restored update budget differs')
    return dict(origin_plan_SHA=spec['origin_plan']['SHA'], restored_updates=cpu['updates'],
                completed_epochs=completed, observed_extra_completed_updates=extra,
                unlogged_inflight_update_possible=True, cost_accounting=cost)


def copy_completed_epoch_evidence(spec, history, method, destination):
    """Copy immutable bytes, without decoding arrays or rerunning any selection."""
    root = P(spec['origin_root']).resolve()
    destination = P(destination).resolve()
    manifest = {row['name']: row for row in exact_json(spec['member_manifest'])}
    for epoch, row in enumerate(history, 1):
        if method != 'old_fixed_A':
            break
        name = f'INNER_epoch{epoch:03d}'
        source = (root / 'out' / (name + '.npz')).resolve()
        if not source.is_relative_to(root) or digest(source) != row['prediction_SHA']:
            raise PermissionError('Original frozen INNER evidence differs')
        for suffix in ('.npz', '_freeze.json'):
            original = root / 'out' / (name + suffix)
            entry = manifest['out/' + name + suffix]
            if digest(original) != entry['sha256'] or original.stat().st_size != entry['bytes']:
                raise PermissionError('Original INNER evidence is not bound to the restored manifest')
            if suffix == '_freeze.json':
                frozen = json.loads(original.read_bytes())
                if frozen['SHA'] != row['prediction_SHA'] or frozen['state_SHA'] != row['state_SHA']:
                    raise PermissionError('Original INNER freeze and history disagree')
            target = destination / (name + suffix)
            if target.exists():
                raise FileExistsError('Fresh recovery evidence destination required')
            shutil.copyfile(original, target)
            if digest(target) != digest(original):
                raise ValueError('Recovery evidence copy differs')
    provenance = destination / 'recovery_origin'
    provenance.mkdir()
    for key, name in (('FIT_steps', 'original_FIT_steps.jsonl'), ('cost_accounting', 'original_cost_accounting.json')):
        source = P(spec[key]['path'])
        if digest(source) != spec[key]['SHA']:
            raise PermissionError('Original recovery cost evidence differs')
        shutil.copyfile(source, provenance / name)
        if digest(provenance / name) != spec[key]['SHA']:
            raise ValueError('Recovery cost evidence copy differs')
