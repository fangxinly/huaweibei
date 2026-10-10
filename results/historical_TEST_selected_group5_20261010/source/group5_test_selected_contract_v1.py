"""New CV scope/parent/resource gates, inert until separately qualified execution.

Old two-method runners and original task weights cannot satisfy this contract.
This module itself performs no scientific, remote, or network execution.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path

METHODS = ('anchored_message20', 'old_fixed_A', 'factorized_aux', 'regression_aux', 'careflow')
PARENTS = {'anchored_message20': 'anchored_parent', 'old_fixed_A': 'old_A_teacher'}


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def validate_parent(method, fold, receipt, fold_spec):
    if method not in PARENTS:
        if receipt is not None:
            raise ValueError('Unexpected trained task parent for direct method')
        return
    if receipt is None:
        raise PermissionError('Fresh same-fold parent absent')
    if receipt['method'] != PARENTS[method] or receipt['fold'] != fold:
        raise PermissionError('Cross-fold/wrong-family parent forbidden')
    if receipt.get('historical_task_weights_used') or not receipt.get('public_pretraining_fresh_start'):
        raise PermissionError('Historical task teacher/parent forbidden')
    if receipt['fit_ids'] != fold_spec['row_ids']['fit'] or receipt['inner_ids'] != fold_spec['row_ids']['inner']:
        raise PermissionError('Parent training/selection scope differs')
    if receipt.get('outer_labels_decoded', True) or not receipt.get('GitHub_original_restoration_verified'):
        raise PermissionError('Parent is unqualified or accessed held-out targets')


def validate_dispatch(design, execution, physical, method, fold, now):
    """Fail closed on incomplete qualification, stale identity or expired lease."""
    if method not in METHODS or fold not in range(5):
        raise ValueError('Method/fold outside fixed roster')
    if execution['design_SHA'] != hashlib.sha256(design).hexdigest():
        raise PermissionError('Execution plan not bound to exact design')
    if not execution.get('execution_enabled') or execution.get('status') != 'GROUP5_NEW_EXECUTION_QUALIFIED':
        raise PermissionError('Preparation is not a training dispatch')
    required = ['source_synthetic_native_CPU_qualified', 'chunk_transport_restore_qualified',
                'new_once_available', 'no_historical_task_weight_reuse', 'all_stage_parent_scope_qualified']
    if not all(execution.get(k, False) for k in required):
        raise PermissionError('Missing new execution qualification')
    captured = dt.datetime.fromisoformat(physical['actual_UTC'])
    if not 0 <= (now-captured).total_seconds() <= 300:
        raise PermissionError('Fresh identity/resource observation required')
    if physical['GPU_UUID'] != execution['GPU_UUID'] or physical['compute_nonempty']:
        raise PermissionError('Physical GPU identity/compute gate failed')
    if not all(physical.get(k, False) for k in ['fullargv_captured', 'runtime_exact', 'assets_source_SHA_verified']):
        raise PermissionError('Native runtime/asset evidence absent')
    if physical['available_RAM_bytes'] < 6*1024**3:
        raise PermissionError('CPU available RAM below6GiB')
    if physical['C_free_bytes'] < 200*1024**2 or physical['D_free_bytes'] < 40*1024**2:
        raise PermissionError('Local saving floors failed')
    if physical['staging_free_bytes'] < execution['measured_staging_requirement_bytes']:
        raise PermissionError('Serial checkpoint/transfer/restore space absent')
    if physical['remote_free_bytes'] < execution['measured_remote_requirement_bytes']:
        raise PermissionError('Remote full recovery/original saving space absent')
    end = dt.datetime.fromisoformat(execution['conservative_lease_end_UTC'])
    if (end-now).total_seconds() < execution['measured_remaining_stage_queue_seconds'] + max(7200, execution['measured_saving_seconds']):
        raise PermissionError('Measured queue plus saving reserve exceeds lease')
    return True


def validate_release_parts(manifest, receipts):
    """Require an exact contiguous range set and remote digests, not just URLs."""
    wanted = manifest['parts']
    if len(wanted) != len(receipts) or not 0 < len(wanted) <= 1000:
        raise ValueError('Release part inventory differs')
    actual = {r['name']:r for r in receipts}
    if len(actual) != len(receipts) or set(actual) != {p['name'] for p in wanted}:
        raise ValueError('Missing/duplicate/unexpected Release parts')
    offset = 0
    for part in wanted:
        if part['offset'] != offset or not 0 < part['bytes'] < 2*1024**3:
            raise ValueError('Invalid contiguous sub2GiB part range')
        r = actual[part['name']]
        if r['state'] != 'uploaded' or r['bytes'] != part['bytes'] or r['digest'] != 'sha256:'+part['SHA']:
            raise PermissionError('GitHub part digest not verified')
        offset += part['bytes']
    if offset != manifest['bytes'] or not manifest.get('whole_SHA'):
        raise ValueError('Whole original archive inventory differs')
    return True
