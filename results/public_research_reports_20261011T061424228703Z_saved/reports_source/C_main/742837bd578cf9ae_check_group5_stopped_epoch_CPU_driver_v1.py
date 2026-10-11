"""New standard-library adversarial gates; no real checkpoint or audit token."""
import ast
import copy
import datetime as dt
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from group5_stopped_epoch_CPU_driver_v1 import gate, inside, run, STATUS, REQUIRED_SOURCE
from group5_release_transport_v1 import digest, write

WORK = Path(__file__).resolve().parent
NOW = dt.datetime(2030, 1, 1, tzinfo=dt.timezone.utc)  # Explicit synthetic time.


def fixture():
    sha = 'a' * 64
    origin = dict(status='GROUP5_COMPOSITE_TRAIN_FROZEN', execution_enabled=True,
        method='old_fixed_A', fold=0, resume=None, old_task_weight_reuse=False,
        new_once_token='/synthetic/original_train.once', split_SHA=sha,
        source_SHA={'synthetic.py': sha}, asset_SHA={'synthetic-pickle': sha}, GPU_UUID='synthetic-original-node',
        runtime_versions={'torch':'synthetic-version','numpy':'synthetic-version'})
    files = {key: dict(path='/synthetic/' + key, SHA=sha, bytes=4)
             for key in ('origin_plan', 'process_exit', 'checkpoint', 'member_manifest')}
    plan = dict(status=STATUS, execution_enabled=True, method=origin['method'], fold=0,
        original_child=123, new_audit_once_token='/synthetic/new_audit.once',
        no_fit_inference_score=True, no_new_target_decode=True, CPU_CUDA_recovery_qualification=False,
        split_SHA=sha, original_source_SHA=origin['source_SHA'], original_assets={'synthetic-pickle': files['checkpoint']},
        audit_source_SHA={name: sha for name in REQUIRED_SOURCE}, original_archive_SHA=sha,
        original_archive_bytes=64, GPU_UUID='synthetic-new-node', endpoint='synthetic.invalid:1',
        python='/synthetic/python', runtime_versions=origin['runtime_versions'],
        lease_end_UTC=(NOW + dt.timedelta(hours=24)).isoformat(), measured_audit_seconds=600,
        measured_saving_seconds=7200, measured_remote_required_bytes=100, **files)
    exited = dict(child=123, natural_exit=1)  # Naturally failed nonfinal original is eligible.
    restoration = dict(status='RELEASE_FULL_ORIGINAL_RESTORED_SHA_ZIP_VERIFIED',
        all_member_SHA_CRC_unique_exact_set_passed=True, whole_SHA=sha, bytes=64, member_manifest_SHA=sha)
    members = [dict(name=name, bytes=4, sha256=sha) for name in
               ('plan.json', 'wrapper_exit.json', 'out/complete_composite_full.pt')]
    lease = dict(status='HUMAN_PROVIDED_NEW_NODE_LEASE', human_provided=True,
        GPU_UUID=plan['GPU_UUID'], endpoint=plan['endpoint'], lease_end_UTC=plan['lease_end_UTC'])
    physical = dict(actual_UTC=NOW.isoformat(), GPU_UUID=plan['GPU_UUID'], compute='',
        fullargv='  9  1 synthetic-only', python=plan['python'], runtime_versions=plan['runtime_versions'],
        all_bound_files_SHA_verified=True, available_RAM_bytes=6*1024**3,
        C_free_bytes=200*1024**2, D_free_bytes=40*1024**2,
        local_space_actual_UTC=NOW.isoformat(), remote_free_bytes=100)
    return [plan, origin, exited, restoration, members, lease, physical, NOW]


def main():
    results = []
    assert gate(*fixture()) is True
    results.append(dict(case='synthetic_only_metadata_positive', passed=True, real_evidence=False))
    cases = [
        ('preparation_not_execution', lambda a: a[0].update(status='PREPARED', execution_enabled=False)),
        ('string_execution_flag', lambda a: a[0].update(execution_enabled='true')),
        ('boolean_fold', lambda a: a[0].update(fold=False)),
        ('direct_method', lambda a: a[0].update(method='careflow')),
        ('different_origin_fold', lambda a: a[1].update(fold=1)),
        ('origin_unfrozen', lambda a: a[1].update(execution_enabled=False)),
        ('resume_chain', lambda a: a[1].update(resume={'restored_updates':1})),
        ('numeric_false_resume', lambda a: a[1].update(resume=0)),
        ('historical_task_weights', lambda a: a[1].update(old_task_weight_reuse=True)),
        ('boolean_exit', lambda a: a[2].update(natural_exit=False)),
        ('wrong_original_child', lambda a: a[2].update(child=124)),
        ('consumed_train_token', lambda a: a[0].update(new_audit_once_token=a[1]['new_once_token'])),
        ('recovery_claim', lambda a: a[0].update(CPU_CUDA_recovery_qualification=True)),
        ('target_decode_scope', lambda a: a[0].update(no_new_target_decode=False)),
        ('changed_original_source', lambda a: a[0].update(original_source_SHA={'different':'b'*64})),
        ('missing_transitive_source', lambda a: a[0]['audit_source_SHA'].pop('legacy_flow_model.py')),
        ('missing_scientific_runtime_inventory', lambda a: (a[0].update(runtime_versions={}),a[1].update(runtime_versions={}))),
        ('different_original_runtime', lambda a: a[1].update(runtime_versions={'torch':'other','numpy':'other'})),
        ('changed_asset_SHA', lambda a: a[1].update(asset_SHA={'synthetic-pickle':'b'*64})),
        ('no_whole_restoration', lambda a: a[3].update(all_member_SHA_CRC_unique_exact_set_passed=False)),
        ('wrong_archive_SHA', lambda a: a[3].update(whole_SHA='b'*64)),
        ('wrong_manifest_binding', lambda a: a[3].update(member_manifest_SHA='b'*64)),
        ('duplicate_member', lambda a: a[4].append(copy.deepcopy(a[4][0]))),
        ('member_escape', lambda a: a[4].append(dict(name='../escape',bytes=0,sha256='a'*64))),
        ('checkpoint_not_in_archive', lambda a: a[4].pop()),
        ('checkpoint_size_mismatch', lambda a: a[4][-1].update(bytes=3)),
        ('no_human_lease', lambda a: a[5].update(human_provided=False)),
        ('lease_node_mismatch', lambda a: a[5].update(GPU_UUID='other')),
        ('lease_end_mismatch', lambda a: a[5].update(lease_end_UTC=NOW.isoformat())),
        ('stale_actual_capture', lambda a: a[6].update(actual_UTC=(NOW-dt.timedelta(seconds=301)).isoformat())),
        ('future_actual_capture', lambda a: a[6].update(actual_UTC=(NOW+dt.timedelta(seconds=1)).isoformat())),
        ('wrong_physical_UUID', lambda a: a[6].update(GPU_UUID='other')),
        ('nonempty_compute', lambda a: a[6].update(compute='synthetic-running-job')),
        ('missing_fullargv', lambda a: a[6].update(fullargv='')),
        ('runtime_mismatch', lambda a: a[6].update(runtime_versions={'torch':'other'})),
        ('files_not_verified', lambda a: a[6].update(all_bound_files_SHA_verified=False)),
        ('RAM_floor', lambda a: a[6].update(available_RAM_bytes=6*1024**3-1)),
        ('C_floor', lambda a: a[6].update(C_free_bytes=200*1024**2-1)),
        ('D_floor', lambda a: a[6].update(D_free_bytes=40*1024**2-1)),
        ('stale_local_space', lambda a: a[6].update(local_space_actual_UTC=(NOW-dt.timedelta(seconds=301)).isoformat())),
        ('remote_space', lambda a: a[6].update(remote_free_bytes=99)),
        ('boolean_budget', lambda a: a[0].update(measured_audit_seconds=True)),
        ('nonfinite_budget', lambda a: a[0].update(measured_audit_seconds=float('nan'))),
        ('insufficient_lease', lambda a: (a[0].update(lease_end_UTC=(NOW+dt.timedelta(seconds=7799)).isoformat()),
                                         a[5].update(lease_end_UTC=a[0]['lease_end_UTC']))),
    ]
    for name, edit in cases:
        values = copy.deepcopy(fixture()); edit(values)
        try: gate(*values)
        except (PermissionError, ValueError): pass
        else: raise AssertionError('Unsafe driver metadata accepted: ' + name)
        results.append(dict(case=name, passed=True))
    values=fixture(); values[1]['GPU_UUID']=values[0]['GPU_UUID']; values[6]['fullargv']='123 1 synthetic-original-still-present'
    try: gate(*values)
    except PermissionError: pass
    else: raise AssertionError('Original live child accepted')
    results.append(dict(case='original_live_child', passed=True))
    root = WORK / ('group5_CPU_driver_synthetic_' + dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    root.mkdir()
    template = root/'inert_plan.json';write(template, dict(status='GROUP5_STOPPED_EPOCH_CPU_AUDIT_PREPARED', execution_enabled=False))
    try: run(SimpleNamespace(plan=template, plan_sha=digest(template)))
    except PermissionError: pass
    else: raise AssertionError('Inert plan reached physical commands/once/scientific import')
    results.append(dict(case='actual_inert_entry_rejected_before_physical_commands_or_token',passed=True))
    for unsafe in ('../escape.py', '/absolute.py', 'C:/escape.py', 'escape\\module.py'):
        try: inside(root, unsafe)
        except PermissionError: pass
        else: raise AssertionError('Unsafe source path accepted')
        results.append(dict(case='source_path_'+unsafe,passed=True))
    tree=ast.parse((WORK/'group5_stopped_epoch_CPU_driver_v1.py').read_text(encoding='utf8'))
    entry=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    gate_line=next(n.lineno for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='gate')
    once_line=next(n.lineno for n in ast.walk(entry) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='open')
    audit_line=next(n.lineno for n in ast.walk(entry) if isinstance(n,ast.ImportFrom) and n.module=='group5_composite_stopped_epoch_CPU_audit_v1')
    assert gate_line < once_line < audit_line
    assert not any(name in sys.modules for name in ('numpy','torch'))
    report=dict(status='PREPARED_STOPPED_EPOCH_CPU_DRIVER_SYNTHETIC_GATE_PASS', actual_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
        passed_cases=len(results), cases=results, source_SHA=digest(WORK/'group5_stopped_epoch_CPU_driver_v1.py'),
        scope='Standard-library synthetic metadata plus inert entry rejection; no original model/tensor audit',
        source_gates_precede_once_and_auditor_import=True, actual_audit_plan_frozen=False,
        actual_new_node_lease_received=False, scientific_imports=False, task_array_or_target_decode=False,
        real_original_checkpoint_CPU_audit=False, CPU_CUDA_recovery_qualification=False, new_actual_audit_once_consumed=False)
    write(root/'actual_synthetic_report.json',report)
    write(WORK/'group5_stopped_epoch_CPU_driver_check_pointer.json',dict(root=str(root),report=str(root/'actual_synthetic_report.json'),SHA=digest(root/'actual_synthetic_report.json')))
    print(json.dumps(dict(status=report['status'],passed_cases=len(results),report=str(root/'actual_synthetic_report.json'))))


if __name__ == '__main__': main()
