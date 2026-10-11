"""Local preparation gates only; never create a purported real GPU receipt."""
from pathlib import Path
import ast
import datetime
import hashlib
import importlib.util
import json
import shutil


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


work = Path(__file__).resolve().parent
new = work / 'correction_calibration_collection_v2_20261006T0414Z'
old = work / 'correction_calibration_preparation_20261006T0354Z'
original = Path('D:/CodexBackups/selective_flow_20261003_1105/group_teacher_completed_20261006/a')
for name in ('run_same_teacher_collection_v2.py', 'audit_same_teacher_collection_v2.py'):
    assert not (new / name).exists()
    shutil.copy2(work / name, new / name)
plan = json.loads((new / 'same_teacher_collection_plan_v2.json').read_text(encoding='utf-8'))
assert sha(new / 'same_teacher_collection_plan_v2.json') == '57b740fed4ee9ac2d9736956f2e6cbc3b9dcd30b102610abe49f563a493738f7'
assert sha(old / 'same_teacher_collection_plan_v1.json') == plan['parent_plan_sha256']
assert sha(new / 'calibration_video_role_plan_v1.json') == sha(old / 'calibration_video_role_plan_v1.json') == plan['role_plan_sha256']
collector = new / 'collect_group_teacher_fit_inner_v2.py'
assert sha(collector) == plan['collector_sha256']
tree = ast.parse(collector.read_text(encoding='utf-8'))
calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
peak_lines = [node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'reset_peak_memory_stats']
construction_lines = [node.lineno for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'prep_for_training']
assert len(peak_lines) == len(construction_lines) == 1 and peak_lines[0] < construction_lines[0]
assert not any(isinstance(node.func, ast.Attribute) and node.func.attr in ('step', 'backward') for node in calls)
for name in plan['new_source_sha256']:
    assert sha(new / name) == plan['new_source_sha256'][name]
    ast.parse((new / name).read_text(encoding='utf-8'))
spec = importlib.util.spec_from_file_location('new_collection_audit', work / 'audit_same_teacher_collection_v2.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)
rejections = []
old_completed = json.loads((original / 'run_v1/completion.json').read_text(encoding='utf-8'))
try:
    audit.audit_receipt_contract(old_completed, {}, {}, plan, plan['folds'][0], 'precheck')
except (AssertionError, KeyError):
    rejections.append('OLD_100_TEACHER_COMPLETION_IS_NOT_NEW_COLLECTION_PRECHECK')
else:
    raise AssertionError('Old teacher completion was accepted as new GPU evidence')
for phase in ('precheck', 'execute'):
    assert not (new / phase).exists() and not (new / (phase + '_exit.json')).exists()
    try:
        audit.audit(new, original, 0, phase)
    except FileNotFoundError:
        rejections.append('NO_ACTUAL_' + phase.upper() + '_RECEIPT_OR_EXIT_NOT_ACCEPTED')
    else:
        raise AssertionError('Absent new experiment was accepted')
bundle = {'status': 'LOCAL_COLLECTION_V2_AUDIT_BUNDLE_PREPARED_NOT_DEPLOYED',
          'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'plan_sha256': sha(new / 'same_teacher_collection_plan_v2.json'),
          'parent_plan_sha256': plan['parent_plan_sha256'],
          'role_plan_sha256': plan['role_plan_sha256'],
          'sources': {name: sha(new / name) for name in ('collect_group_teacher_fit_inner_v2.py',
                     'run_same_teacher_collection_v2.py', 'audit_same_teacher_collection_v2.py')},
          'actual_precheck_executed': False, 'actual_collection_executed': False,
          'actual_calibration_fitted': False}
(new / 'collection_v2_audit_bundle_v1.json').write_text(json.dumps(bundle, indent=2), encoding='utf-8')
proof = {**bundle, 'status': 'LOCAL_COLLECTION_V2_AST_AND_NEGATIVE_GATES_PASSED_NOT_GPU_VALIDATED',
         'whole_process_peak_reset_before_model_construction': True,
         'no_optimizer_steps_or_backward_in_collector_ast': True,
         'negative_gates': rejections,
         'exit_wrapper_does_not_terminate_or_timeout_healthy_child': True,
         'audit_is_numpy_metadata_and_arrays_not_cpu_full_model_forward': True,
         'original_v1_and_roles_unchanged': True,
         'no_real_labels_or_new_gpu_arrays_read': True}
(work.parent / 'outputs/同折教师采集v2原回执数组审核本地准备核验.json').write_text(json.dumps(proof, indent=2), encoding='utf-8')
print(json.dumps({'status': proof['status'], 'negative_gates': rejections, 'sources': bundle['sources']}))
