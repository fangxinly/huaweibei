"""Preserve v1; tighten whole-process budget and execution evidence in new v2."""
from pathlib import Path
import datetime, hashlib, json, shutil

work = Path(__file__).resolve().parent
old = work/'correction_calibration_preparation_20261006T0354Z'
new = work/'correction_calibration_collection_v2_20261006T0414Z'
assert not new.exists()
new.mkdir()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
parent = json.loads((old/'same_teacher_collection_plan_v1.json').read_text(encoding='utf-8'))
assert sha(old/'same_teacher_collection_plan_v1.json') == 'efb55291d836c850ab8b34d375466940fa163fe274efc342d8e94928a1ce1903'
source = (old/'collect_group_teacher_fit_inner_v1.py').read_text(encoding='utf-8')
assert source.count('    torch.cuda.reset_peak_memory_stats()\n') == 1
source = source.replace('    torch.cuda.reset_peak_memory_stats()\n', '')
source = source.replace('    torch.use_deterministic_algorithms(True)\n',
    '    torch.use_deterministic_algorithms(True)\n'
    '    # Include original construction/loading peaks in the process budget.\n'
    '    torch.cuda.reset_peak_memory_stats()\n')
source = source.replace('    selected = plan[\'folds\'][args.fold]\n',
    '    assert sha(plan[\'capture_source_path\']) == plan[\'capture_source_sha256\']\n'
    '    selected = plan[\'folds\'][args.fold]\n')
source = source.replace("        assert auth['capture_source_sha256'] == plan['capture_source_sha256']\n",
    "        assert auth['capture_source_sha256'] == plan['capture_source_sha256']\n"
    "        independent = new_root/'precheck/independent_precheck_audit.json'\n"
    "        assert auth['precheck_independent_audit_sha256'] == sha(independent)\n"
    "        original_audit = json.loads(independent.read_text(encoding='utf-8'))\n"
    "        assert original_audit['status'] == 'ACTUAL_COLLECTION_PRECHECK_METADATA_INDEPENDENT_AUDIT_PASSED'\n"
    "        assert original_audit['original_gpu_receipt_sha256'] == sha(receipt)\n"
    "        assert original_audit['fold'] == args.fold\n")
source = source.replace("    core = model.dberta\n", "    assert torch.cuda.max_memory_allocated() <= plan['maximum_peak_allocated_bytes']\n    core = model.dberta\n")
source = source.replace("    model.to(author.DEVICE).eval().requires_grad_(False)\n",
    "    model.to(author.DEVICE).eval().requires_grad_(False)\n"
    "    assert torch.cuda.max_memory_allocated() <= plan['maximum_peak_allocated_bytes']\n")
source = source.replace("    receipt = {'passed': True,", "    produced = {path.name: {'sha256': sha(path), 'bytes': path.stat().st_size}\n"
    "                for path in out.glob('*_scalar_inputs.npz')}\n"
    "    total_payload = sum(path.stat().st_size for path in new_root.rglob('*') if path.is_file())\n"
    "    assert total_payload <= plan['maximum_new_payload_bytes']\n"
    "    assert time.monotonic()-started <= plan['maximum_seconds']\n"
    "    receipt = {'passed': True,")
source = source.replace("        'collection_outputs_written': args.phase == 'execute'}\n",
    "        'collection_outputs_written': args.phase == 'execute',\n"
    "        'whole_process_peak_includes_model_construction': True,\n"
    "        'actual_new_payload_bytes_before_receipt': total_payload, 'output_files': produced}\n")
target = work/'collect_group_teacher_fit_inner_v2.py'
assert not target.exists()
target.write_text(source.replace('Prepared locally, not GPU executed.',
                                'v2 whole-process budget; prepared locally, not GPU executed.'), encoding='utf-8')
shutil.copy2(target, new/target.name)
for name in parent['new_source_sha256']:
    if name != 'collect_group_teacher_fit_inner_v1.py':
        shutil.copy2(old/name, new/name)
for path in old.iterdir():
    if path.is_file() and path.suffix in ['.npy', '.npz']:
        shutil.copy2(path, new/path.name)
shutil.copy2(old/'calibration_video_role_plan_v1.json', new/'calibration_video_role_plan_v1.json')
plan = dict(parent)
plan['revision_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
plan['version'] = 2
plan['parent_plan_sha256'] = sha(old/'same_teacher_collection_plan_v1.json')
plan['collector_sha256'] = sha(target)
plan['collection_subdirectory'] = new.name
plan['capture_source_path'] = plan['base_root']+'/capture_soft_vector_v19.py'
plan['new_source_sha256'] = {name: value for name, value in parent['new_source_sha256'].items()
                             if name != 'collect_group_teacher_fit_inner_v1.py'}
plan['new_source_sha256'][target.name] = sha(target)
plan['whole_process_peak_includes_model_construction'] = True
plan['require_actual_phase_exit_receipts'] = True
plan['revision_reason'] = 'Local review found v1 reset peak after construction. v2 counts construction/loading, bounds total elapsed/payload, pins actual capture source, requires original independent precheck audit hash, and hashes generated arrays. v1 never deployed or executed; preserved.'
path = new/'same_teacher_collection_plan_v2.json'
path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
assert sha(old/'same_teacher_collection_plan_v1.json') == plan['parent_plan_sha256']
assert sha(old/'calibration_video_role_plan_v1.json') == sha(new/'calibration_video_role_plan_v1.json') == parent['role_plan_sha256']
out = {'status': 'LOCAL_COLLECTION_V2_RESOURCE_AND_RECEIPT_GATES_PREPARED_NOT_GPU_RUN',
       'utc': plan['revision_utc'], 'parent_plan_sha256': plan['parent_plan_sha256'],
       'v2_plan_sha256': sha(path), 'v2_collector_sha256': sha(target),
       'role_plan_unchanged_sha256': parent['role_plan_sha256'],
       'old_v1_preserved': True, 'old_v1_gpu_executed': False,
       'actual_v2_gpu_executed': False, 'actual_calibration_performed': False,
       'revision_reason': plan['revision_reason']}
(work.parent/'outputs/同折教师采集v2资源与原回执门控修订.json').write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(out))
