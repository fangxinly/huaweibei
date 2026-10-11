"""Static capture coverage; excludes interpreter caches, never executes capture."""
from pathlib import Path
import ast, datetime, hashlib, json

work = Path(__file__).resolve().parent
capture = work/'capture_soft_vector_v19.py'
digest = hashlib.sha256(capture.read_bytes()).hexdigest()
assert digest == '6852df98cd6687c9875f56c908ef1bd57033b514e9d473ed22a24e1451f0c1ad'
tree = ast.parse(capture.read_text(encoding='utf-8'))
recursive = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'tree')
assert any(isinstance(node, ast.Attribute) and node.attr == 'rglob' for node in ast.walk(recursive))
assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'tree'
           and len(node.args) == 2 and isinstance(node.args[0], ast.Name) and node.args[0].id == 'teacher'
           for node in ast.walk(tree))
root = work/'correction_calibration_preparation_20261006T0354Z'
scientific = [path for path in root.rglob('*') if path.is_file() and '__pycache__' not in path.parts]
assert all(path.suffix in ['.py', '.json', '.npy', '.npz'] for path in scientific)
result = {'status': 'LOCAL_CAPTURE19_RECURSIVE_SUFFIX_COVERAGE_INSPECTED_NOT_REMOTE_CAPTURE',
    'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'capture_source_sha256': digest,
    'new_subroot': str(root), 'new_scientific_files': len(scientific),
    'files': [path.relative_to(root).as_posix() for path in scientific],
    'prior_check_exit1_retained': True,
    'prior_check_reason': 'Initial all-suffix assertion included generated __pycache__ directory as if a scientific file; scientific source hashes were correct.',
    'revision': 'Only enumerate files and explicitly exclude interpreter cache; no scientific source or capture code changed.',
    'actual_remote_source_fresh_sha_checked': False, 'actual_capture_executed': False,
    'actual_new_collector_process_inventory_checked': False,
    'scope': 'Static recursive file inclusion only; actual new argv and completion need explicit fresh inventory, CAPTURE_COMPLETE/exit0 and downloaded ZIP/member audit.'}
out = work.parent/'outputs/新校准采集子根capture19静态覆盖检查.json'
assert not out.exists()
out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
print(result['status'], result['new_scientific_files'])
