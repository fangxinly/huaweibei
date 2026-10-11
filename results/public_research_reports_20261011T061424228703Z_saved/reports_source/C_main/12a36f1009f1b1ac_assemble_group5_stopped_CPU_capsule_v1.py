"""Assemble exact, isolated CPU-audit dependencies without scientific execution."""
import argparse
import ast
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tomllib
import zipfile

BASE = Path(__file__).resolve().parent.parent
WORK = BASE / 'work'
sys.path.insert(0, str(WORK))
from group5_release_transport_v1 import digest, write, seal, verify_zip

POINTER = WORK / 'group5_stopped_CPU_capsule_pointer.json'
CONTROL = Path('D:/CodexBackups/selective_flow_20261003_1105/candidate_posttrain_lowC_20261009T005229Z')
LOCAL = ('group5_stopped_epoch_CPU_driver_v1.py',
         'group5_composite_stopped_epoch_CPU_audit_v1.py',
         'group5_composite_epoch_resume_v2.py',
         'group5_release_transport_v1.py', 'group5_test_selected_contract_v1.py')
SCIENTIFIC = ('fixed_flow_components_candidate.py', 'encoder_adapter.py',
              'minimal_fixed_flow_v2.py', 'legacy_flow_model.py',
              'finite_single_token_reader_v1.py')
PROBE = '''import importlib, importlib.abc, json, pathlib, sys
source = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(source))
class ScientificImportForbidden(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy'}:
            raise RuntimeError('Scientific import is forbidden in this standard-library probe')
sys.meta_path.insert(0, ScientificImportForbidden())
names = ['group5_stopped_epoch_CPU_driver_v1',
         'group5_composite_stopped_epoch_CPU_audit_v1',
         'group5_composite_epoch_resume_v2', 'group5_test_selected_contract_v1']
for name in names:
    module = importlib.import_module(name)
    assert pathlib.Path(module.__file__).resolve().parent == source
assert not {'numpy', 'torch'} & set(sys.modules)
assert not list(source.glob('*.once'))
print(json.dumps({'isolated_inert_module_imports': names, 'scientific_imports': False,
                  'original_checkpoint_audit': False, 'task_arrays_or_targets_loaded': False,
                  'training_prediction_or_score': False, 'actual_once_consumed': False}))
'''


def dependency_inventory(source):
    """CPU entry closure; transport publication functions are explicitly excluded."""
    local = {p.stem for p in source.glob('*.py')}
    records = []
    excluded = []
    for path in sorted(source.glob('*.py')):
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        nodes = []
        for top in tree.body:
            if path.name == 'group5_release_transport_v1.py' and isinstance(top, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if top.name not in {'digest', 'blocks', 'write', 'utc'}:
                    excluded.append({'source': path.name, 'function': top.name,
                                     'reason': 'Not called by the frozen CPU audit entry; publication/restoration CLI is outside this capsule scope.'})
                    continue
            nodes.extend(ast.walk(top))
        imports = []
        for node in nodes:
            if isinstance(node, ast.Import):
                imports.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    raise ValueError('Relative import needs a separately reviewed package layout')
                imports.append(node.module or '')
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in {'eval', 'exec', '__import__'}:
                    raise ValueError('Uninventoried dynamic execution/import')
                if isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module':
                    raise ValueError('Dynamic module import needs separate explicit inventory')
        dependencies = []
        for name in sorted(set(imports)):
            root = name.split('.')[0]
            if root in local:
                kind = 'exact_capsule_source'
            elif root in sys.stdlib_module_names:
                kind = 'standard_library'
            elif root in {'torch', 'numpy'}:
                kind = 'scientific_runtime_pending_original_version_binding'
            else:
                raise ValueError('Unresolved CPU audit dependency: ' + name)
            dependencies.append({'module': name, 'kind': kind})
        records.append({'name': path.name, 'SHA': digest(path), 'bytes': path.stat().st_size,
                        'dependencies': dependencies})
    assert {r['name'] for r in records} == set(LOCAL + SCIENTIFIC)
    return {'source_files': records, 'excluded_deferred_transport_functions': excluded,
            'closure_scope': 'Stopped-epoch CPU audit entry and its imported scientific definitions only; transport publication/restore CLI is excluded.',
            'source_AST_dependency_closure_verified': True,
            'scientific_package_runtime_execution_qualified': False}


def assemble():
    stamp = dt.datetime.now(dt.timezone.utc)
    root = BASE / 'outputs' / ('group5_stopped_CPU_capsule_prepared_' + stamp.strftime('%Y%m%dT%H%M%SZ'))
    root.mkdir()
    payload = root / 'payload'
    source = payload / 'source'
    source.mkdir(parents=True)
    prior = json.loads((WORK / 'group5_epoch_CPU_driver_source_pointer.json').read_bytes())
    pipeline = json.loads((WORK / 'group5_pipeline_pointer.json').read_bytes())
    original = Path(pipeline['proof']['archive'])
    assert digest(original) == pipeline['proof']['archive_SHA']
    previous_source = Path(prior['root']) / 'payload/source'
    for name in LOCAL:
        assert digest(previous_source / name) == digest(WORK / name)
        shutil.copyfile(previous_source / name, source / name)
    origins = []
    with zipfile.ZipFile(original) as archive:
        members = json.loads(archive.read('member_manifest.json'))
        assert len(archive.namelist()) == len(set(archive.namelist()))
        assert set(archive.namelist()) == {row['name'] for row in members} | {'member_manifest.json'}
        inventory = {row['name']: row for row in members}
        for name in SCIENTIFIC:
            same = []
            for family in ('anchored_parent', 'old_A_teacher'):
                rel = family + '/' + name
                record = inventory[rel]
                data = archive.read(rel)  # CRC checked for this selected source member.
                assert len(data) == record['bytes'] and hashlib.sha256(data).hexdigest() == record['sha256']
                assert digest(Path(pipeline['source_root']) / rel) == record['sha256']
                same.append(data)
                origins.append({'capsule_name': name, 'family': family, 'original_member': rel,
                                'SHA': record['sha256'], 'bytes': record['bytes']})
            assert same[0] == same[1], 'Different family bytes require separate capsules'
            (source / name).write_bytes(same[0])
    inventory = dependency_inventory(source)
    write(payload / 'audit_dependency_inventory.json', inventory)
    write(payload / 'exact_original_source_origins.json', {
        'previous_driver_source': {'root': prior['root'], 'proof': prior['D_proof'], 'github': prior['github']},
        'original_pipeline': pipeline, 'selected_member_source_bindings': origins,
        'whole_pipeline_SHA_bound': True, 'selected_members_CRC_SHA_exact_bytes_verified': True,
        'no_original_archive_recompression_or_republication': True})
    probe = payload / 'isolated_standard_library_probe.py'
    probe.write_text(PROBE, encoding='utf8', newline='\n')
    process = subprocess.run([sys.executable, '-I', '-S', '-B', '-X', 'utf8', str(probe), str(source)],
                             cwd=root, capture_output=True, text=True, timeout=30)
    observation = {'actual_observed_completion_UTC': dt.datetime.now(dt.timezone.utc).isoformat(),
                   'natural_exit': process.returncode, 'stdout': process.stdout, 'stderr': process.stderr,
                   'isolated_without_site_packages': True, 'probe_SHA': digest(probe),
                   'inventory_SHA': digest(payload / 'audit_dependency_inventory.json')}
    write(payload / 'actual_isolated_import_probe.json', observation)
    assert process.returncode == 0, 'Retain original probe failure; do not claim source closure'
    result = json.loads(process.stdout)
    assert result['scientific_imports'] is False and result['actual_once_consumed'] is False
    shutil.copyfile(Path(prior['root']) / 'payload/inert_plan_template.json', payload / 'inert_plan_template.json')
    shutil.copyfile(Path(__file__), payload / Path(__file__).name)
    scope = {'actual_UTC': stamp.isoformat(), 'status': 'STOPPED_CPU_AUDIT_DEPENDENCY_CAPSULE_PREPARATION_ONLY',
             'audit_entry_source_dependency_capsule_assembled': True, 'source_files': len(inventory['source_files']),
             'isolated_standard_library_module_probe_natural_exit': 0,
             'transport_publication_CLI_outside_audit_capsule_scope': True,
             'actual_original_and_new_node_audit_plan_frozen': False,
             'scientific_runtime_import_or_original_CPU_audit': False, 'CPU_CUDA_recovery_qualification': False,
             'task_arrays_or_targets_loaded': False, 'actual_audit_training_prediction_scoring_once_consumed': False,
             'outputs_preserved_unscored': 1, 'outer_scores': 0, 'current_leases_extended': False,
             'new_valid_node_details_received': False}
    write(payload / 'preparation_scope.json', scope)
    (payload / 'preparation.md').write_text(
        'Exact CPU-audit entry dependencies have been assembled from the preserved driver and original pipeline source members. '
        'The ten files retain their original bytes; both parent families share the selected five scientific definition files. '
        'AST dependency inventory covers the audit entry. Deferred transport publication and restoration functions are explicitly outside this entry scope.\n\n'
        'An isolated Python process with site packages disabled imported only the inert metadata modules while forbidding NumPy and Torch imports. '
        'This is a standard-library source/import observation. It does not execute scientific definitions or audit an original checkpoint, qualify the native runtime, '
        'restore CPU/CUDA model/Adam/RNG state, train, predict, decode task targets or score.\n\n'
        'The future actual plan still must bind an eligible original stopped same-fold composite, observed natural exit, complete Release restoration, '
        'the current source and asset inventory, human-provided node/actual lease, exact runtime, fresh resource and saving gates, and a new audit token. '
        'The unchanged inert template is not an execution plan. One of25 unscored outputs remains preserved; zero OUTER scores exist.\n', encoding='utf8')
    proof = seal(payload, payload / 'complete_stopped_CPU_dependency_capsule_original.zip')
    write(root / 'C_receipt.json', proof)
    write(POINTER, {'local_root': str(root), 'payload': str(payload), 'proof': proof, 'scope': scope})
    config = Path('C:/Users/21234/.codex/automations/automation/automation.toml')
    write(WORK / 'group5_automation_before_CPU_capsule.json', tomllib.loads(config.read_text(encoding='utf8')))
    print(json.dumps({'root': str(root), 'proof': proof, 'scope': scope}))


def publish():
    from publish_test_selected_group5_preparation_v1 import sync
    from raw_TRAIN_save_and_publish_v1 import publish as small
    from group5_publish_exact_local_v1 import publish_tree
    p = json.loads(POINTER.read_bytes())
    local = Path(p['payload'])
    root = CONTROL.parent / Path(p['local_root']).name.replace('_prepared_', '_saved_')
    assert root.resolve().parent == CONTROL.parent.resolve() and not root.exists()
    needed = sum(f.stat().st_size for f in local.rglob('*') if f.is_file())
    assert shutil.disk_usage('C:/').free > 200 * 1024**2
    assert shutil.disk_usage('D:/').free > 40 * 1024**2 + needed
    shutil.copytree(local, root / 'payload')
    for original in local.rglob('*'):
        if original.is_file():
            assert digest(original) == digest(root / 'payload' / original.relative_to(local))
    archive = root / 'payload' / Path(p['proof']['archive']).name
    assert digest(archive) == p['proof']['archive_SHA']
    proof = dict(p['proof'], archive=str(archive), D_exact_copy_verified=True,
                 D_verification_UTC=dt.datetime.now(dt.timezone.utc).isoformat(), **verify_zip(archive))
    write(root / 'D_receipt.json', proof)
    p.update(root=str(root), D_proof=proof)
    write(POINTER, p)
    small(archive, proof['archive_SHA'], 'group5-stopped-CPU-dependency-capsule-' + proof['archive_SHA'][:12] + '.zip', root / 'Release_receipt.json')
    github = publish_tree(root, 'results/' + root.name, 'Assemble exact stopped-epoch CPU audit dependencies; original audit and recovery remain unexecuted')
    write(root / 'GitHub_receipt.json', github)
    p['github'] = github
    write(POINTER, p)
    state = json.loads((CONTROL / 'D_current_research_state.json').read_bytes())
    state['latest_human_TEST_selected_group5']['latest_stopped_CPU_dependency_capsule_preparation'] = p
    state.update(updated_at_utc=github['actual_UTC'], github_source=github)
    sync(state)
    with (BASE / 'outputs/研究接续状态.md').open('a', encoding='utf8') as stream:
        stream.write('\n' + github['actual_UTC'] + ' 独立CPU审核入口十项源码依赖同字节装配，隔离无site科学导入探针自然0；仅标准库/源码准备。真实原件/新节点冻结plan、原CPU审核与CPU/CUDA恢复均待，1/25未评分预测及0/25评分不变。原ZIP' + proof['archive_SHA'] + '、GitHub' + github['commit'] + '闭合，D/C同字节。\n')
    print(json.dumps({'root': str(root), 'proof': proof, 'github': github}))


def close():
    from publish_test_selected_group5_preparation_v1 import sync
    from raw_TRAIN_save_and_publish_v1 import publish as small
    from group5_publish_exact_local_v1 import publish_tree
    p = json.loads(POINTER.read_bytes())
    root = Path(p['root']) / 'heartbeat_closure'
    root.mkdir()
    config = Path('C:/Users/21234/.codex/automations/automation/automation.toml')
    actual = tomllib.loads(config.read_text(encoding='utf8'))
    before = json.loads((WORK / 'group5_automation_before_CPU_capsule.json').read_bytes())
    expected = (WORK / 'group5_heartbeat_CPU_capsule_prompt.txt').read_text(encoding='utf8').rstrip('\n')
    assert actual['prompt'] == expected
    assert {k: v for k, v in actual.items() if k not in ('prompt', 'updated_at')} == {k: v for k, v in before.items() if k not in ('prompt', 'updated_at')}
    verification = {'status': 'EXISTING_HEARTBEAT_CPU_CAPSULE_CURRENT_PROMPT_SCHEDULE_UNCHANGED',
                    'actual_UTC': dt.datetime.now(dt.timezone.utc).isoformat(),
                    'prompt_SHA': hashlib.sha256(expected.encode()).hexdigest(), 'actual_TOML_SHA': digest(config),
                    'id': actual['id'], 'active': actual['status'] == 'ACTIVE', 'rrule_unchanged': True,
                    'target_and_other_fields_unchanged': True, 'no_new_automation_or_chat': True,
                    'no_new_original_training_audit_or_score': True}
    write(root / 'actual_automation_verification.json', verification)
    shutil.copyfile(WORK / 'group5_heartbeat_CPU_capsule_prompt.txt', root / 'actual_prompt.txt')
    write(root / 'preparation_reference.json', {'proof': p['D_proof'], 'scope': p['scope'], 'github': p['github']})
    proof = seal(root, root / 'complete_heartbeat_closure.zip')
    write(root.parent / 'heartbeat_D_receipt.json', proof)
    small(Path(proof['archive']), proof['archive_SHA'], 'group5-stopped-CPU-capsule-heartbeat-' + proof['archive_SHA'][:12] + '.zip', root.parent / 'heartbeat_Release_receipt.json')
    github = publish_tree(root, 'results/' + root.parent.name + '_heartbeat', 'Preserve quiet continuation for exact CPU-audit dependency capsule; no original execution')
    write(root.parent / 'heartbeat_GitHub_receipt.json', github)
    p['heartbeat_closure'] = {'verification': verification, 'proof': proof, 'github': github}
    write(POINTER, p)
    state = json.loads((CONTROL / 'D_current_research_state.json').read_bytes())
    state['latest_human_TEST_selected_group5']['latest_stopped_CPU_dependency_capsule_preparation'] = p
    state.update(updated_at_utc=github['actual_UTC'], github_source=github)
    sync(state)
    print(json.dumps({'proof': proof, 'github': github, 'verification': verification}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('assemble', 'publish', 'close'), required=True)
    args = parser.parse_args()
    {'assemble': assemble, 'publish': publish, 'close': close}[args.stage]()
