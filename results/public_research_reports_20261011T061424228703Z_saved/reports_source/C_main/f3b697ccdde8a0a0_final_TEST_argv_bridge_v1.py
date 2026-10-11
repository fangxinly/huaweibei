"""Explicit separately pinned comparison-view supplement; original sources stay byte exact.

CLI: bridge.py MODE ORIGINAL_BUNDLE BRIDGE_PLAN_SHA [original mode arguments].
Only process versus sys.argv representation is corrected. Predict is prohibited.
Original __file__ hashes identify original functions, not this supplemental audit;
the supplemental source and plan are separately preserved in every new stage.
"""
import hashlib
import inspect
import json
import shutil
import sys
from pathlib import Path, PurePosixPath

PYTHON = '/data/coding/multimodal_flow_public_20261006T1341Z/.venv/bin/python'
PLAN = 'f41d05150d5ff1522c45508869981ea9e878aa75327e79f8a6895a0323ef62b6'

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def argv_equal(process, script, expected_script, native_python=PYTHON):
    return (type(process) is list and type(script) is list and
            all(type(x) is str for x in process + script) and
            native_python == PYTHON and len(process) > 2 and
            process[0] == PYTHON and process[1] == expected_script and
            process[1:] == script)

def replace_function(module, name, replacements, additions=None):
    source = inspect.getsource(getattr(module, name))
    original_hash = hashlib.sha256(source.encode()).hexdigest()
    for old, new in replacements:
        if source.count(old) != 1:
            raise PermissionError('Exact single original comparison site required: ' + name)
        source = source.replace(old, new)
    if additions:
        module.__dict__.update(additions)
    exec(compile(source, '<separately-pinned-argv-comparison-view:'+name+'>', 'exec'), module.__dict__)
    return original_hash

def install(bundle, bridge_sha):
    bp = Path(__file__).with_name('argv_bridge_plan.json')
    if digest(bp) != bridge_sha:
        raise PermissionError('Supplement plan SHA mismatch')
    spec = json.loads(bp.read_text())
    if spec['bridge_source_sha256'] != digest(__file__) or spec['original_plan_sha256'] != PLAN:
        raise PermissionError('Supplement source/unchanged original plan mismatch')
    bundle = Path(bundle)
    sys.path.insert(0, str(bundle))
    import paired_final_TEST_pipeline_candidate_v1 as pipeline
    plan = pipeline.plan_gate(bundle, PLAN)
    for name, h in spec['original_sources'].items():
        if digest(bundle/name) != h:
            raise PermissionError('Supplement original exact source mismatch: '+name)
    hashes = {}
    hashes['original_prediction_gate'] = replace_function(pipeline, 'original_prediction_gate', [
        ("r['fullargv'] == launch['fullargv']", "argv_equal(launch['fullargv'], r['fullargv'], str(PurePosixPath(launch['root'])/'source/paired_final_TEST_pipeline_candidate_v1.py'), load(root/'out/actual_native_preflight.json')['actual_python_executable'])")
    ], {'argv_equal': argv_equal, 'PurePosixPath': PurePosixPath})
    import paired_final_TEST_D_audit_candidate_v1 as auditor
    hashes['audit_capsule'] = replace_function(auditor, 'audit_capsule', [
        ("x['capture_child_fullargv']==r['capture_fullargv']", "argv_equal(x['capture_child_fullargv'], r['capture_fullargv'], r['capture_fullargv'][0], load(dest/'actual_capture_native_preflight.json')['actual_python_executable']) and PurePosixPath(r['capture_fullargv'][0]).name=='paired_final_TEST_capture_candidate_v1.py'"),
        ("stage['fullargv']==launch['fullargv']", "stage_argv_equal(launch, stage, root)")
    ], {'argv_equal': argv_equal, 'stage_argv_equal': stage_argv_equal})
    hashes['pipeline_main'] = replace_function(pipeline, 'main', [
        ('a=p.parse_args();started=', 'a=p.parse_args(sys.argv[4:]);started=')
    ])
    import paired_final_TEST_natural_wrapper_candidate_v1 as wrapper
    hashes['wrapper_main'] = replace_function(wrapper, 'main', [
        ('a=p.parse_args();plan=', 'a=p.parse_args(sys.argv[4:]);plan='),
        ('plan_gate(source,a.plan_sha)\n    command=', 'plan_gate(source,a.plan_sha)\n    require(a.stage in ("cpu","score"), "Supplement forbids new prediction")\n    supplement=a.root/"supplement";supplement.mkdir()\n    shutil.copyfile(bridge_file,supplement/"final_TEST_argv_bridge_v1.py")\n    shutil.copyfile(bridge_plan,supplement/"argv_bridge_plan.json")\n    command='),
        ("str(source/'paired_final_TEST_pipeline_candidate_v1.py'),", "str(supplement/'final_TEST_argv_bridge_v1.py'),'stage',str(source),bridge_digest,")
    ], {'bridge_file': Path(__file__), 'bridge_plan': bp, 'bridge_digest': bridge_sha})
    if spec['original_function_sha256'] != hashes:
        raise PermissionError('Exact supplemental function patch provenance mismatch')
    return pipeline, auditor, wrapper, spec

def stage_argv_equal(launch, stage, root):
    remote = PurePosixPath(launch['root'])
    if launch['stage'] == 'predict':
        expected = str(remote/'source/paired_final_TEST_pipeline_candidate_v1.py')
    else:
        expected = str(remote/'supplement/final_TEST_argv_bridge_v1.py')
        bp = root/'supplement/argv_bridge_plan.json'
        if (digest(root/'supplement/final_TEST_argv_bridge_v1.py') != digest(__file__) or
                digest(bp) != digest(Path(__file__).with_name('argv_bridge_plan.json'))):
            return False
    native = json.loads((root/'out/actual_native_preflight.json').read_text())
    return argv_equal(launch['fullargv'], stage['fullargv'], expected, native['actual_python_executable'])

def main():
    mode, bundle, bridge_sha = sys.argv[1:4]
    pipeline, auditor, wrapper, spec = install(bundle, bridge_sha)
    if mode == 'wrapper':
        wrapper.main()
    elif mode == 'stage':
        if '--stage' not in sys.argv or sys.argv[sys.argv.index('--stage')+1] not in ('cpu','score'):
            raise PermissionError('No predictive or training execution in supplement')
        pipeline.main()
    elif mode == 'audit':
        import argparse
        p=argparse.ArgumentParser()
        p.add_argument('--dest',type=Path,required=True);p.add_argument('--node',required=True)
        a=p.parse_args(sys.argv[4:])
        result=auditor.audit_capsule(a.dest, PLAN, a.node)
        pipeline.write(a.dest/'actual_D_argv_supplement_provenance.json', {
            'actual_utc':pipeline.utc(), 'status':'EXACT_ARGV_REPRESENTATION_SUPPLEMENT_D_AUDIT_PASSED',
            'bridge_plan_sha256':bridge_sha, 'bridge_source_sha256':digest(__file__),
            'original_plan_sha256':PLAN,'original_failure_preserved':True,
            'original_audit_result_sha256':digest(a.dest/'actual_D_saved_capsule_audit.json'),
            'no_model_forward_or_labels':True})
        print(json.dumps(result))
    elif mode == 'binding':
        import paired_final_TEST_score_binding_candidate_v1 as binding
        replace_function(binding,'main',[('a=p.parse_args();plan=', 'a=p.parse_args(sys.argv[4:]);plan=')])
        binding.main()
    else:
        raise PermissionError('Unknown supplement mode')

if __name__ == '__main__':
    main()
