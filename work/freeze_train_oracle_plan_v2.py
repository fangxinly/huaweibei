"""Freeze a new diagnostic using already preserved C2 source/selection evidence."""
from pathlib import Path
import ast, datetime, hashlib, json, zipfile

work = Path(__file__).resolve().parent
outputs = work.parent / 'outputs'
saved = Path('D:/CodexBackups/selective_flow_20261003_1105/finite_c2_completed_20261005/c')
snapshot = Path('D:/CodexBackups/selective_flow_20261003_1105/finite_c2_completed_snapshots_20261005T1627Z/a/snapshot.zip')
sha = lambda b: hashlib.sha256(b).hexdigest()
protocol = json.loads((saved/'protocol.json').read_text(encoding='utf-8'))
selection = json.loads((saved/'selection.json').read_text(encoding='utf-8'))
oldplan = json.loads((saved/'finite_formal_plan_v2.json').read_text(encoding='utf-8'))
assert selection['epochs'] == 100 and selection['best_epoch'] == 37
base = '/data/coding/soft_vector_research_20261005T1220Z'
formal = '/data/coding/finite_task_risk_c2_deployment_20261005T1600Z'
pins = {}
for name, expected in protocol['source_sha256'].items():
    assert sha((saved/name).read_bytes()) == expected
    pins[formal+'/'+name] = expected
for name, expected in protocol['base_source_sha256'].items():
    pins[base+'/'+name] = expected
for name in ['selection.json','best_addon.pt','protocol.json']:
    pins[formal+'/run/'+name] = sha((saved/name).read_bytes())
for name in ['finite_formal_plan_v2.json','train_scales.json']:
    pins[formal+'/'+name] = sha((saved/name).read_bytes())
with zipfile.ZipFile(snapshot) as z:
    large = json.loads(z.read('large_file_manifest.json'))
    for filename in ['train_cache.npz','frozen_terminal.pt','train_gradient_rms.npy']:
        key = 'source/teacher_cache_v1/'+filename
        if key in z.namelist():
            pins[base+'/teacher_cache_v1/'+filename] = sha(z.read(key))
        else:
            matches = [v for v in large.values() if v['path'] == base+'/teacher_cache_v1/'+filename]
            if matches:
                assert len(matches) == 1
                pins[base+'/teacher_cache_v1/'+filename] = matches[0]['sha256']
    collection = json.loads(z.read('source/teacher_cache_v1/collection.json'))
    mapping_bytes = z.read('train_group_mapping/train_row_video_mapping.json')
    mapping = json.loads(mapping_bytes)
    mapping_path = '/data/coding/train_group_mapping_20261005T1628Z/train_row_video_mapping.json'
    pins[mapping_path] = sha(mapping_bytes)
# Cache hashes not captured as source assets are pinned by original collection.
def find_hash(obj, filename):
    hits = []
    if isinstance(obj, dict):
        if obj.get('path','').endswith('/'+filename) and 'sha256' in obj:
            hits.append(obj['sha256'])
        if filename in obj:
            value = obj[filename]
            if isinstance(value, dict) and 'sha256' in value: hits.append(value['sha256'])
            if isinstance(value, str) and len(value) == 64: hits.append(value)
        for value in obj.values(): hits.extend(find_hash(value, filename))
    elif isinstance(obj, list):
        for value in obj: hits.extend(find_hash(value, filename))
    return hits
for filename in ['train_cache.npz','frozen_terminal.pt','train_gradient_rms.npy']:
    path = base+'/teacher_cache_v1/'+filename
    if path not in pins:
        hits = set(find_hash(collection, filename))
        assert len(hits) == 1, ('Missing preserved cache fingerprint', filename, list(collection.keys()))
        pins[path] = hits.pop()
assert len(mapping) == 1281 and len(set(x['video_id'] for x in mapping)) == 52
source = work/'diagnose_train_oracle_v2.py'
ast.parse(source.read_text(encoding='utf-8'))
utc = datetime.datetime.now(datetime.timezone.utc)
stamp = utc.strftime('%Y%m%dT%H%M%SZ')
newroot = '/data/coding/train_oracle_diagnostic_'+stamp
plan = dict(status='FROZEN_TRAIN_ORACLE_DIAGNOSTIC_NOT_EXECUTED', utc=utc.isoformat(),
    new_root=newroot, base_root=base, formal_root=formal,
    expected_uuid='GPU-2a0c83a4-0a1a-494d-6e80-0c756fb075aa',
    rows=1281, videos=52, batch_size=32, row_order='official TRAIN original order; last partial retained',
    conditions=['raw_fixed','learned_residual_last','true_residual_oracle_last',
                'true_residual_oracle_best_of_F_and_three_steps','zero_residual_last','cyclic_mismatched_label_last'],
    controller=oldplan['controller'], controller_override=False, optimizer_steps=0,
    maximum_seconds=2700, preservation_reserve_seconds=7200,
    minimum_remote_free_bytes=1073741824, maximum_peak_allocated_bytes=1073741824,
    estimated_lease_end_utc='2026-10-06T12:08:17+00:00', platform_lease_end_verified=False,
    mapping_path=mapping_path, diagnostic_source_sha256=sha(source.read_bytes()), pinned_files=pins,
    local_permanent_free_required_bytes=1073741824,
    first_batch_gate='Finite gradient/HVP, direct/expanded identity; complete only after all1281 and original controller replay.',
    oracle_label_scope='TRAIN diagnostic only; true y prohibited from learned and deployable controllers.',
    metrics=['sample-weighted/video-equal MSE and MAE','delta/risk means/improvement fraction',
             'four candidate predictions/prox/objectives/trust norms/channel distances/cosines',
             'residual MAE/MSE/sign; proxy-vs-observed Pearson and Spearman'],
    interpretation='Finite three-step feasible diagnostic, no global upper bound, no train-to-test guarantee.',
    save_contract='No new fullweight: preserve source/plan/arrays/original receipt/log; local D SHA and independent CPU receipt. Extend capture coverage before actual deployment.',
    scope='Frozen selected C2 and fullTRAINfit/DEVselected reference. TRAIN label-known feasibility only, not OOF, semantic truth, generalization or a new trained model.',
    deployment_executed=False, gpu_precheck_executed=False, diagnostic_executed=False,
    remote_tool_status='Same SSH input rejected by current Granular.sandbox_approval=false; no bypass or alternative connection.')
target = work/'train_oracle_plan_v2.json'
assert not target.exists()
target.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding='utf-8')
proof = dict(status='LOCAL_PINNED_ORACLE_PLAN_FROZEN_NOT_GPU_TESTED',utc=utc.isoformat(),
             source_sha256=plan['diagnostic_source_sha256'], plan_sha256=sha(target.read_bytes()),
             pinned_files=len(pins), selection=37, rows=1281, video_groups=52,
             gpu_executed=False, source_ast_parsed=True, labels_separated_in_controls=True)
(outputs/'TRAIN_Oracle实验本地冻结核验.json').write_text(json.dumps(proof, ensure_ascii=False, indent=2),encoding='utf-8')
print(json.dumps(proof, ensure_ascii=True))
