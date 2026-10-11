"""Prepare a source-only bundle and label-free fold-0 orders; no model/data run."""
import hashlib, json, shutil
from pathlib import Path
from datetime import datetime, timezone
import numpy as np

ROOT = Path('C:/Users/21234/Documents/Codex/2026-10-05/ni')
NEW = ROOT/'work/minimal_fixed_runtime_v1_local_20261006T1228Z'
FLOW = ROOT/'work/minimal_fixed_flow_v2_local_20261006T1214Z'
ROLE = ROOT/'work/group_teacher_plan_20261005T1650Z'
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
parent = json.loads((ROLE/'group_teacher_plan_v3.json').read_text(encoding='utf-8'))
assert sha(ROLE/'group_teacher_plan_v3.json') == 'f0878166da09433c27119a08fdfd8682f9b9e29f2bfe0a758f29eee45cf131a4'
flow_plan = json.loads((FLOW/'local_candidate_plan.json').read_text(encoding='utf-8'))
for name in ('minimal_fixed_flow_v2.py', 'legacy_flow_model.py', 'finite_single_token_reader_v1.py'):
    shutil.copyfile(FLOW/name, NEW/name)
    assert sha(NEW/name) == sha(FLOW/name)
shutil.copyfile(ROOT/'work/inflow_counterfactual_v5_deployment_20261005T0520Z/encoder_adapter.py', NEW/'encoder_adapter.py')
assert sha(NEW/'encoder_adapter.py') == parent['asset_sha256']['frozen_v5/encoder_adapter.py']
for role in ('fit', 'inner', 'outer'):
    name = role+'_0.npy'
    assert sha(ROLE/name) == parent['folds'][0]['files'][name]['sha256']
    shutil.copyfile(ROLE/name, NEW/name)
mapping = ROOT/'work/train_row_video_mapping.json'
assert sha(mapping) == parent['mapping_receipt']['mapping_sha256']
shutil.copyfile(mapping, NEW/mapping.name)
fit = np.load(NEW/'fit_0.npy', allow_pickle=False)
# Separate stream: no consuming author/Torch/dropout/init random state.
rng = np.random.Generator(np.random.PCG64(91819))
orders = np.stack([rng.permutation(fit) for _ in range(100)])
np.save(NEW/'fit_orders_seed91819_100.npy', orders, allow_pickle=False)
singleton_candidates = [454, 505, 620]
assert set(singleton_candidates) <= set(fit.tolist())
mixed = singleton_candidates + [int(i) for i in fit if i not in singleton_candidates][:29]
source = {p.name: sha(p) for p in sorted(NEW.glob('*.py'))}
role_files = {p.name: sha(p) for p in sorted(NEW.iterdir()) if p.suffix in ('.npy', '.json')}
plan = {
 'status': 'LOCAL_CONSTRUCTOR_AND_ROLE_GUARD_SOURCE_CANDIDATE_NOT_GPU_OR_FORMAL100',
 'created_actual_utc': datetime.now(timezone.utc).isoformat(),
 'fold': 0, 'seed': 91819, 'epochs_planned': 100, 'batch_size': 32,
 'fit_rows': 695, 'inner_rows': 153, 'outer_rows': 433,
 'fit_videos': 30, 'inner_videos': 4, 'outer_videos': 18,
 'role_parent_plan_sha256': sha(ROLE/'group_teacher_plan_v3.json'),
 'flow_parent_candidate_plan_sha256': sha(FLOW/'local_candidate_plan.json'),
 'source_sha256': source, 'role_file_sha256': role_files,
 'asset_sha256': {k:v for k,v in parent['asset_sha256'].items() if k != 'frozen_v5/encoder_adapter.py'},
 'scope': 'Only official TRAIN entry indexed. Trusted pickle contains other split bytes; no DEV/TEST entries indexed. No actual pickle loaded locally.',
 'optimizer_proposal': 'Pinned author AdamW lr1e-5, decay groups, warmup .1; optimizer coverage must be actual verified.',
 'task_initialization': 'Public DeBERTa plus new random task seed91819, remove unused author modules before new optimizer. Never task checkpoint or full-TRAIN feature cache.',
 'orders': {'file': 'fit_orders_seed91819_100.npy', 'algorithm': 'NumPy PCG64 independent seed91819 permutations of frozen FIT row IDs', 'numpy_version': np.__version__, 'shape': [100,695], 'updates_per_epoch':22, 'tail':23, 'total_updates':2200, 'shared_first_10_available':True, 'control_candidate_comparison_protocol_frozen':False},
 'precheck_row_candidates': {'normal': orders[0,:32].tolist(), 'mixed_singleton':mixed, 'tail': orders[0,-23:].tolist(), 'actual_masks_checked':False},
 'precheck_proposal': 'Two real optimizer steps: normal and mixed-singleton FIT, full retained finite nonNone gradients; tail23 forward/backward timing only, no extra optimizer step. Construction includes first optimizer state allocation in budget.',
 'inner_precheck': 'Original-input dummy-label INNER153 strict fresh-instance full checkpoint replay <=1e-6, label replacement0 and exact FIT statistics buffer check; actual evidence absent.',
 'outer_authorization': 'Disabled in this candidate. Formal100 completion, earliest INNER minimum, strict disk replay and frozen zero-label OUTER authorization need separate source.',
 'clean_initial': 'Clone state and RNG before any optimizer step; no inheriting precheck optimizer/scheduler/RNG. State persistence and replay runner still missing.',
 'budget_proposal': {'max_GPU_peak_bytes':6*1024**3, 'max_precheck_seconds':1200, 'saving_reserve_seconds':7200, 'actual_runtime_budget_verified':False, 'current_lease_satisfies_reserve':False},
 'GPU_executed':False, 'Torch_model_constructed':False, 'autograd_or_replay_verified':False,
 'precheck_runner_implemented':False, 'formal100_runner_implemented':False,
 'new_scores':False, 'formal_training_allowed':False}
(NEW/'runtime_candidate_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':plan['status'], 'source_count':len(source), 'planned_orders_shape':list(orders.shape), 'new_GPU':False, 'new_scores':False, 'plan_sha256':sha(NEW/'runtime_candidate_plan.json')}))
