"""Preserved CAL mechanism links and FIT0 shape/role metadata. No model or raw y."""
from pathlib import Path
import hashlib
import json
import numpy as np

root=Path(__file__).resolve().parents[1]
evidence=Path('D:/CodexBackups/selective_flow_20261003_1105/native_radius_cal_mechanism_actual_20261006T072031Z')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((evidence/'mechanism/receipt.json').read_text(encoding='utf-8'))
audit=json.loads((evidence/'mechanism/independent_audit.json').read_text(encoding='utf-8'))
joint=json.loads((evidence/'joint_preservation_audit.json').read_text(encoding='utf-8'))
coverage=json.loads((evidence/'nonF_CAL_coverage_analysis.json').read_text(encoding='utf-8'))
pred=evidence/'mechanism/predictions_frozen.npz'
if not pred.exists():
    files=list((evidence/'mechanism').glob('*.npz'))
    assert len(files)==1
    pred=files[0]
expected='f6685f7797f0deca4ceaf12858b2ba26831c9ca6179313f8497a1cf61d6ec8b6'
assert sha(pred)==expected
assert receipt['prediction_sha256']==audit['array_sha256']==joint['prediction_sha256']==coverage['GPU_array_sha256']==expected
assert receipt['model_state_before']==receipt['model_state_after']
with np.load(pred,allow_pickle=False) as z:
    members=list(z.files)
    assert 'y' not in members
planroot=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni/work/group_teacher_plan_20261005T1650Z')
fit_path=planroot/'fit_0.npy'; inner_path=planroot/'inner_0.npy'
assert sha(fit_path)=='b93d8e4f5271354a1b5741e824f895ecec292b1397be3521ae8474600cce110d'
assert sha(inner_path)=='c3484b79520d283d9b3785d8f374c804f9c67e9f4df8fa5a9ce68bb83294d081'
fit=np.load(fit_path,allow_pickle=False); inner=np.load(inner_path,allow_pickle=False)
assert len(fit)==695 and len(inner)==153 and not np.intersect1d(fit,inner).size
previous=Path('D:/CodexBackups/selective_flow_20261003_1105/matched_message_pools_actual_v2_20261006T065508Z/execute/predictions_frozen.npz')
with np.load(previous,allow_pickle=False) as z:
    rows=z['row']; lengths=z['valid_lengths']
    assert np.array_equal(rows,np.arange(1281))
    singleton_fit=fit[lengths[fit]==1].tolist()
out=dict(scope='CAL no-label mechanism integrity and fold0 role/length metadata only',
         raw_y_read=False,new_gpu=False,new_forward=False,new_fit=False,
         array_path=str(pred),array_sha256=expected,array_members=members,
         coverage={k:v['all'] for k,v in coverage['pools'].items()},
         zero_radius_boundary_tested=coverage['zero_radius_synthetic_GPU_boundary_tested'],
         selected_predictions_generated=receipt['selected_predictions_generated'],
         fold0_fit_singleton_candidates_from_metadata=singleton_fit,
         singleton_roles={str(i):'FIT0' if i in fit else 'INNER0' if i in inner else 'OUTER0' for i in (454,620)},
         full_batches32=695//32,tail_rows32=695%32,updates_per_epoch_batch32=(695+31)//32)
(root/'work/fifth_coverage_and_fit_roles_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
