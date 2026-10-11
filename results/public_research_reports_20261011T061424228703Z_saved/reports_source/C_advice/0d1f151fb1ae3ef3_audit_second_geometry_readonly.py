"""Read already preserved geometry and link original receipts. No outcomes or models."""
from pathlib import Path
import hashlib
import json
import numpy as np

root=Path(__file__).resolve().parents[1]
evidence=Path('D:/CodexBackups/selective_flow_20261003_1105/scalar_geometry_actual_20261006T060719Z')
source=Path('C:/Users/21234/Documents/Codex/2026-10-05/ni/work/diagnose_scalar_direction_geometry_v1.py')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
expected={
    evidence/'execute/geometry_frozen.npz':'e0016c23628ca0584784e417465277e4ef516cc884ff22ea4f2113c26f9ead91',
    source:'c84b97838d76d373aa4d1e65efd8ca39d9b929ba4da8b377306b820f8e44bcd1',
    evidence/'execute/receipt.json':'6006f0c94925b358b3f149b47991cb310502f3181c623654f296314ac5bc6864',
    evidence/'independent_cpu/cpu_preservation_receipt.json':'46c5e5500f2fb8a8062c0dcad281e3c43dbee25d3cca026faf63876ef5c6615c',
}
for p,value in expected.items(): assert sha(p)==value, str(p)
gpu=json.loads((evidence/'execute/receipt.json').read_text(encoding='utf-8'))
cpu=json.loads((evidence/'independent_cpu/cpu_preservation_receipt.json').read_text(encoding='utf-8'))
assert gpu['model_state_before']==gpu['model_state_after']
assert cpu['original_gpu_receipt_sha256']==sha(evidence/'execute/receipt.json')
assert json.loads((evidence/'execute_exit.json').read_text())['exit_code']==0
assert json.loads((evidence/'independent_cpu/cpu_exit.json').read_text())['exit_code']==0
with np.load(evidence/'execute/geometry_frozen.npz',allow_pickle=False) as z:
    members=list(z.files)
    assert 'y' not in members
    rows=z['row']; assert np.array_equal(rows,np.arange(1281))
    assert len(np.unique(z['video']))==52
    old=z['old_gradient'].reshape(1281,-1).astype(np.float64)
    teacher=z['teacher_gradient'].reshape(1281,-1).astype(np.float64)
    norms_o=np.linalg.norm(old,axis=1); norms_t=np.linalg.norm(teacher,axis=1)
    assert np.all(norms_o>0) and np.all(norms_t>0)
    cosine=np.sum((old/norms_o[:,None])*(teacher/norms_t[:,None]),axis=1)
    same=int(np.sum(cosine>0)); opposite=int(np.sum(cosine<0))
    assert (same,opposite)==(666,615)
    result=dict(
        scope='Local original-file hashes, receipt links and unlabelled geometry members only',
        outcomes_read=False, cal_parameters_fitted=False, model_forward=False, gpu_called=False,
        files=[dict(path=str(p),sha256=v) for p,v in expected.items()],
        members=members,rows=1281,videos=52,same=same,opposite=opposite,
        collinearity_max_error=float(np.max(np.abs(np.abs(cosine)-1))),
        original_gpu_exit0=True,original_cpu_exit0=True,
        original_gpu_seconds=gpu['seconds'],original_gpu_peak_bytes=gpu['actual_peak_allocated_bytes'],
        source_implementation_note='prox uses (point-point.detach()) in geometry probe; see report for fixed-anchor solver distinction',
    )
(root/'work/second_geometry_readonly_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
