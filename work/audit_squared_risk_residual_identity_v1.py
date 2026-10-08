"""Independent NumPy reconstruction of synthetic witnesses only."""
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'work/squared_risk_residual_math_20261006T1259Z'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt=json.loads((OUT/'synthetic_math_receipt.json').read_text())
assert sha(ROOT/'work/squared_risk_residual_identity_v1.py')==receipt['source_sha256']
assert sha(OUT/'synthetic_witnesses.npz')==receipt['arrays_sha256']
with np.load(OUT/'synthetic_witnesses.npz',allow_pickle=False) as z:
    p,d,r,h=(z[k] for k in ('p','delta','r','h'))
    q1=(d-r)**2-r**2
    q2=d*(d-2*r)
    err1=float(np.max(np.abs(q1-q2)))
    err2=float(np.max(np.abs((2*d*(r-h))**2-4*d*d*(h-r)**2)))
    assert err1<1e-12 and err2<1e-12
    assert np.array_equal(np.argmin((z['pool']-h[:,None])**2,axis=1),z['index_Q'])
    toy_q=(z['toy_delta']-z['toy_y'])**2-z['toy_y']**2
    assert np.allclose(toy_q,z['toy_Q'],atol=1e-15,rtol=0)
    for x in (-1.,1.):
        idx=z['toy_x']==x
        assert abs(toy_q[idx].mean()+.5)<1e-15
    assert set(z.files)=={'p','delta','r','h','pool','index_Q','index_nearest','toy_x','toy_noise','toy_y','toy_delta','toy_Q','toy_full_Qhat','toy_coarse_Qhat'}
assert not any(receipt[k] for k in ('official_data_or_labels_read','Torch_or_GPU','new_task_scores','other_node_CPU_or_remote_capture'))
result={'status':'LOCAL_INDEPENDENT_SYNTHETIC_ARRAY_SHA_AND_RECONSTRUCTION_PASSED',
 'actual_utc':datetime.now(timezone.utc).isoformat(),'max_reconstruction_error':[err1,err2],
 'source_sha256':sha(Path(__file__)),'original_receipt_sha256':sha(OUT/'synthetic_math_receipt.json'),
 'other_node_CPU':False,'official_task_metrics':False}
(OUT/'independent_synthetic_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
