"""Readonly TRAIN-gradient reproduction from saved shared10 checkpoint."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,datetime,hashlib,json,sys,traceback
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--base-root',type=Path,required=True);p.add_argument('--deployment',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path[:0]=[str(a.deployment),str(a.base_root)]
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(91817)
assert not a.out.exists();a.out.mkdir()
scales=json.loads((a.deployment/'train_scales.json').read_text());model=FrozenCoordinateLearner(a.base_root,'finite_vector',scales).cuda()
checkpoint=a.deployment/'run/shared_phase_addon.pt';model.restore_addon(torch.load(checkpoint,map_location='cpu'))
cache=np.load(a.base_root/'teacher_cache_v1/train_cache.npz',allow_pickle=False);order=np.load(a.deployment/'run/orders.npy')[10]
records=[];firstbad=None
for index in range(40):
    ids=order[index*32:(index+1)*32];b={n:torch.as_tensor(cache[n][ids],device='cuda') for n in ['state','mask','old_context','pooled_state','reference_prediction','y']}
    model.zero_grad(set_to_none=True)
    pred,_,obs=model(b,'finite_vector');loss=(pred-b['y']).square().mean();loss.backward()
    bad={n:int((~torch.isfinite(p.grad)).sum()) for n,p in model.donor.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()}
    records.append(dict(batch=index,finite=not bool(bad),bad_parameter_counts=bad,loss=float(loss.detach())))
    if bad:
        firstbad=index;np.savez(a.out/'witness_TRAIN_batch.npz',row_id=ids,**{n:cache[n][ids] for n in ['state','mask','old_context','pooled_state','reference_prediction','y']})
        model.zero_grad(set_to_none=True)
        try:
            with torch.autograd.detect_anomaly(check_nan=True):
                prediction=model(b,'finite_vector')[0];(prediction-b['y']).square().mean().backward()
        except Exception:
            (a.out/'anomaly_trace.txt').write_text(traceback.format_exc())
        break
report=dict(status='READONLY_SHARED10_TRAIN_GRADIENT_REPRODUCTION',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),first_nonfinite_batch=firstbad,records=records,dev_requested=False,test_requested=False,training_not_restarted=True,scope='No optimizer updates; this probes40epoch11batches at fixed shared10weights, not a recovered full optimizer state or a100epoch run.')
(a.out/'reproduction.json').write_text(json.dumps(report,indent=2));print('READONLY_GRADIENT_REPRODUCTION_COMPLETE',firstbad,flush=True)
