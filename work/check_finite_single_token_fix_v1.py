import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import json,sys,datetime,hashlib,copy,time
import torch,numpy as np
base=Path('/data/coding/soft_vector_research_20261005T1220Z');old=Path('/data/coding/finite_task_risk_v1_deployment_20261005T1450Z');new=Path('/data/coding/finite_single_token_precheck_20261005T1525Z');sys.path[:0]=[str(new),str(old),str(base)]
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner
from finite_single_token_reader_v1 import install_for_flow
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(91817)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
scales=json.loads((old/'train_scales.json').read_text());m=FrozenCoordinateLearner(base,'finite_vector',scales).cuda();m.restore_addon(torch.load(old/'run/shared_phase_addon.pt',map_location='cpu'));v=copy.deepcopy(m);install_for_flow(v.flow)
assert all(torch.equal(a,b) for a,b in zip(m.state_dict().values(),v.state_dict().values()))
cache=np.load(base/'teacher_cache_v1/train_cache.npz');orders=np.load(old/'run/orders.npy');fields=['state','mask','old_context','pooled_state','reference_prediction','y'];records=[];maxerror=0.;start=time.monotonic()
for i in range(0,1281,32):
 ids=np.arange(i,min(i+32,1281));b={n:torch.as_tensor(cache[n][ids],device='cuda') for n in fields}
 with torch.no_grad():
  p0=m(b,'finite_vector')[0];p1=v(b,'finite_vector')[0]
 error=float((p0-p1).abs().max());assert error<2e-5;maxerror=max(maxerror,error)
 v.zero_grad(set_to_none=True);pred,n,obs=v(b,'finite_vector');loss=(pred-b['y']).square().mean();loss.backward()
 assert all(p.grad is None or torch.isfinite(p.grad).all() for p in v.parameters())
 records.append(dict(first_row=i,count=len(ids),forward_error=error,all_gradients_finite=True))
# Full actual failing witness, frozen shared10 model, no optimizer update.
z=np.load('/data/coding/failed_finite_vector_diagnostics_20261005T1503Z/witness_TRAIN_batch.npz');b={n:torch.as_tensor(z[n],device='cuda') for n in fields};v.zero_grad(set_to_none=True)
with torch.autograd.detect_anomaly(check_nan=True):
 pred,n,obs=v(b,'finite_vector');(pred-b['y']).square().mean().backward()
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in v.parameters())
# Pilot uses only TRAIN. This does not launch/recover original formal v1.
fit=np.load(old/'head_fit_rows.npy');fit_mask=np.zeros(1281,dtype=bool);fit_mask[fit]=True
donors=list(v.donor.parameters());heads=list(v.feedback.parameters());opts=[torch.optim.AdamW(g,lr=1e-4,weight_decay=.01) for g in [donors,heads]]
pilot=[]
for step in range(200):
 ids=orders[10+step//40,(step%40)*32:(step%40+1)*32];b={n:torch.as_tensor(cache[n][ids],device='cuda') for n in fields};v.zero_grad(set_to_none=True)
 pred,n,obs=v(b,'finite_vector');task=(pred-b['y']).square().mean();aux=v.feedback.residual_loss(n,b['reference_prediction']-b['y'],torch.as_tensor(fit_mask[ids],device='cuda'));(task+.01*aux).backward()
 for g,opt in zip([donors,heads],opts):
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in g);torch.nn.utils.clip_grad_norm_(g,1);opt.step()
 pilot.append(float(task.detach()))
r=dict(status='SINGLE_TOKEN_ANALYTIC_BRANCH_TRAIN_PRECHECK_COMPLETE_NOT_FORMAL100',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),reader_source_sha256=sha(new/'finite_single_token_reader_v1.py'),original_frozen_reader_source_sha256=sha(base/'frozen_v5/legacy_flow_model.py'),train_rows=1281,all_train_shared10_gradient_batches=records,maximum_all_train_prediction_error=maxerror,failed_witness_gradient_finite=True,pilot_steps=200,pilot_losses=pilot,scope='Pilot starts shared10 addon with fresh Adam states, not original optimizer recovery; original C remains failed. All TRAIN labels only main loss; controller is label free. Analytic single-token cosine=0 branch avoids constructing undefined norm double backward. Other rows unchanged. No raw full classifier check yet.',test_requested=False,dev_requested=False,elapsed_seconds=time.monotonic()-start)
(new/'mechanism_receipt.json').write_text(json.dumps(r,indent=2));print('SINGLE_TOKEN_PRECHECK_COMPLETE',maxerror,flush=True)
