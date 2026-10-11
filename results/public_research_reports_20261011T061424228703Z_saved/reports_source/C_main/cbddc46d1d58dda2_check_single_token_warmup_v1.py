import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import sys,json,hashlib,datetime,numpy as np,torch
base=Path('/data/coding/soft_vector_research_20261005T1220Z');old=Path('/data/coding/finite_task_risk_v1_deployment_20261005T1450Z');new=Path('/data/coding/finite_single_token_precheck_20261005T1525Z');sys.path[:0]=[str(new),str(old),str(base)]
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner
from finite_single_token_reader_v1 import install_for_flow
def tsha(values):
 h=hashlib.sha256()
 for n,v in sorted(values.items()):h.update(n.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(91817);np.random.seed(91817)
m=FrozenCoordinateLearner(base,'finite_vector',json.loads((old/'train_scales.json').read_text())).cuda();install_for_flow(m.flow);m.eval();plan=json.loads((old/'finite_formal_plan_v1.json').read_text());initial=tsha(m.state_dict());assert initial==plan['initial_tensor_sha256']
orders=np.stack([np.random.RandomState(91817+1327+i).permutation(1281) for i in range(100)]);np.save(new/'orders.npy',orders);assert hashlib.sha256((new/'orders.npy').read_bytes()).hexdigest()==plan['orders_sha256']
cache=np.load(base/'teacher_cache_v1/train_cache.npz');fields=['state','mask','old_context','pooled_state','reference_prediction','y'];fit=np.load(old/'head_fit_rows.npy');fit_mask=np.zeros(1281,dtype=bool);fit_mask[fit]=True
donors=list(m.donor.parameters());heads=list(m.feedback.parameters());opts=[torch.optim.AdamW(g,lr=1e-4,weight_decay=.01) for g in [donors,heads]]
for step in range(400):
 ids=orders[step//40,(step%40)*32:(step%40+1)*32];b={n:torch.as_tensor(cache[n][ids],device='cuda') for n in fields};m.zero_grad(set_to_none=True)
 for opt in opts:
  for g in opt.param_groups:g['lr']=1e-4*min((step+1)/400,max(0.,(4000-step)/3600))
 pred,n,obs=m(b,'fixed');task=(pred-b['y']).square().mean();aux=m.feedback.residual_loss(n,b['reference_prediction']-b['y'],torch.as_tensor(fit_mask[ids],device='cuda'));(task+.01*aux).backward()
 for g,opt in zip([donors,heads],opts):
  assert all(p.grad is None or torch.isfinite(p.grad).all() for p in g);torch.nn.utils.clip_grad_norm_(g,1);opt.step()
 if step==19:assert tsha(m.state_dict())==plan['preflight_after20_tensor_sha256']
expected=FrozenCoordinateLearner(base,'finite_vector',json.loads((old/'train_scales.json').read_text())).cuda();expected.restore_addon(torch.load(old/'run/shared_phase_addon.pt',map_location='cpu'))
errors={n:float((v-expected.state_dict()[n]).abs().max()) if v.dtype.is_floating_point else int((v!=expected.state_dict()[n]).any()) for n,v in m.state_dict().items()};assert max(errors.values())==0
r=dict(status='SINGLE_TOKEN_INITIAL_AFTER20_ALL100ORDERS_SHARED400_WARMUP_EXACT',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),initial_tensor_sha256=initial,after20_matches_original=True,orders_sha256=plan['orders_sha256'],shared10_tensor_sha256=tsha(m.state_dict()),original_shared10_tensor_sha256=tsha(expected.state_dict()),all_tensor_max_error=0.,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),test_requested=False,dev_requested=False,scope='TRAIN400updates warmup mechanism only, no100formal orDEVselection. All original frozen parameters unchanged; old C failure remains.')
(new/'warmup_receipt.json').write_text(json.dumps(r,indent=2));print('SINGLE_TOKEN_WARMUP_EXACT_COMPLETE')
