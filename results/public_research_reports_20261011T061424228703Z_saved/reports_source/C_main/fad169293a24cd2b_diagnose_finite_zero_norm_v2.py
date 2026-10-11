import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import sys,json,numpy as np,torch,datetime,hashlib
base=Path('/data/coding/soft_vector_research_20261005T1220Z');p=Path('/data/coding/finite_task_risk_v1_deployment_20261005T1450Z');d=Path('/data/coding/failed_finite_vector_diagnostics_20261005T1503Z');sys.path[:0]=[str(p),str(base)]
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner
torch.set_num_threads(2);torch.manual_seed(91817);m=FrozenCoordinateLearner(base,'finite_vector',json.loads((p/'train_scales.json').read_text())).cuda();m.restore_addon(torch.load(p/'run/shared_phase_addon.pt',map_location='cpu'));z=np.load(d/'witness_TRAIN_batch.npz');b={n:torch.as_tensor(z[n],device='cuda') for n in ['state','mask','old_context','pooled_state','reference_prediction']};rows=[]
def hook(module,args,result):
 s=result['slots'];v=s-s.mean(2,keepdim=True);norm=v.norm(dim=-1);bad=(norm==0).flatten(1).any(1);ids=bad.nonzero().flatten();rows.append(dict(zero_count=int((norm==0).sum()),row_ids=z['row_id'][ids.cpu().numpy()].tolist(),valid_lengths=b['mask'][ids].sum(1).cpu().tolist(),minimum_norm=float(norm.min()),q_for_zero_rows=result['q'][ids].detach().cpu().numpy().tolist()))
h=m.flow.reader.register_forward_hook(hook);m(b,'finite_vector');h.remove()
(d/'zero_norm_probe_v2.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=rows,no_optimizer_update=True,no_label_requested=True,test_requested=False,dev_requested=False),indent=2));print('ZERO_NORM_PROBE_V2_COMPLETE')
