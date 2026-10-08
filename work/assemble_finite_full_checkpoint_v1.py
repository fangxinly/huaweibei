import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import argparse,datetime,hashlib,json,sys
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--formal',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--node',choices=['a','b'],required=True);a=p.parse_args()
sys.path.insert(0,str(a.formal));sys.path.append(str(a.root))
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner,make_full_reference,install_finite_inference
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
base=Path('/data/coding/teacher_a_seed91814_best41.pt');assert sha(base)=='967d6a087bb7f117e26df7c6319c37ec992f01d7f7c48bdc9a822a8bd5d9e978'
plan=json.loads((a.formal/'finite_formal_plan_v1.json').read_text())
for n,h in plan['source_sha256'].items():assert sha(a.formal/n)==h
original=torch.load(base,map_location='cpu');model,author,helpers=make_full_reference(a.root,base)
from encoder_adapter import forward_batch
data=helpers.dataset(author,a.root/'assets/mosi.pkl','dev');assert len(data)==229
sel=json.loads((a.run/'selection.json').read_text());history=json.loads((a.run/'history.json').read_text());protocol=json.loads((a.run/'protocol.json').read_text())
mode={'a':'fixed','b':'finite_scalar'}[a.node];assert protocol['requested_mode']==sel['requested_mode']==mode
assert len(history)==100 and sel['epochs']==100 and sel['best_epoch']==np.argmin([x['dev_author_batch_mean_mse'] for x in history])+1
assert sha(a.run/'best_addon.pt')==sel['addon_sha256'] and sha(a.run/'predictions.npz')==sel['prediction_sha256']
assert protocol['formal_plan_sha256']==sha(a.formal/'finite_formal_plan_v1.json')
scales=json.loads((a.formal/'train_scales.json').read_text());learner=FrozenCoordinateLearner(a.root,mode,scales).cuda();learner.restore_addon(torch.load(a.run/'best_addon.pt',map_location='cpu'))
effective='fixed' if sel['best_epoch']<=10 else mode;assert sel['effective_mode']==effective
install_finite_inference(model.dberta,learner,effective)
combined=dict(original)
for key,value in learner.donor.state_dict().items():combined['dberta.own_flow.donor_feedback.'+key]=value.detach().cpu()
for key,value in learner.feedback.state_dict().items():combined['dberta.own_flow.vector_feedback.'+key]=value.detach().cpu()
unchanged=[k for k in original if not k.startswith('dberta.own_flow.donor_feedback.')]
assert all(torch.equal(original[k],combined[k]) for k in unchanged)
a.out.mkdir(parents=True,exist_ok=False);cp=a.out/'full_checkpoint.pt';torch.save(combined,cp)
model.load_state_dict(torch.load(cp,map_location='cpu'),strict=True);model.eval();model.requires_grad_(False)
pred=[];labels=[]
with torch.no_grad():
 for raw in torch.utils.data.DataLoader(data,batch_size=128,shuffle=False):
  raw=tuple(x.cuda() for x in raw);pred.append(forward_batch(model,raw)[0].view(-1).cpu().numpy());labels.append(raw[3].view(-1).cpu().numpy())
pred=np.concatenate(pred);y=np.concatenate(labels)
with np.load(a.run/'predictions.npz') as saved:
 assert np.array_equal(y,saved['valid_y']);error=float(np.max(np.abs(pred-saved['valid_pred'])));assert error<2e-5,error
np.savez(a.out/'full_replay_predictions.npz',valid_pred=pred,valid_y=y)
nz=y!=0
metrics=dict(MAE=float(np.mean(np.abs(pred-y))),MSE229=float(np.mean((pred-y)**2)),Corr=float(np.corrcoef(pred,y)[0,1]),Has0_acc2=float(np.mean((pred>=0)==(y>=0))),Non0_acc2=float(np.mean((pred[nz]>=0)==(y[nz]>=0))),Non0_F1=float(author.f1_score(y[nz]>=0,pred[nz]>=0,average='weighted')),Acc7=float(author.multiclass_acc(np.clip(pred,-3,3),np.clip(y,-3,3))),author_batch_mse=float(np.mean([np.mean((pred[:128]-y[:128])**2),np.mean((pred[128:]-y[128:])**2)])),samples=229,nonzero_samples=int(nz.sum()))
r=dict(status='FINITE_FULL_CHECKPOINT_STRICT_DISK_RELOAD_OFFICIAL_DEV229_REPLAY_COMPLETE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),training_node=a.node,assembly_node='a',requested_mode=mode,effective_mode=effective,best_epoch=sel['best_epoch'],full_checkpoint_sha256=sha(cp),full_checkpoint_bytes=cp.stat().st_size,base_checkpoint_sha256=sha(base),addon_sha256=sha(a.run/'best_addon.pt'),full_raw_vs_cached_prediction_max_error=error,unchanged_base_tensor_count=len(unchanged),full_state_tensor_count=len(combined),allowed_replacements='donor_feedback only plus finite vector_feedback module and TRAIN calibration buffers',metrics=metrics,full_replay_predictions_sha256=sha(a.out/'full_replay_predictions.npz'),source_sha256=sha(__file__),formal_plan_sha256=sha(a.formal/'finite_formal_plan_v1.json'),test_requested=False)
(a.out/'full_checkpoint_receipt.json').write_text(json.dumps(r,indent=2));print('FINITE_FULL_CHECKPOINT_COMPLETE',a.node,sel['best_epoch'],cp.stat().st_size,flush=True)
