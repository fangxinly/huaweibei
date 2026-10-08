import os
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import argparse,datetime,hashlib,json,sys
import numpy as np
import torch
from soft_vector_runtime_v2 import FrozenCoordinateLearner,make_full_reference,install_vector_inference
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--incoming',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
torch.set_num_threads(2);base=Path('/data/coding/teacher_a_seed91814_best41.pt');assert sha(base)=='967d6a087bb7f117e26df7c6319c37ec992f01d7f7c48bdc9a822a8bd5d9e978'
original=torch.load(base,map_location='cpu');model,author,helpers=make_full_reference(a.root,base)
from encoder_adapter import forward_batch
data=helpers.dataset(author,a.root/'assets/mosi.pkl','dev');assert len(data)==229
for node,mode in [('a','fixed'),('b','scalar'),('c','soft_projected')]:
 run=a.incoming/node;sel=json.loads((run/'selection.json').read_text());history=json.loads((run/'history.json').read_text());protocol=json.loads((run/'protocol.json').read_text())
 assert len(history)==100 and sel['epochs']==100 and sel['best_epoch']==np.argmin([x['dev_author_batch_mean_mse'] for x in history])+1
 assert sha(run/'best_addon.pt')==sel['addon_sha256'] and sha(run/'predictions.npz')==sel['prediction_sha256']
 learner=FrozenCoordinateLearner(a.root,mode,.1).cuda();learner.restore_addon(torch.load(run/'best_addon.pt',map_location='cpu'));effective='fixed' if sel['best_epoch']<=10 else mode
 install_vector_inference(model.dberta,learner,effective)
 combined=dict(original)
 for key,value in learner.donor.state_dict().items():combined['dberta.own_flow.donor_feedback.'+key]=value.detach().cpu()
 for key,value in learner.feedback.state_dict().items():combined['dberta.own_flow.vector_feedback.'+key]=value.detach().cpu()
 unchanged=[k for k in original if not k.startswith('dberta.own_flow.donor_feedback.')]
 assert all(torch.equal(original[k],combined[k]) for k in unchanged)
 dest=a.out/node;dest.mkdir(parents=True,exist_ok=False);cp=dest/'full_checkpoint.pt';torch.save(combined,cp)
 # Actual disk reload into complete classifier, all keys strict; addon not substituted.
 model.load_state_dict(torch.load(cp,map_location='cpu'),strict=True);model.eval();model.requires_grad_(False)
 pred=[];labels=[]
 with torch.no_grad():
  for raw in torch.utils.data.DataLoader(data,batch_size=128,shuffle=False):
   raw=tuple(x.cuda() for x in raw);pred.append(forward_batch(model,raw)[0].view(-1).cpu().numpy());labels.append(raw[3].view(-1).cpu().numpy())
 pred=np.concatenate(pred);y=np.concatenate(labels)
 with np.load(run/'predictions.npz') as saved:
  assert np.array_equal(y,saved['valid_y']);error=float(np.max(np.abs(pred-saved['valid_pred'])));assert error<2e-5,error
 np.savez(dest/'full_replay_predictions.npz',valid_pred=pred,valid_y=y)
 nonzero=y!=0;metrics={'MAE':float(np.mean(np.abs(pred-y))),'MSE229':float(np.mean((pred-y)**2)),'Corr':float(np.corrcoef(pred,y)[0,1]),'Has0_acc2':float(np.mean((pred>=0)==(y>=0))),'Non0_acc2':float(np.mean((pred[nonzero]>=0)==(y[nonzero]>=0))),'Non0_MAE':float(np.mean(np.abs(pred[nonzero]-y[nonzero]))),'Non0_F1':float(author.f1_score(y[nonzero]>=0,pred[nonzero]>=0,average='weighted')),'Acc7':float(author.multiclass_acc(np.clip(pred,-3,3),np.clip(y,-3,3))),'author_batch_mse':float(np.mean([np.mean((pred[:128]-y[:128])**2),np.mean((pred[128:]-y[128:])**2)])),'samples':229,'nonzero_samples':int(nonzero.sum())}
 r={'status':'FULL_CHECKPOINT_STRICT_DISK_RELOAD_AND_OFFICIAL_DEV229_REPLAY_COMPLETE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'node':node,'requested_mode':mode,'effective_mode':effective,'best_epoch':sel['best_epoch'],'full_checkpoint_sha256':sha(cp),'full_checkpoint_bytes':cp.stat().st_size,'base_checkpoint_sha256':sha(base),'addon_sha256':sha(run/'best_addon.pt'),'full_raw_vs_cached_prediction_max_error':error,'unchanged_base_tensor_count':len(unchanged),'allowed_replacements':'donor_feedback tensors only, plus new vector_feedback tensors','metrics':metrics,'full_replay_predictions_sha256':sha(dest/'full_replay_predictions.npz'),'source_sha256':sha(__file__),'test_requested':False,'full_state_tensor_count':len(combined)}
 (dest/'full_checkpoint_receipt.json').write_text(json.dumps(r,indent=2));print('FULL_CHECKPOINT_COMPLETE',node,sel['best_epoch'],cp.stat().st_size,flush=True)
