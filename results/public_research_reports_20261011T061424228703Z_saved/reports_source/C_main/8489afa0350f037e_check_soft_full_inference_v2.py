import os
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,json,datetime,hashlib
import numpy as np
import torch
from soft_vector_runtime_v2 import FrozenCoordinateLearner,make_full_reference,install_vector_inference
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--addon',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(91815)
model,author,helpers=make_full_reference(a.root,Path('/data/coding/teacher_a_seed91814_best41.pt'))
from encoder_adapter import forward_batch
data=helpers.dataset(author,a.root/'assets/mosi.pkl','train');raw=next(iter(torch.utils.data.DataLoader(data,batch_size=32,shuffle=False)))
raw=tuple(v.cuda() for v in raw)
jac=np.load(a.root/'label_free_jacobian_v1/train_jacobian.npz');cache=np.load(a.root/'teacher_cache_v1/train_cache.npz');b={k:torch.as_tensor(cache[k][:32],device='cuda') for k in cache.files if k!='row_id'}
b['reference_jacobian']=torch.as_tensor(jac['reference_jacobian'][:32],device='cuda')
learner=FrozenCoordinateLearner(a.root,'fixed',.1).cuda();learner.restore_addon(torch.load(a.addon,map_location='cpu'))
errors={}
for mode in ['fixed','scalar','soft_projected']:
 install_vector_inference(model.dberta,learner,mode)
 with torch.no_grad():
  expected=learner(b,mode)[0];actual=forward_batch(model,raw)[0].view(-1)
  changed=forward_batch(model,(*raw[:3],raw[3].flip(0)+10,raw[4]))[0].view(-1)
 errors[mode]={'full_vs_cached_prediction_max_error':float((actual-expected).abs().max()),'label_replacement_max_error':float((actual-changed).abs().max())}
 assert errors[mode]['full_vs_cached_prediction_max_error']<2e-5 and torch.equal(actual,changed)
 assert all(v.grad is None for v in model.parameters())
r={'status':'FULL_CLASSIFIER_LABEL_FREE_VECTOR_HOOK_THREE_MODES_REPLAY_CHECKED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'addon_sha256':hashlib.sha256(a.addon.read_bytes()).hexdigest(),'rows':32,'split':'train','modes':errors,'test_requested':False,'teacher_parameter_gradients_none':True,'checkpoint_requirement':'Final combined full state must replace original donor tensors and register vector_feedback tensors, strict-load and replay official229; addon alone is not full checkpoint.'}
a.out.write_text(json.dumps(r,indent=2),encoding='utf-8');print('FULL_INFERENCE_CHECK_COMPLETE',flush=True)
