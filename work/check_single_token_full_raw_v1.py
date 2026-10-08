import os
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
import sys,json,hashlib,datetime,numpy as np,torch
base=Path('/data/coding/soft_vector_research_20261005T1220Z');old=Path('/data/coding/finite_task_risk_v1_deployment_20261005T1450Z');new=Path('/data/coding/finite_single_token_precheck_20261005T1525Z');sys.path[:0]=[str(new),str(old),str(base)]
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner,make_full_reference,install_finite_inference
from finite_single_token_reader_v1 import install_for_flow
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(91817)
model,author,helpers=make_full_reference(base,Path('/data/coding/teacher_a_seed91814_best41.pt'))
from encoder_adapter import forward_batch
dataset=helpers.dataset(author,base/'assets/mosi.pkl','train');ids=list(range(31))+[620];raw=tuple(x.cuda() for x in next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(dataset,ids),batch_size=32))))
cache=np.load(base/'teacher_cache_v1/train_cache.npz');b={n:torch.as_tensor(cache[n][ids],device='cuda') for n in ['state','mask','old_context','pooled_state','reference_prediction','y']}
learner=FrozenCoordinateLearner(base,'finite_vector',json.loads((old/'train_scales.json').read_text())).cuda();learner.restore_addon(torch.load(old/'run/shared_phase_addon.pt',map_location='cpu'));install_for_flow(learner.flow);install_for_flow(model.dberta.own_flow)
errors={}
for mode in ['fixed','finite_scalar','finite_vector']:
 install_finite_inference(model.dberta,learner,mode)
 with torch.no_grad():
  expected=learner(b,mode)[0];actual=forward_batch(model,raw)[0].view(-1);changed=forward_batch(model,(*raw[:3],raw[3].flip(0)+20,raw[4]))[0].view(-1)
 err=float((expected-actual).abs().max());assert err<2e-5 and torch.equal(actual,changed)
 errors[mode]=dict(full_vs_cached_error=err,label_replacement_error=0.)
r=dict(status='SINGLE_TOKEN_FULL_RAW_TRAIN32_INCLUDING_WITNESS620_THREE_MODES_CHECKED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reader_source_sha256=hashlib.sha256((new/'finite_single_token_reader_v1.py').read_bytes()).hexdigest(),train_row_ids=ids,modes=errors,dev_requested=False,test_requested=False,scope='Shared10addon on full classifier, frozen params unchanged; analytic reader installed in cached and raw model. Raw TRAIN only, not100formal orDEVselection.')
(new/'full_raw_receipt.json').write_text(json.dumps(r,indent=2));print('SINGLE_TOKEN_FULL_RAW_COMPLETE')
