import os
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
from pathlib import Path
import argparse,datetime,hashlib,json,sys,time
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--base-root',type=Path,required=True);p.add_argument('--checks',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
sys.path.append(str(a.base_root))
from finite_task_risk_runtime_v1 import FrozenCoordinateLearner,make_full_reference,install_finite_inference
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.manual_seed(91817)
start=time.monotonic();model,author,helpers=make_full_reference(a.base_root,Path('/data/coding/teacher_a_seed91814_best41.pt'))
from encoder_adapter import forward_batch
dataset=helpers.dataset(author,a.base_root/'assets/mosi.pkl','train')
raw=tuple(v.cuda() for v in next(iter(torch.utils.data.DataLoader(dataset,batch_size=32,shuffle=False))))
cache=np.load(a.base_root/'teacher_cache_v1/train_cache.npz',allow_pickle=False)
b={n:torch.as_tensor(cache[n][:32],device='cuda') for n in ['state','mask','old_context','pooled_state','reference_prediction','y']}
scales=json.loads((a.checks/'train_scales.json').read_text());learner=FrozenCoordinateLearner(a.base_root,'fixed',scales).cuda()
learner.restore_addon(torch.load(a.checks/'after20_addon.pt',map_location='cpu'))
errors={}
for mode in ['fixed','finite_scalar','finite_vector']:
    tick=time.monotonic();install_finite_inference(model.dberta,learner,mode)
    with torch.no_grad():
        expected=learner(b,mode)[0];actual=forward_batch(model,raw)[0].view(-1)
        changed=forward_batch(model,(*raw[:3],raw[3].flip(0)+20,raw[4]))[0].view(-1)
    error=float((expected-actual).abs().max());label_error=float((actual-changed).abs().max())
    assert error<2e-5 and torch.equal(actual,changed),(mode,error,label_error)
    assert all(p.grad is None for p in model.parameters())
    errors[mode]=dict(full_vs_cached_prediction_max_error=error,label_replacement_max_error=label_error,seconds=time.monotonic()-tick)
r=dict(status='FINITE_RISK_FULL_RAW_TRAIN32_THREE_MODE_REPLAY_CHECKED_NOT_FORMAL_TRAINING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),modes=errors,split='train',rows=32,dev_requested=False,test_requested=False,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),addon_sha256=hashlib.sha256((a.checks/'after20_addon.pt').read_bytes()).hexdigest(),scales_sha256=hashlib.sha256((a.checks/'train_scales.json').read_bytes()).hexdigest(),peak_allocated_bytes=torch.cuda.max_memory_allocated(),elapsed_seconds=time.monotonic()-start,argv=sys.argv,checkpoint_contract='Eventual new complete model must include finite controller buffers/donor/head tensors and strict reload/official229 replay;2MBaddon not fullweight.')
assert not a.out.exists();a.out.write_text(json.dumps(r,indent=2));print('FINITE_FULL_RAW_INFERENCE_CHECK_COMPLETE',flush=True)
