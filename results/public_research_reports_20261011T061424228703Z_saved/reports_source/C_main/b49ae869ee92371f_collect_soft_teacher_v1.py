"""Frozen A TRAIN-gradient collection and DEV replay; no TEST split requested."""
import os
os.environ['HF_HUB_OFFLINE']=os.environ['TRANSFORMERS_OFFLINE']='1'
from pathlib import Path
from types import MethodType,SimpleNamespace
import argparse,datetime,hashlib,importlib,json,subprocess,sys,time
import numpy as np
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--expected-gpu-uuid',required=True);p.add_argument('--out',type=Path,required=True);c=p.parse_args()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def tensor_sha(model):
 h=hashlib.sha256()
 for n,v in sorted(model.state_dict().items()):
  h.update(n.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
q=lambda args:subprocess.check_output(['nvidia-smi',*args],text=True).strip()
gpu=q(['--query-gpu=uuid,memory.used','--format=csv,noheader,nounits']);assert gpu.split(',')[0].strip()==c.expected_gpu_uuid and int(gpu.split(',')[-1])<=16
assert not q(['--query-compute-apps=pid','--format=csv,noheader,nounits'])
assert not c.out.exists();c.out.mkdir()
start=time.monotonic();torch.set_num_threads(2)
root=c.root.resolve();base=root/'assets';frozen=root/'frozen_v5';meta=root/'teacher_a_metadata'
protocol=json.loads((meta/'protocol.json').read_text());selection=json.loads((meta/'selection.json').read_text())
assert selection['best_epoch']==41 and protocol['mode']=='none' and selection['epochs']==100
assert sha(c.checkpoint)==selection['checkpoint_sha256']
for group,folder in [('own_source_sha256',frozen),('helper_source_sha256',base),('author_source_sha256',base/'CaReFlow'),('backbone_sha256',base/'deberta-v3-base')]:
 for name,digest in protocol[group].items():assert sha(folder/name)==digest
assert sha(base/'mosi.pkl')==protocol['data_sha256']
sys.path[:0]=[str(frozen),str(base)]
from counterfactual_flow_model import WholeStateFlow
from encoder_adapter import install,forward_v6,forward_batch
helpers=importlib.import_module('run_careflow')
cli=SimpleNamespace(backbone=base/'deberta-v3-base',repo=base/'CaReFlow',epochs=100,seed=91814)
author,args=helpers.load_author(cli);author.set_random_seed(91814)
model,optimizer,scheduler=author.prep_for_training(4000);del optimizer,scheduler
core=model.dberta
for name in ('reflow_a','reflow_v','reflow_a_b','reflow_v_b','rf_a','rf_v','rf_a_b','rf_v_b'):delattr(core,name)
core.own_flow=WholeStateFlow('none').to(author.DEVICE)
stats={m:{k:torch.tensor(v,dtype=torch.bool if k=='active' else torch.float32) for k,v in fields.items()} for m,fields in protocol['normalization_statistics'].items()}
install(core,stats);core.to(author.DEVICE);core.forward=MethodType(forward_v6,core)
model.load_state_dict(torch.load(c.checkpoint,map_location='cpu'),strict=True)
model.eval();model.requires_grad_(False);flow=core.own_flow;flow.set_epoch(41)
before=tensor_sha(model);original_update=flow.update_context;captured={}
def capture(self,old,relation):
 if self._calls==0:
  captured.update(state=relation['stage_states'].detach(),pooled_state=relation['slots'].mean(2).detach(),old_context=old.detach(),mask=self.valid.bool().detach())
 return original_update(old,relation)
flow.update_context=MethodType(capture,flow)
def terminal(state,mask,context):
 flow.valid=mask.to(state.dtype)
 valid=flow.valid[:,None,:,None]
 velocity=torch.stack([flow.forward_fields[m](state[:,m],.5,context[:,m]) for m in range(3)],1)
 final=(state+.5*velocity)*valid
 return flow.read_prediction(final,flow.reader(final,mask),lambda x:core.predictor(core.fusion(x))).view(-1)
reports={};first_mechanism=None
for split,count in [('train',1281),('dev',229)]:
 data=helpers.dataset(author,base/'mosi.pkl',split);assert len(data)==count
 loader=torch.utils.data.DataLoader(data,batch_size=32,shuffle=False)
 chunks={};predictions=[]
 for index,batch in enumerate(loader):
  batch=tuple(t.to(author.DEVICE) for t in batch);neutral=(*batch[:3],torch.zeros_like(batch[3]),batch[4])
  with torch.no_grad():prediction=forward_batch(model,neutral)[0].view(-1)
  state=captured['state'];mask=captured['mask'];old=captured['old_context'];pool=captured['pooled_state']
  context=(.5*old).detach().requires_grad_(split=='train')
  with torch.set_grad_enabled(split=='train'):p0=terminal(state,mask,context)
  replay=(p0.detach()-prediction).abs().max().item();assert replay<2e-5,replay
  values={'state':state,'mask':mask,'old_context':old,'pooled_state':pool,'reference_prediction':p0.detach(),'y':batch[3].view(-1)}
  if split=='train':
   gradient=torch.autograd.grad((p0-batch[3].view(-1)).square().sum(),context)[0].detach()
   assert torch.isfinite(gradient).all() and gradient.square().sum()>0
   values['teacher_gradient']=gradient
  if first_mechanism is None:
   assert split=='train'
   with torch.no_grad():changed=forward_batch(model,(*batch[:3],batch[3].flip(0)+10,batch[4]))[0].view(-1)
   assert torch.equal(changed,prediction),'Label-fed inference'
   k=4;cx=(.5*old[:k]).detach().requires_grad_();pb=terminal(state[:k],mask[:k],cx)
   g0=torch.autograd.grad(pb[0],cx,retain_graph=True)[0]
   assert g0[1:].abs().max()==0,'Cross-sample gradient'
   gb=torch.autograd.grad((pb-batch[3].view(-1)[:k]).square().sum(),cx)[0]
   errs=[]
   for j in range(k):
    cj=(.5*old[j:j+1]).detach().requires_grad_();pj=terminal(state[j:j+1],mask[j:j+1],cj)
    gj=torch.autograd.grad((pj-batch[3].view(-1)[j:j+1]).square().sum(),cj)[0]
    errs.append((gj-gb[j:j+1]).abs().max().item())
   assert max(errs)<2e-5,errs
   assert all(v.grad is None for v in model.parameters())
   first_mechanism={'label_replacement_max_error':0.,'cross_sample_prediction_gradient_max':float(g0[1:].abs().max()),'per_example_vs_batch_sum_gradient_max_error':max(errs),'teacher_parameter_gradients_none':True,'reference_replay_max_error':replay}
  for name,v in values.items():chunks.setdefault(name,[]).append(v.detach().cpu().numpy())
  predictions.append(prediction.detach().cpu().numpy())
  if index%10==0:print(json.dumps({'stage':'cache','split':split,'batch':index,'rows_processed':min((index+1)*32,count)}),flush=True)
 arrays={n:np.concatenate(v) for n,v in chunks.items()};arrays['row_id']=np.arange(count,dtype=np.int64)
 if split=='train':
  rms=np.sqrt(np.mean(np.square(arrays['teacher_gradient'].astype(np.float64)),axis=(0,2)))
  assert rms.shape==(3,) and np.isfinite(rms).all() and (rms>=1e-8).all()
  np.save(c.out/'train_gradient_rms.npy',rms.astype(np.float32),allow_pickle=False)
 else:
  with np.load(meta/'predictions.npz',allow_pickle=False) as saved:
   assert np.array_equal(arrays['y'],saved['valid_y'])
   err=np.max(np.abs(arrays['reference_prediction']-saved['valid_pred']));assert err<2e-5,float(err)
  first_mechanism['official_229_saved_A_default_replay_max_error']=float(err)
 np.savez(c.out/(split+'_cache.npz'),**arrays)
 reports[split]={'rows':count,'state_shape':list(arrays['state'].shape),'fields':sorted(arrays),'file_sha256':sha(c.out/(split+'_cache.npz'))}
flow.update_context=original_update
after=tensor_sha(model);assert before==after
cpu=lambda m:{k:v.detach().cpu() for k,v in m.state_dict().items()}
torch.save({'flow':cpu(flow),'fusion':cpu(core.fusion),'predictor':cpu(core.predictor),'source_checkpoint_sha256':selection['checkpoint_sha256']},c.out/'frozen_terminal.pt')
r={'status':'FROZEN_A_TRAIN_GRADIENT_AND_DEV_REPLAY_COLLECTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'argv':sys.argv,'gpu_uuid':c.expected_gpu_uuid,'compute':q(['--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),'source_sha256':sha(__file__),'base_checkpoint_sha256':selection['checkpoint_sha256'],'base_model_tensor_sha_before':before,'base_model_tensor_sha_after':after,'splits':reports,'mechanism':first_mechanism,'gradient_rms':rms.tolist(),'elapsed_seconds':time.monotonic()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'teacher_scope':'Already full-TRAIN-fitted, DEV-selected frozen A; in-sample mechanism teacher, not crossfit or unbiased validation. Fixed deterministic encoder/first-flow/terminal coordinates.','test_split_requested':False,'dev_label_used_for_gradient_target':False,'formal_training_started':False}
r['files']={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in c.out.iterdir() if p.is_file()}
(c.out/'collection.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
print('TEACHER_COLLECTION_COMPLETE '+json.dumps({k:r[k] for k in ['status','elapsed_seconds','peak_allocated_bytes','mechanism','gradient_rms']}),flush=True)
