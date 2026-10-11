"""TRAIN video-isolated scalar teacher. No task-trained checkpoint initialization."""
from pathlib import Path
from types import MethodType,SimpleNamespace
import sys,pickle,json,hashlib,datetime,math
import numpy as np
import torch

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def tensor_sha(values):
 h=hashlib.sha256()
 for n,v in sorted(values.items()):
  h.update(n.encode());h.update(str((tuple(v.shape),str(v.dtype))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def plain_masked(self,input_ids,visual,acoustic,label_ids=None,input_mask=None):
 from encoder_adapter import content_mask,encode_masked
 valid=content_mask(input_mask);hidden=self.model(input_ids,attention_mask=input_mask)[0]
 text=self.LayerNorm_l(self.proj_l(hidden))*valid[...,None]
 def normalized(value,name):
  return ((value-getattr(self,'v6_'+name+'_mean'))/getattr(self,'v6_'+name+'_std'))*getattr(self,'v6_'+name+'_active')*valid[...,None]
 audio=self.proj_a(normalized(acoustic,'audio').transpose(1,2)).permute(2,0,1)
 visual=self.proj_v(normalized(visual,'visual').transpose(1,2)).permute(2,0,1)
 audio=self.LayerNorm_a(encode_masked(self.transa,audio,valid).transpose(0,1))*valid[...,None]
 visual=self.LayerNorm_v(encode_masked(self.transv,visual,valid).transpose(0,1))*valid[...,None]
 pool=lambda x:(x*valid[...,None]).sum(1)/valid.sum(1)[:,None]
 prediction=self.predictor(self.fusion(torch.cat([pool(text),pool(audio),pool(visual)],-1)))
 return prediction,prediction.new_zeros(()),prediction.new_zeros(())

class GuardedTrainData:
 def __init__(self,base,root,fold,author):
  self.author=author;self.examples=pickle.load((base/'assets/mosi.pkl').open('rb'))['train'];assert len(self.examples)==1281
  self.ids={k:np.load(root/f'{k}_{fold}.npy') for k in ['fit','inner','outer']};self.journal=[];self.outer_prediction_authorized=False
  assert all(not set(self.ids[x])&set(self.ids[y]) for x,y in [('fit','inner'),('fit','outer'),('inner','outer')])
 def dataset(self,ids,role):
  ids=np.asarray(ids,dtype=np.int64);assert role in ['fit','inner'];allowed=set(self.ids[role].tolist())
  if not set(ids.tolist())<=allowed:raise PermissionError('ROLE_ROW_GUARD_FORBIDS_OUTER_OR_OTHER_ROLE_LABEL_ACCESS')
  self.journal.append(dict(role=role,rows=ids.tolist(),label_access=True))
  return self.author.get_appropriate_dataset([self.examples[int(i)] for i in ids])
 def outer_inputs(self):
  assert self.outer_prediction_authorized
  ids=self.ids['outer'];self.journal.append(dict(role='outer_prediction_only_after_selected100',rows=ids.tolist(),label_access=False))
  return self.author.get_appropriate_dataset([(self.examples[int(i)][0],np.zeros((1,1),dtype=np.float32),self.examples[int(i)][2]) for i in ids])

def setup(base,root,fold):
 sys.path[:0]=[str(base/'assets'),str(base/'frozen_v5')]
 import run_careflow as helpers,run_control_baseline as control
 from encoder_adapter import fit_statistics,install
 args=SimpleNamespace(repo=base/'assets/CaReFlow',backbone=base/'assets/deberta-v3-base',epochs=100,seed=91818)
 author,arguments=helpers.load_author(args);author.set_random_seed(91818)
 data=GuardedTrainData(base,root,fold,author);fit=data.dataset(data.ids['fit'],'fit');inner=data.dataset(data.ids['inner'],'inner')
 stats=fit_statistics(fit);steps=100*math.ceil(len(fit)/32)
 model,oldopt,oldsched=author.prep_for_training(steps);matched=helpers.pretrained_check(model,args.backbone)
 core=model.dberta
 removed=['reflow_a','reflow_v','reflow_a_b','reflow_v_b','rf_a','rf_v','rf_a_b','rf_v_b']
 for n in removed:delattr(core,n)
 # The custom scalar forward never calls the author's pooler. Remove it
 # before optimizer construction instead of claiming gradients on unused tensors.
 delattr(core,'pooler')
 install(core,stats)
 for enc in [core.transa,core.transv]:enc.embed_positions._float_tensor.zero_()
 core.forward=MethodType(plain_masked,core);core.to(author.DEVICE)
 del oldopt,oldsched
 optimizer,scheduler=control.optimizer_for(author,model,steps)
 return model,optimizer,scheduler,data,fit,inner,stats,matched,arguments

def forward(model,batch):
 ids,visual,audio,y,mask=batch
 return model(ids,visual.squeeze(1),audio.squeeze(1),y,mask)[0].view(-1)

def source_checks(base,root,plan):
 assert sha(base/'assets/mosi.pkl')==plan['dataset_sha256']
 for n,v in plan['source_sha256'].items():assert sha(root/n)==v
 for n,v in plan['asset_sha256'].items():assert sha(base/n)==v

def collect(model,dataset,batch_size=128):
 model.eval();pred=[];labels=[]
 with torch.no_grad():
  for batch in torch.utils.data.DataLoader(dataset,batch_size=batch_size,shuffle=False):
   batch=tuple(x.cuda() for x in batch);pred.append(forward(model,batch).cpu().numpy());labels.append(batch[3].view(-1).cpu().numpy())
 return np.concatenate(pred),np.concatenate(labels)

def row_batch(fit,fit_ids,global_ids):
 lookup={int(v):i for i,v in enumerate(fit_ids)};local=torch.tensor([lookup[int(i)] for i in global_ids],dtype=torch.long)
 return tuple(x[local].cuda() for x in fit.tensors)

def norm_free_sha(model):return tensor_sha({n:v for n,v in model.state_dict().items() if not n.startswith(('dberta.v6_audio_','dberta.v6_visual_'))})

def write_json(path,obj):
 temp=path.with_suffix('.pending.json');temp.write_text(json.dumps(obj,indent=2),encoding='utf-8');temp.replace(path)
